"""AxisKT case analyzer.

The knowledge state is the local branch's per-KC private scan state,
captured by wrapping :meth:`AxisKT._scan_states` and hooking ``local_write``
— the shared pipeline of both execution paths. Per-skill mastery is the
mean over every question covering the skill of the IRT head applied to
``[per-KC standing state, question event embedding]`` (the global conv
state is excluded: it changes at every column, so including it would make
a frozen skill's mastery curve wiggle between its own events).

Alignment convention: row ``r`` of the collected DataFrame predicts
``response[r + 1]``; the exported knowledge state is the post-event state
after events ``<= r + 1``.
"""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.case_analysis.base_analyzer import BaseCaseAnalyzer
from utils.core import get_logger, register_analyzer
from utils.data_process import DataSource
from utils.training.runtime_components import RuntimeComponents

from .AxisKT_data import (
    AxisKTDataset,
    AxisKTModelData,
    axiskt_packed_collate_fn,
    build_axiskt_model,
    use_compact_global,
)

logger = get_logger(__name__)


class _UserAwareAxisKTDataset(Dataset):
    """AxisKTDataset wrapper that annotates each row with its user id.

    Split rows are subsequences, not users, so the dataset index is not a
    user id; ``user_ids`` carries the real student id per row. The extra
    trailing element is transparent to :func:`axiskt_packed_collate_fn`.
    """

    def __init__(self, dataset: AxisKTDataset, user_ids: np.ndarray):
        self.dataset = dataset
        self.user_ids = np.asarray(user_ids)

    def __getitem__(self, index: int):
        return (*self.dataset[index], int(self.user_ids[index]))

    def __len__(self) -> int:
        return len(self.dataset)


def _analyzer_collate_fn(batch):
    """Collate a batch, prepending the stacked user ids."""
    users = torch.tensor([row[-1] for row in batch], dtype=torch.long)
    return (users, *axiskt_packed_collate_fn(batch))


@register_analyzer("AxisKT")
class AxisKTAnalyzer(BaseCaseAnalyzer):
    """AxisKT-specific case analyzer for inference and visualization."""

    def build_components(self, rc, data_src: DataSource) -> RuntimeComponents:
        """Assemble the AxisKT model and user-annotated val dataset."""
        model_data = AxisKTModelData(data_src)
        _, val_data, _, extra = model_data.prepare_data(rc)

        self.num_questions = data_src.get_metadata("num_questions")
        self.num_skills = data_src.get_metadata("num_skills")
        # CPU copies of the question-KC table for skill extraction.
        self.question_skill_ids_np = np.asarray(extra["question_skill_ids"])
        self.question_skill_mask_np = np.asarray(extra["question_skill_mask"])
        # Every question covering each skill, used for the mean mastery
        # readout over all associated questions.
        self._skill_questions = self._build_skill_questions()

        model = build_axiskt_model(rc, data_src, extra)

        val_dataset = _UserAwareAxisKTDataset(val_data, extra["user_ids"]["val"])
        return RuntimeComponents(
            model=model, val_data=val_dataset, collate_fn=_analyzer_collate_fn
        )

    def forward_pass(
        self, batch_data: tuple[torch.Tensor, ...]
    ) -> dict[str, torch.Tensor]:
        """AxisKT forward pass for inference.

        Args:
            batch_data: Tuple of (users, questions, responses, times, mask,
                kc_order, valid_idx)

        Returns:
            Dictionary with y_hat (flattened logits), y_label (flattened),
            y_predict, and knowledge_states [B, S, num_skills] per-skill
            mastery pseudo-probabilities.
        """
        users, questions, responses, times, mask, kc_order, _, valid_idx = batch_data
        compact_global = use_compact_global(mask)
        questions = self._move_tensor_to_device(questions)
        responses = self._move_tensor_to_device(responses)
        times = self._move_tensor_to_device(times)
        mask = self._move_tensor_to_device(mask)
        kc_order = self._move_tensor_to_device(kc_order)
        valid_idx = self._move_tensor_to_device(valid_idx)

        logits_full, captured = self._forward_with_captured_states(
            questions, responses, times, mask, kc_order, compact_global
        )

        logits = logits_full[:, :-1].flatten()[valid_idx]
        labels = responses.float()[:, 1:].flatten()[valid_idx]
        logits, labels = self._handle_empty_batch(logits, labels)

        knowledge_states = self._per_skill_mastery(
            captured, questions.size(0), questions.size(1)
        )

        return {
            "y_hat": logits,
            "y_label": labels,
            "y_predict": self._generate_binary_predictions(logits, threshold=0.0),
            "knowledge_states": knowledge_states,
        }

    def _forward_with_captured_states(
        self,
        questions: torch.Tensor,
        responses: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        kc_order: torch.Tensor,
        compact_global: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the model forward, capturing per-KC states and features.

        Wraps :meth:`AxisKT._scan_states` (the shared pipeline of both
        execution paths) and hooks ``local_write``; the patches are scoped
        to this call and removed in a ``finally`` block.
        """
        captured: dict[str, torch.Tensor] = {}
        model = self.model
        orig_scan = model._scan_states
        write_hook = model.local_write.register_forward_hook(
            lambda mod, inp, out: captured.__setitem__(
                "local_write_out", out.detach().float()
            )
        )

        def hooked_scan(q, r, t, m, q_features=None, kc_order=None, **kwargs):
            out = orig_scan(q, r, t, m, q_features, kc_order, **kwargs)
            captured["packed_state"] = out[0].detach().float()
            captured["packed_skill"] = out[2].detach()
            captured["packed_valid"] = out[3].detach()
            captured["packed_pos"] = out[4].detach()
            return out

        model._scan_states = hooked_scan
        try:
            use_amp = bool(self.run_config.model.amp)
            with torch.autocast(
                device_type=self.device_.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                logits_full = model(
                    questions,
                    responses,
                    times,
                    mask,
                    kc_order=kc_order,
                    compact_global=compact_global,
                )
            if use_amp:
                logits_full = logits_full.float()
        finally:
            model._scan_states = orig_scan
            write_hook.remove()
        return logits_full, captured

    def _build_skill_questions(self) -> list[np.ndarray]:
        """Return one list of question ids per skill: every question covering it.

        A question covers a skill when the skill is listed among the
        question's non-masked KC slots.
        """
        ids = self.question_skill_ids_np
        mask = self.question_skill_mask_np
        return [
            np.asarray(
                [q for q in range(len(ids)) if k in ids[q][mask[q]].tolist()],
                dtype=np.int64,
            )
            for k in range(self.num_skills)
        ]

    def _per_skill_mastery(
        self,
        captured: dict[str, torch.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Model-predicted per-skill mastery at every position.

        Returns:
            Tensor [B, S, num_skills] in [0, 1]: ``mastery[b, t, k]`` is the
            mean over every question covering skill ``k`` of the IRT head
            applied to ``[standing state of k after events <= t, question
            event embedding]`` (global term excluded). Columns before a
            skill's first occurrence use its learned initial state; NaN
            only when the local branch is ablated.
        """
        model = self.model
        device = self.device_
        if not model.use_local or "packed_state" not in captured:
            logger.warning(
                "Local branch ablated or not executed; returning NaN knowledge states"
            )
            return torch.full(
                (batch_size, seq_len, self.num_skills),
                float("nan"),
                device=device,
            )

        h = model.hidden_dim
        ability = model.ability_head
        w = ability.weight[0]  # [3h]
        bias = ability.bias  # scalar; included to match the model's own head
        packed_skill = captured["packed_skill"]
        packed_pos = captured["packed_pos"]
        packed_valid = captured["packed_valid"]
        # Standing state = captured pre-state + the write bias (tanh of the
        # captured linear output).
        standing = captured["packed_state"] + torch.tanh(captured["local_write_out"])
        # Per-skill initial state (the exclusive scan's segment start) for
        # columns before the skill's first occurrence.
        skill_range = torch.arange(self.num_skills, device=device)
        init_states = torch.tanh(model.local_init(model.skill_embed(skill_range)))

        mastery = torch.full(
            (batch_size, seq_len, self.num_skills), float("nan"), device=device
        )
        rows = torch.arange(batch_size, device=device)
        for k in range(self.num_skills):
            q_ids = torch.as_tensor(self._skill_questions[k], device=device)
            if q_ids.numel() == 0:
                continue
            occ = packed_valid & (packed_skill == k)
            b_idx, p_idx = occ.nonzero(as_tuple=True)
            positions = packed_pos[b_idx, p_idx]

            # Per-column standing state: scatter the post-write states at
            # their occurrence positions, then forward-fill along time;
            # columns before the first occurrence keep the initial state.
            state = torch.full((batch_size, seq_len, h), float("nan"), device=device)
            state[b_idx, positions] = standing[b_idx, p_idx]
            marker = torch.where(
                torch.isfinite(state[:, :, 0]),
                torch.arange(seq_len, device=device).view(1, -1).expand(batch_size, -1),
                torch.full((batch_size, seq_len), -1, device=device),
            )
            last = torch.cummax(marker, dim=1).values
            init_k = init_states[k].view(1, 1, h).expand(batch_size, seq_len, h)
            state = torch.where(
                (last >= 0).unsqueeze(-1),
                state[
                    rows.unsqueeze(1).expand(batch_size, seq_len),
                    last.clamp_min(0),
                ],
                init_k,
            )

            # Linear head: theta splits as W_s·s + W_e·e + b; the global
            # term is dropped (see the docstring).
            base = torch.nn.functional.linear(state, w[h : 2 * h].view(1, h)).squeeze(
                -1
            )

            # Event embedding per question of skill k, using k's own skill
            # embedding as in the model's event encoder.
            skill_k = torch.tensor(k, device=device)
            emb_k = (
                model._question_vector(q_ids)
                + model.skill_embed(skill_k).unsqueeze(0)
                + model.question_diff(q_ids) * model.skill_change(skill_k).unsqueeze(0)
            )  # [Q_k, H]
            beta_k = model.question_diff(q_ids).squeeze(-1)  # [Q_k]
            emb_logit = (
                torch.nn.functional.linear(emb_k, w[2 * h :].view(1, h)).squeeze(-1)
                + bias
            )  # [Q_k]

            # Mean P(correct) over the skill's questions.
            chunk = 256
            mean_parts = []
            for start in range(0, q_ids.numel(), chunk):
                el = emb_logit[start : start + chunk]
                bt = beta_k[start : start + chunk]
                logit = model.irt_disc * (
                    base.unsqueeze(-1) + el.view(1, 1, -1) - bt.view(1, 1, -1)
                )
                mean_parts.append(torch.sigmoid(logit).mean(dim=-1))
            mastery[:, :, k] = torch.stack(mean_parts).mean(dim=0)
        return mastery

    def extract_case_data(self, batch_data: Any, outputs: dict) -> dict:
        """Extract case data from batch outputs.

        Args:
            batch_data: Raw batch tuple (users, questions, responses, times,
                mask, kc_order, valid_idx)
            outputs: Output dict from forward_pass

        Returns:
            Dictionary with extracted data (all flattened to valid positions)
        """
        users, questions, responses, times, mask, kc_order, _, _ = batch_data

        seq_len = questions.shape[1]
        # Row r predicts response[r + 1]; its knowledge state is the
        # post-event mastery at position t+1 (the state after events <= t+1).
        masks_bool = mask.bool()
        adjacent = masks_bool[:, :-1] & masks_bool[:, 1:]
        valid_pair = adjacent.view(-1).nonzero(as_tuple=True)[0]
        row_idx = valid_pair // (seq_len - 1)
        t = valid_pair % (seq_len - 1)
        flat = row_idx * seq_len + t + 1

        question_ids_flat = questions.view(-1)[flat].cpu().numpy()
        user_ids_flat = users[row_idx].cpu().numpy()

        knowledge_states = outputs["knowledge_states"]
        num_skills = knowledge_states.shape[-1]
        knowledge_states_flat = (
            knowledge_states.view(-1, num_skills)[flat].cpu().numpy()
        )

        return {
            "user_ids": user_ids_flat,
            "question_ids": question_ids_flat,
            "skills": self._get_all_skills_for_questions(question_ids_flat),
            "labels": outputs["y_label"].cpu().numpy(),
            "predictions": outputs["y_predict"].cpu().numpy(),
            "logits": outputs["y_hat"].cpu().numpy(),
            "mask": masks_bool.view(-1)[flat].cpu().numpy(),
            "knowledge_states": knowledge_states_flat,
        }

    def _get_all_skills_for_questions(
        self, question_ids: np.ndarray
    ) -> list[list[int]]:
        """Get all skills for each question; ``[0]`` if none found."""
        skills_list = []
        for q_id in question_ids:
            slots = np.where(self.question_skill_mask_np[q_id])[0]
            question_skills = self.question_skill_ids_np[q_id][slots].tolist()
            skills_list.append(question_skills if question_skills else [0])
        return skills_list


__all__ = ["AxisKTAnalyzer"]
