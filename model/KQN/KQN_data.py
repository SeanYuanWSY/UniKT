"""Data adapter for the KQN baseline."""

from typing import Any

import torch
from torch.utils.data import Dataset
from typing_extensions import override

from utils.core import get_logger
from utils.data_process import DataSource
from utils.model_data import SkillModelData

logger = get_logger(__name__)


class KQNDataset(Dataset):
    """KC-level sequence dataset for KQN.

    Training/validation items:
        ``(concept, response, mask, question)``.

    Shapes:
        concept: [S] 0-based KC ids; padding values are ignored by ``mask``.
        response: [S] binary labels.
        mask: [S] true for valid sequence positions.
        question: [S] question ids retained for batch contract and analysis;
            KQN itself does not consume question ids.
    """

    def __init__(self, concepts, responses, masks, questions):
        self.concepts = concepts
        self.responses = responses
        self.masks = masks
        self.questions = questions

    def __len__(self) -> int:
        return len(self.concepts)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.concepts[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.bool),
            torch.tensor(self.questions[idx], dtype=torch.long),
        )


class KQNModelData(SkillModelData):
    """Build KC sequence data for KQN."""

    def __init__(self, data_src: DataSource):
        super().__init__(data_src)

    @override
    def prepare_data(self, rc: Any) -> tuple:
        fold_idx = rc.data.fold if rc.data.fold >= 0 else None

        concepts, responses, masks, _, questions = self.build_sequence_data()

        if fold_idx is None:
            raise ValueError("K-fold cross-validation is not enabled.")

        kfold_n_splits = self.data_src.get_metadata("kfold_n_splits")
        if fold_idx < 0 or fold_idx >= kfold_n_splits:
            raise ValueError(
                f"fold_idx {fold_idx} is out of range [0, {kfold_n_splits})"
            )
        logger.info(
            f"Using K-fold cross-validation: fold {fold_idx + 1}/{kfold_n_splits}"
        )

        train_data, val_data, _ = self.split_kfold_data(
            concepts,
            responses,
            masks,
            questions,
            fold_idx=fold_idx,
        )

        window_test_data = self.create_windowlate_iterable_dataset()

        train_dataset = KQNDataset(
            train_data[0], train_data[1], train_data[2], train_data[3]
        )
        val_dataset = KQNDataset(val_data[0], val_data[1], val_data[2], val_data[3])
        test_dataset = window_test_data.create_dataloader(
            batch_size=rc.model.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )

        logger.debug(
            f"KQN data prepared: train={len(train_dataset)}, val={len(val_dataset)}, "
            f"test(window)={len(test_dataset)}"
        )
        return train_dataset, val_dataset, test_dataset
