"""KC name tables for every dataset in the transfer study.

Every UniKT dataset builds ``_id_mappings["skill"]`` = {skill_string -> dense_id}
during transform; replaying load/clean/transform (in memory, no save) recovers
the exact mapping the frozen models were trained with. assistments09
additionally joins human-readable skill names from its raw CSV.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import polars as pl  # noqa: E402

from utils.data_process import get_data_source  # noqa: E402


class _General:
    seed = 42
    device = "cpu"


def _make_rc(dataset: str, data_base_path: str = "./data"):
    from utils.config import RunDataConfig

    class _RC:
        data = RunDataConfig(dataset=dataset, data_base_path=data_base_path)
        general = _General()

    return _RC()


def build_kc_table(dataset: str, data_base_path: str = "./data") -> dict[int, str]:
    """dense skill id -> display name for a dataset (replays its own transform).

    Returns {} if the dataset has no processed data yet.
    """
    ds = get_data_source(_make_rc(dataset, data_base_path))
    ds.load_src_data()
    ds.clean_raw_data()
    ds.transform_data()
    skill_map: dict[str, int] = ds._id_mappings["skill"]  # noqa: SLF001

    if dataset == "assistments09":
        pairs = ds.raw_data.select(["skill_id", "skill_name"]).unique().collect()
        part_to_name: dict[str, str] = {}
        for sid, sname in zip(
            pairs["skill_id"].to_list(), pairs["skill_name"].to_list()
        ):
            if sid is None or sname is None:
                continue
            id_parts = str(sid).split("_")
            name_parts = str(sname).split("_")
            if len(id_parts) == len(name_parts):
                for a, b in zip(id_parts, name_parts):
                    part_to_name.setdefault(a, b.strip())
            elif len(id_parts) == 1:
                part_to_name.setdefault(str(sid), str(sname).strip())
        return {
            dense: (part_to_name.get(skill_str, "").strip() or f"KC {skill_str}")
            for skill_str, dense in skill_map.items()
        }
    # assistments17 / junyi2015 / ednet_kt1: the skill string IS the name
    return {dense: str(skill_str) for skill_str, dense in skill_map.items()}


def kc_table_markdown(table: dict[int, str]) -> str:
    lines = ["| KC ID | 名称 |", "|---|---|"]
    for dense in sorted(table):
        lines.append(f"| {dense} | {table[dense]} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    args = ap.parse_args()
    t = build_kc_table(args.dataset)
    print(f"{args.dataset}: {len(t)} KCs")
    for k in sorted(t)[:10]:
        print(k, t[k])
