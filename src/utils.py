from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def infer_common_id_column(left_columns: list[str], right_columns: list[str]) -> str:
    candidate_order = [
        "customer_id",
        "user_id",
        "client_id",
        "id",
        "profile_id",
    ]

    left_lower = {c.lower(): c for c in left_columns}
    right_lower = {c.lower(): c for c in right_columns}

    for cand in candidate_order:
        if cand in left_lower and cand in right_lower:
            return left_lower[cand]

    intersection = [c for c in left_columns if c in right_columns]
    if intersection:
        return intersection[0]

    raise ValueError(
        "No shared key found between datasets. Add a shared key (e.g., customer_id)."
    )
