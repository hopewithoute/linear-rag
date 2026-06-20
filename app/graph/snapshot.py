import os
import json
import numpy as np
from scipy import sparse
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class GraphSnapshot:
    snapshot_id: int
    version: str
    matrix: sparse.csr_matrix
    chunk_count: int
    entity_count: int
    active_mask: np.ndarray
    chunk_id_map: Dict[int, int] = field(default_factory=dict)


_active_snapshot: Optional[GraphSnapshot] = None


def get_active_snapshot() -> Optional[GraphSnapshot]:
    return _active_snapshot


def set_active_snapshot(snapshot: GraphSnapshot):
    global _active_snapshot
    _active_snapshot = snapshot


def save_snapshot(snapshot: GraphSnapshot, snapshot_dir: str):
    path = Path(snapshot_dir)
    path.mkdir(parents=True, exist_ok=True)

    matrix_path = path / f"snapshot_{snapshot.snapshot_id}_matrix.npz"
    sparse.save_npz(str(matrix_path), snapshot.matrix)

    meta_path = path / f"snapshot_{snapshot.snapshot_id}_meta.json"
    meta = {
        "snapshot_id": snapshot.snapshot_id,
        "version": snapshot.version,
        "chunk_count": snapshot.chunk_count,
        "entity_count": snapshot.entity_count,
        "chunk_id_map": {str(k): v for k, v in snapshot.chunk_id_map.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    mask_path = path / f"snapshot_{snapshot.snapshot_id}_mask.npy"
    np.save(str(mask_path), snapshot.active_mask)


def load_snapshot(snapshot_id: int, snapshot_dir: str) -> Optional[GraphSnapshot]:
    path = Path(snapshot_dir)

    matrix_path = path / f"snapshot_{snapshot_id}_matrix.npz"
    meta_path = path / f"snapshot_{snapshot_id}_meta.json"
    mask_path = path / f"snapshot_{snapshot_id}_mask.npy"

    if not matrix_path.exists() or not meta_path.exists():
        return None

    matrix = sparse.load_npz(str(matrix_path))
    with open(meta_path) as f:
        meta = json.load(f)

    active_mask = np.load(str(mask_path)) if mask_path.exists() else np.ones(meta["chunk_count"], dtype=bool)

    return GraphSnapshot(
        snapshot_id=meta["snapshot_id"],
        version=meta["version"],
        matrix=matrix,
        chunk_count=meta["chunk_count"],
        entity_count=meta["entity_count"],
        active_mask=active_mask,
        chunk_id_map={int(k): v for k, v in meta.get("chunk_id_map", {}).items()},
    )
