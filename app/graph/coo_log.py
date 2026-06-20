import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class COOLog:
    rows: List[int] = field(default_factory=list)
    cols: List[int] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def append(self, row: int, col: int, value: float = 1.0):
        self.rows.append(row)
        self.cols.append(col)
        self.values.append(value)

    def append_bidirectional(self, node_a: int, node_b: int, value: float = 1.0):
        self.append(row=node_a, col=node_b, value=value)
        self.append(row=node_b, col=node_a, value=value)

    def __len__(self) -> int:
        return len(self.rows)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.array(self.rows, dtype=np.int32),
            np.array(self.cols, dtype=np.int32),
            np.array(self.values, dtype=np.float32),
        )
