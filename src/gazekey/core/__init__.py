import numpy as np
from numpy.typing import NDArray

F32 = NDArray[np.float32]
F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
U8 = NDArray[np.uint8]

__all__ = ["F32", "F64", "I64", "U8"]
