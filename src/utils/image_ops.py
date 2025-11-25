from pathlib import Path
from typing import Union

import cv2
import numpy as np


def load_image_bgr(path: Union[str, Path]) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_COLOR)
