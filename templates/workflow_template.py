from scripts.model_inference import model_inference
from typing import Callable
import pandas as pd

class Workflow:
    async def __call__(self) -> pd.DataFrame:
        pass