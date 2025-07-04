import click

from scripts.evaluator import Evaluator, Range, EvaluationType
from templates.pipeline_template import Pipeline


class SamplePipeline(Pipeline):
    async def __call__(self, task_description: str) -> str:
        code ="""import os
from PIL import Image
import pandas as pd
import base64

from utils.cost_manager import CostManager
from configs.models_config import ModelsConfig
from templates.zeroshot_prompt import ZERO_SHOT_PROMPT, parse_prediction

from scripts.model_inference import model_inference
from provider.llm_provider_registry import create_llm_instance

llm = create_llm_instance(ModelsConfig.default().get("gpt-4o-mini"))
llm.cost_manager = CostManager()

class Workflow:
    async def __call__(self, input_folder: str) -> pd.DataFrame:
        with open(os.path.join(input_folder, "task_description.txt"), "r", encoding="utf-8") as f:
            description = f.read()
        results = {}


        df = pd.DataFrame(list(results.items()), columns=["id", "prediction"])
        return df"""
        return code

@click.command()
@click.option("--range", type=click.Choice(Range.values()), required=True)
@click.option("--result_folder_name", type=str, default=None, required=True)
@click.option("--folder_name", type=str, default=None, required=False)
@click.option("--evaluation_type", type=click.Choice(EvaluationType.values()), required=False, default=EvaluationType.CALCULATE_SCORE.name)
def evaluate(range, folder_name, result_folder_name, evaluation_type):
    async def run():
        evaluator = Evaluator()
        pipeline = None
        range_enum = Range(range)
        evaluation_type_enum = EvaluationType(evaluation_type)
        score = await evaluator.evaluate(_range=range_enum, result_folder_name=result_folder_name, folder_name=folder_name, evaluation_type=evaluation_type_enum, pipeline=pipeline)
        print(score)

    import asyncio
    asyncio.run(run())
    
if __name__ == "__main__":
    evaluate()