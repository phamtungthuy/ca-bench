import os
import json
from enum import Enum
from typing import Callable, Optional
from templates.workflow_template import Workflow
from templates.pipeline_template import Pipeline
from utils.logs import logger
from scripts.calculator import Calculator
from scripts.calculator import MetricType
import pandas as pd
import numpy as np
import uuid

class Range(Enum):
    TASK = "task"
    NODE = "node"
    CHAIN = "chain"
    GRAPH = "graph"
    ALL = "all"

    @classmethod
    def values(cls):
        return [r.value for r in cls]
    

class EvaluationType(Enum):
    CALCULATE_SCORE = "calculate_score"
    GENERATE_AND_RUN_WORKFLOW = "generate_and_run_workflow"
    RUN_WORKFLOW = "run_workflow"
    GENERATE_WORKFLOW = "generate_workflow"
    ALL = "all"
    
    @classmethod
    def values(cls):
        return [r.value for r in cls]
    
class Evaluator:
    
    def __init__(self):
        self.calculator = Calculator()
    
    def load_workflow(self, folder_path: str) -> Workflow:
        folder_path =  folder_path.replace("\\", ".").replace("/", ".")
        while folder_path.endswith("."):
            folder_path = folder_path[:-1]
        workflow_module_name = f"{folder_path}.workflow"
        try:
            workflow_module = __import__(workflow_module_name, fromlist=[""])
            workflow_class = getattr(workflow_module, "Workflow")
            workflow = workflow_class()
            return workflow
        except ImportError as e:
            logger.info(f"Error loading workflow: {e}")
            raise
        except Exception as e:
            logger.info(f"Error loading workflow: {e}")
            raise
        
    def load_metric_type(self, task_folder: str) -> MetricType:
        if not os.path.exists(os.path.join(task_folder, "evaluation_type.json")):
            raise FileNotFoundError(f"Evaluation type file not found in {task_folder}")
        with open(os.path.join(task_folder, "evaluation_type.json"), "r", encoding="utf-8") as f:
            config = json.loads(f.read())
        return MetricType(config["metric"])
    
    def load_task_description(self, task_folder: str) -> str:
        if not os.path.exists(os.path.join(task_folder, "task_description.txt")):
            raise FileNotFoundError(f"Task description file not found in {task_folder}")
        with open(os.path.join(task_folder, "task_description.txt"), "r", encoding="utf-8") as f:
            return f.read()
    
    def load_predictions(self, folder: str) -> pd.DataFrame:
        try:
            return pd.read_csv(os.path.join(folder, "predictions.csv"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Predictions file not found in {folder}")
    
    def load_references(self, folder: str) -> pd.DataFrame:
        return pd.read_csv(os.path.join(folder, "labels.csv"))
    
    def find_folder_path(self, folder_name: str, base_dir: str = "tasks") -> str:
        levels = ["node-level", "chain-level", "graph-level"]
        for level in levels:
            candidate = os.path.join(base_dir, level, folder_name)
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(f"Task folder '{folder_name}' not found in any level.")

    def write_workflow(self, folder_path: str, code: str):
        with open(os.path.join(folder_path, "workflow.py"), "w", encoding="utf-8") as f:
            f.write(code)
            
    def save_csv(self, folder_path: str, file_name: str, df: pd.DataFrame):
        try:
            df.to_csv(os.path.join(folder_path, file_name), index=False)
        except PermissionError:
            file_name = f"{file_name}_{uuid.uuid4()}.csv"
            self.save_csv(folder_path, file_name, df)
    
    def iter_task_folders(self, _range: Range, result_folder_name: str, folder_name: Optional[str] = None,
                          generate_workflow: Optional[bool] = False):
        """
        The generator returns (task_folder, result_folder, task_name) for each task corresponding to _range.
        """
        if generate_workflow:
            os.makedirs(os.path.join("results", result_folder_name), exist_ok=True)
        if _range == Range.TASK:
            if folder_name is None:
                raise ValueError(f"Task folder is required for range: {_range}")
            task_folder = self.find_folder_path(folder_name, "tasks")
            try:
                result_folder = self.find_folder_path(folder_name, os.path.join("results", result_folder_name))
            except FileNotFoundError as e:
                level = task_folder.split("\\")[-2]
                if generate_workflow:
                    result_folder = os.path.join("results", result_folder_name, level, folder_name)
                    os.makedirs(result_folder, exist_ok=True)
                else:
                    raise e
            print(result_folder)
            yield (task_folder, result_folder, folder_name)
        elif _range in [Range.NODE, Range.CHAIN, Range.GRAPH]:
            level = {
                Range.NODE: "node-level",
                Range.CHAIN: "chain-level",
                Range.GRAPH: "graph-level"
            }[_range]
            task_directory = f"tasks/{level}"
            result_directory = f"results/{result_folder_name}/{level}"
            if generate_workflow:
                os.makedirs(result_directory, exist_ok=True)
            if not os.path.exists(result_directory):
                return
            for folder in os.listdir(task_directory) if generate_workflow else os.listdir(result_directory):
                if os.path.isfile(os.path.join(task_directory, folder)) if generate_workflow else os.path.isfile(os.path.join(result_directory, folder)):
                    continue
                task_folder = os.path.join(task_directory, folder)
                result_folder = os.path.join(result_directory, folder)
                if generate_workflow and not os.path.exists(result_folder):
                    os.makedirs(result_folder, exist_ok=True)
                elif not os.path.exists(result_folder) or not os.path.isdir(task_folder):
                    print(os.path.exists(result_folder), os.path.isdir(task_folder))
                    logger.warning(f"Result folder not found for task: {folder}")
                    continue
                print(result_folder)
                
                yield (task_folder, result_folder, folder)
        elif _range == Range.ALL:
            task_directory = "tasks"
            result_directory = f"results/{result_folder_name}"
            if generate_workflow:
                os.makedirs(result_directory, exist_ok=True)
            if not os.path.exists(result_directory):
                return
            for level_folder in os.listdir(task_directory) if generate_workflow else os.listdir(result_directory):
                if os.path.isfile(os.path.join(task_directory, level_folder)) if generate_workflow else os.path.isfile(os.path.join(result_directory, level_folder)):
                    continue
                for folder in os.listdir(os.path.join(task_directory, level_folder)) if generate_workflow else os.listdir(os.path.join(result_directory, level_folder)):
                    if os.path.isfile(os.path.join(result_directory, level_folder, folder)) or os.path.isfile(os.path.join(task_directory, level_folder, folder)):
                        continue
                    task_folder = os.path.join(task_directory, level_folder, folder)
                    result_folder = os.path.join(result_directory, level_folder, folder)
                    if generate_workflow and not os.path.exists(result_folder):
                        os.makedirs(result_folder, exist_ok=True)
                    elif not os.path.exists(result_folder) or not os.path.isdir(task_folder):
                        logger.warning(f"Result folder not found for task: {folder}")
                        continue
                    print(result_folder)
                    yield (task_folder, result_folder, folder)
        else:
            raise ValueError("Invalid range")

    async def process_task_score(self, task_folder, result_folder):
        metric_type = self.load_metric_type(task_folder)
        try:
            predictions = self.load_predictions(result_folder)
        except FileNotFoundError:
            print(f"Error loading predictions: {result_folder}")
            return 0, pd.DataFrame()
        references = self.load_references(task_folder)
        score, df = self.calculator(metric_type, predictions, references)
        self.save_csv(result_folder, "scores.csv", df)
        return score, df

    async def process_task_generate(self, task_folder, result_folder, pipeline, run_workflow):
        try:
            task_description = self.load_task_description(task_folder)
        except FileNotFoundError:
            logger.warning(f"Task description file not found in {task_folder}")
            return None, None
        code = await pipeline(task_description)
        self.write_workflow(result_folder, code)
        if run_workflow:
            workflow = self.load_workflow(result_folder)
            predictions = await workflow(task_folder)
            self.save_csv(result_folder, "predictions.csv", predictions)
            return await self.process_task_score(task_folder, result_folder)
        return None, None

    async def process_task_run(self, task_folder, result_folder):
        try:
            workflow = self.load_workflow(result_folder)
            predictions = await workflow(task_folder)
            self.save_csv(result_folder, "predictions.csv", predictions)
            return await self.process_task_score(task_folder, result_folder)
        except Exception as e:
            print(f"Error processing {task_folder}: {e}")
            return 0, pd.DataFrame()

    async def calculate_score(self, _range: Range, result_folder_name: str,
        folder_name: Optional[str] = None):
        scores = []
        tasks = []
        result_directory = None
        for task_folder, result_folder, task_name in self.iter_task_folders(_range, result_folder_name, folder_name):
            result_directory = os.path.dirname(result_folder) if _range == Range.TASK else os.path.commonpath([result_folder, result_directory]) if result_directory else result_folder
            score, _ = await self.process_task_score(task_folder, result_folder)
            if score is not None:
                scores.append(score)
                tasks.append(task_name)
        if not scores:
            return None
        final_scores = pd.DataFrame({"id": range(len(tasks)), "task": tasks, "score": scores})
        if _range == Range.TASK:
            self.save_csv(result_folder, "final_scores.csv", final_scores)
        else:
            self.save_csv(result_directory, "final_scores.csv", final_scores)
        print(scores)
        return np.mean(scores)

    async def generate_workflow(self, _range: Range, result_folder_name: str, pipeline: Pipeline, folder_name: Optional[str] = None,
                                run_workflow: bool = False):
        scores = []
        tasks = []
        result_directory = None
        for task_folder, result_folder, task_name in self.iter_task_folders(_range, result_folder_name, folder_name,
                                                                            generate_workflow=True):
            result_directory = os.path.dirname(result_folder) if _range == Range.TASK else os.path.commonpath([result_folder, result_directory]) if result_directory else result_folder
            score, _ = await self.process_task_generate(task_folder, result_folder, pipeline, run_workflow)
            if run_workflow and score is not None:
                scores.append(score)
                tasks.append(task_name)
        if run_workflow and scores:
            final_scores = pd.DataFrame({"id": range(len(tasks)), "task": tasks, "score": scores})
            if _range == Range.TASK:
                self.save_csv(result_folder, "final_scores.csv", final_scores)
            else:
                self.save_csv(result_directory, "final_scores.csv", final_scores)
            return np.mean(scores)
        else:
            return None

    async def run_workflow(self, _range: Range, result_folder_name: str,
        folder_name: Optional[str] = None):
        scores = []
        tasks = []
        result_directory = None
        for task_folder, result_folder, task_name in self.iter_task_folders(_range, result_folder_name, folder_name):
            result_directory = os.path.dirname(result_folder) if _range == Range.TASK else os.path.commonpath([result_folder, result_directory]) if result_directory else result_folder
            score, _ = await self.process_task_run(task_folder, result_folder)
            if score is not None:
                scores.append(score)
                tasks.append(task_name)
        if not scores:
            return None
        final_scores = pd.DataFrame({"id": range(len(tasks)), "task": tasks, "score": scores})
        if _range == Range.TASK:
            self.save_csv(result_folder, "final_scores.csv", final_scores)
        else:
            self.save_csv(result_directory, "final_scores.csv", final_scores)
        return np.mean(scores)
    
    async def evaluate(self, _range: Range, result_folder_name: str, folder_name: Optional[str] = None,
                       evaluation_type: EvaluationType = EvaluationType.CALCULATE_SCORE, pipeline: Optional[Pipeline] = None):
        if evaluation_type == EvaluationType.CALCULATE_SCORE:
            return await self.calculate_score(_range=_range, result_folder_name=result_folder_name, folder_name=folder_name)
        elif evaluation_type == EvaluationType.GENERATE_AND_RUN_WORKFLOW:
            return await self.generate_workflow(_range=_range, result_folder_name=result_folder_name, pipeline=pipeline, folder_name=folder_name, run_workflow=True)
        elif evaluation_type == EvaluationType.RUN_WORKFLOW:
            return await self.run_workflow(_range=_range, result_folder_name=result_folder_name, folder_name=folder_name)
        elif evaluation_type == EvaluationType.GENERATE_WORKFLOW:
            return await self.generate_workflow(_range=_range, result_folder_name=result_folder_name, pipeline=pipeline, folder_name=folder_name, run_workflow=False)
        elif evaluation_type == EvaluationType.ALL:
            return await self.generate_workflow(_range=_range, result_folder_name=result_folder_name, pipeline=pipeline, folder_name=folder_name, run_workflow=True)
        else:
            raise ValueError("Invalid evaluation type")