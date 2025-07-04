import os
import io
import datetime
import json
from typing import Optional

from fastapi import APIRouter, Query, Form, UploadFile, File
from server.schema_base import DataResponse

import time
from server.settings import device, ROOT_PATH
from transformers.pipelines import pipeline
from PIL import Image
from ultralytics import YOLO
from server.custom_pipeline.tabular_pipeline import (
    TabularClassificationPipeline,
    TabularRegressionPipeline,
)
import soundfile as sf

MAX_MODELS_IN_RAM = 5
MAX_MODELS_IN_DISK = 25  # More space for disk storage

local_fold = f"{ROOT_PATH}/server/models"

router = APIRouter()

pipes = {}


def download_model(model_id: str):
    model_path = f"{local_fold}/{model_id}"
    if os.path.exists(model_path) and os.listdir(model_path):
        return
    
    os.makedirs(model_path, exist_ok=True)
    
    # Try git clone first
    clone_cmd = f"git clone --recurse-submodules https://huggingface.co/{model_id} {model_path}"
    result = os.system(clone_cmd)
    
    if result != 0:
        print(f"[ error ] Failed to clone model {model_id} using git. Error code: {result}")
        # Clean up empty directory
        import shutil
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        raise Exception(f"Failed to download model {model_id}. Make sure git is installed and model exists on HuggingFace.")
    
    # Update lasted_used_in_disk
    if model_id in pipes:
        pipes[model_id]["lasted_used_in_disk"] = datetime.datetime.now()
    
    print(f"[ success ] Downloaded model {model_id}")


with open(f"{ROOT_PATH}/data/model_desc.jsonl", "r", encoding="utf-8") as f:
    for line in f.readlines():
        info = json.loads(line)
        model_folder = f"{local_fold}/{info['id']}"
        
        # Check if model already exists on disk
        model_exists_on_disk = os.path.exists(model_folder) and os.listdir(model_folder)
        lasted_used_in_disk = datetime.datetime.now() if model_exists_on_disk else None
        
        if model_exists_on_disk:
            print(f"[ detected ] Model {info['id']} already exists on disk")
        
        pipes[info["id"]] = {
            "model": None,
            "lasted_used": None,
            "lasted_used_in_disk": lasted_used_in_disk,
            "type": info["tag"],
            "device": device,
            "model_path": model_folder,
            "desc": info["desc"],
            "inference_type": "default",
            "metadata": info["metadata"] if "metadata" in info else {}
        }


def manage_disk_space():
    """Manage disk space - delete oldest models if exceeds the limit"""
    while True:
        models_on_disk = [k for k, v in pipes.items() if v["lasted_used_in_disk"] is not None]
        
        if len(models_on_disk) <= MAX_MODELS_IN_DISK:
            break
        # Find models NOT in RAM to prioritize for deletion
        models_not_in_ram = [k for k in models_on_disk if pipes[k]["model"] is None]
        
        if models_not_in_ram:
            # Prefer deleting models not in RAM
            oldest_model_id = min(models_not_in_ram, key=lambda x: pipes[x]["lasted_used_in_disk"])
            print(f"[ cleanup disk ] Deleting model {oldest_model_id} (not in RAM, disk full: {len(models_on_disk)}/{MAX_MODELS_IN_DISK})")
        else:
            # If all models are in RAM, delete the oldest one
            oldest_model_id = min(models_on_disk, key=lambda x: pipes[x]["lasted_used_in_disk"])
            print(f"[ cleanup disk ] Deleting model {oldest_model_id} (in RAM but oldest, disk full: {len(models_on_disk)}/{MAX_MODELS_IN_DISK})")
            
            # Unload from RAM since we're deleting from disk
            if pipes[oldest_model_id]["model"] is not None:
                del pipes[oldest_model_id]["model"]
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pipes[oldest_model_id]["lasted_used"] = None
                print(f"[ unloaded ] Model {oldest_model_id} from RAM (disk deleted)")
        
        # Delete folder from disk
        model_folder = f"{local_fold}/{oldest_model_id}"
        if os.path.exists(model_folder):
            import shutil
            try:
                shutil.rmtree(model_folder)
                print(f"[ deleted ] Model folder {model_folder}")
            except Exception as e:
                print(f"[ warning ] Failed to delete model folder {model_folder}: {e}")
        
        # Reset disk tracking
        pipes[oldest_model_id]["lasted_used_in_disk"] = None
        # Note: model is already None or handled above
        # Loop will continue until within limit


def get_pipe(model_id: str):
    if model_id not in pipes:
        return
    
    # If already loaded in RAM, update both timestamps and return
    if pipes[model_id]["model"] is not None:
        pipes[model_id]["lasted_used"] = datetime.datetime.now()
        pipes[model_id]["lasted_used_in_disk"] = datetime.datetime.now()
        return pipes[model_id]
    
    # Manage RAM FIRST - unload the oldest model if exceeds the limit
    loaded_models = [k for k, v in pipes.items() if v["model"] is not None]
    if len(loaded_models) >= MAX_MODELS_IN_RAM:
        oldest_model_id = min(loaded_models, key=lambda x: pipes[x]["lasted_used"])
        print(f"[ unloading model {oldest_model_id} ] from RAM only (keeping on disk)")
        del pipes[oldest_model_id]["model"]
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pipes[oldest_model_id]["lasted_used"] = None
        pipes[oldest_model_id]["model"] = None
    
    # Download model if not exists on disk
    download_model(model_id)
    
    # Manage disk space after download (now safe - no RAM models will be deleted)
    manage_disk_space()

    pipe = pipes[model_id]
    if pipe["type"] == "object-detection":
        model = YOLO(f"{local_fold}/{model_id}/yolo11l.pt")
    elif pipe["type"] == "tabular-regression":
        model = TabularRegressionPipeline(
            model_path=pipe["model_path"],
            feature_order=[
                "MedInc",
                "HouseAge",
                "AveRooms",
                "AveBedrms",
                "Population",
                "AveOccup",
                "Latitude",
                "Longitude",
            ],
        )
    elif pipe["type"] == "tabular-classification":
        model = TabularClassificationPipeline(
            model_path=pipe["model_path"],
            feature_order=[
                "Time",
                "V1",
                "V2",
                "V3",
                "V4",
                "V5",
                "V6",
                "V7",
                "V8",
                "V9",
                "V10",
                "V11",
                "V12",
                "V13",
                "V14",
                "V15",
                "V16",
                "V17",
                "V18",
                "V19",
                "V20",
                "V21",
                "V22",
                "V23",
                "V24",
                "V25",
                "V26",
                "V27",
                "V28",
                "Amount",
            ],
            label_map={0: "0", 1: "1"},
        )
    else:
        if pipe["type"] == "token-classification":
            model = pipeline(task=pipe["type"], model=pipe["model_path"], aggregation_strategy="simple")
        else:
            model = pipeline(task=pipe["type"], model=pipe["model_path"])
    pipe["model"] = model
    pipe["lasted_used"] = datetime.datetime.now()
    pipe["lasted_used_in_disk"] = datetime.datetime.now()  # Update disk usage tracking
    return pipe


@router.get("/running")
def running():
    return DataResponse().success_response(data={"running": True})


@router.get("/status/{model_id}")
def status(model_id: str):
    disabled_models = [
        # "microsoft/trocr-base-printed",
        "microsoft/trocr-base-handwritten",
    ]
    if model_id in pipes.keys() and model_id not in disabled_models:
        print(f"[ check {model_id} ] success")
        return DataResponse().success_response(data={"loaded": True})
    else:
        print(f"[ check {model_id} ] failed")
        return DataResponse().custom_response(
            code="001", message="Model not found", data={"loaded": False}
        )


@router.post("/models")
async def inference(
    model_id: str = Query(...),
    data: str = Form(...),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    if model_id not in pipes:
        return DataResponse().custom_response(
            code="001", message="Model not found", data={"inference": "failed"}
        )
    data = json.loads(data)
    # while "using" in pipes[model_id] and pipes[model_id]["using"]:
    #     print(f"[ inference {model_id} ] waiting")
    #     time.sleep(0.1)

    print(f"[ inference {model_id} ] start")
    pipe = get_pipe(model_id)
    pipe["using"] = True

    start = time.time()
    model_type = pipe["type"]
    model = pipe["model"]
    device = pipe["device"]

    result = {}
    if "device" in pipe:
        try:
            # Nếu là pipeline HuggingFace có model torch => chuyển
            if hasattr(model, "model") and hasattr(model.model, "to"):
                model.model.to(device)
            elif hasattr(model, "to"):
                model.to(device)
            else:
                print(f"[ info ] Skip device transfer for model_id={model_id}")
        except Exception as e:
            print(f"[ warning ] Cannot move model to device: {e}")
    if model_type == "token-classification":
        text = data["text"]
        r = model(text)
        result["predicted"] = []
        for item in r:
            result["predicted"].append({
                "word": item["word"],
                "entity_group": item["entity_group"],
            })
    if model_type == "text-classification":
        text = data["text"]
        r = model(text)
        result["predicted"] = r[0].pop("label")
    if model_type == "zero-shot-classification":
        text = data["text"]
        labels = data["labels"]
        r = model(text, candidate_labels=labels)
        result["predicted"] = r["labels"][0]
    if model_type == "translation":
        text = data["text"]
        r = model(text)
        result["text"] = r[0].pop("translation_text")
    if model_type == "summarization":
        text = data["text"]
        labels = data["labels"]
        r = model(text, candidate_labels=labels)
        result["predicted"] = r["labels"][0]
    if model_type == "question-answering":
        context = data["context"]
        question = data["question"]
        r = model(question=question, context=context)
        result["predicted"] = r.pop("answer")
    if model_type == "text-generation":
        text = data["text"]
        r = model(data["text"])
        result["text"] = r[0].pop("generated_text")
    if model_type == "sentence-similarity":
        pass
    if model_type == "tabular-classification":
        row = data["row"]
        r = model(row)
        result["predicted"] = r[0].pop("label")
    if model_type == "object-detection":
        image_data = Image.open(io.BytesIO(await image.read())).convert("RGB")
        r = model(image_data)
        result["predicted"] = []
        for item in r:
            boxes = item.boxes.xyxy.cpu().numpy()
            scores = item.boxes.conf.cpu().numpy()
            classes = item.boxes.cls.cpu().numpy().astype(int)
            for box, score, cls in zip(boxes, scores, classes):
                label = model.names[cls]
                result["predicted"].append(
                    {
                        "label": label,
                        "score": float(score),
                        "box": {
                            "xmin": int(box[0]),
                            "ymin": int(box[1]),
                            "xmax": int(box[2]),
                            "ymax": int(box[3]),
                        },
                    }
                )

    if model_type == "image-classification":
        image_data = Image.open(io.BytesIO(await image.read())).convert("RGB")            
        r = model(image_data)
        result["predicted"] = r[0].pop("label")
        if "id2label" in pipe["metadata"]:
            result["predicted"] = pipe["metadata"]["id2label"][result["predicted"]]
    if model_type == "image-to-text":
        image_data = Image.open(io.BytesIO(await image.read())).convert("RGB")
        r = model(image_data)
        result["text"] = r[0].pop("generated_text")
    if model_type == "automatic-speech-recognition":
        raise NotImplementedError("Automatic speech recognition is not implemented")
    if model_type == "audio-classification":
        audio_data = await audio.read()
        r = model(audio_data)
        result["predicted"] = r[0].pop("label")
    if model_type == "tabular-regression":
        row = data["row"]
        r = model(row)
        result["predicted"] = r[0].pop("prediction")
    if model_type == "video-classification":
        video = data["video"].replace("\\", "/")
        r = model(video)
        result["predicted"] = r[0].pop("label")

    pipe["using"] = False

    if result is None:
        return DataResponse().custom_response(
            code="002", message="Model not found", data={"inference": "failed"}
        )

    end = time.time()
    during = end - start
    print(f"[ complete {model_id} ] {during}s")
    print(f"[ result {model_id} ] {result}")

    return DataResponse().success_response(data=result)
