import os
import base64
import json
from io import BytesIO
import re
import requests
import numpy as np
import torch
from PIL import Image
from huggingface_hub import InferenceClient
from utils.constants import (
    config,
    HUGGINGFACE_HEADERS,
    PROXY,
    LOCAL_INFERENCE_ENDPOINT_URL,
    ROOT_PATH
)
from utils.common import image_to_bytes, audio_to_bytes, video_to_bytes
import soundfile as sf

model_desc = {}

with open(os.path.join(ROOT_PATH, "data", "huggingface_models.jsonl"), "r", encoding="utf-8") as f:
    for line in f.readlines():
        info = json.loads(line)
        model_desc[info["id"]] = {
            "metadata": info["metadata"] if "metadata" in info else {},
        }
        

def huggingface_model_inference(model_id: str, data: dict, task: str) -> dict:
    task_url = f"https://api-inference.huggingface.co/models/{model_id}"  # InferenceApi does not yet support some tasks

    client = InferenceClient(
        provider="hf-inference", api_key=config["huggingface"]["token"]
    )
    print(task, model_id)
    if task in ["image-classification", "image-to-text"]:
        HUGGINGFACE_HEADERS["Content-Type"] = "image/jpeg"
        if isinstance(data["image"], str):
            base64_pattern = r"^[A-Za-z0-9+/=]+$"
            img_url_or_base64 = data["image"]
            if re.match(base64_pattern, img_url_or_base64):
                # If input is base64, decode to bytes
                img_data = base64.b64decode(img_url_or_base64)
            else:
                # If input is URL, download bytes from URL
                img_data = image_to_bytes(img_url_or_base64)
        elif isinstance(data["image"], Image.Image):
            # If input is PIL Image, save to bytes
            img_obj = data["image"]
            img_data = BytesIO()
            img_obj.save(img_data, format="PNG")
            img_data = img_data.getvalue()
        elif isinstance(data["image"], (np.ndarray, torch.Tensor, list)):
            # If input is numpy array or torch tensor
            img_array = data["image"]
            if isinstance(img_array, list):
                # Convert list to NumPy array
                img_array = np.array(img_array, dtype=np.float32)
            if isinstance(img_array, torch.Tensor):
                img_array = img_array.cpu().numpy()  # Convert tensor to NumPy
            if img_array.shape[-1] == 1:  # Handle grayscale
                img_array = img_array.squeeze(-1)
            # Convert to uint8 if needed
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            img_obj = Image.fromarray(img_array)
            img_data = BytesIO()
            img_obj.save(img_data, format="PNG")
            img_data = img_data.getvalue()
        else:
            # If input is bytes
            img_data = data["image"]
        r = requests.post(
            task_url, headers=HUGGINGFACE_HEADERS, data=img_data, proxies=PROXY
        )
        result = {}
        if task == "image-classification":
            print(r)
            result["predicted"] = r.json()[0].pop("label")
        elif task == "image-to-text":
            result["text"] = r.json()[0].pop("generated_text")
        return result

    if task == "zero-shot-classification":
        text = data["text"]
        labels = data["labels"]
        r = requests.post(
            task_url,
            json={"inputs": text, "parameters": {"candidate_labels": labels}},
            headers=HUGGINGFACE_HEADERS,
        )
        result = {}
        result["predicted"] = r.json()["labels"][0]
        return result

    if task == "question-answering":
        result = client.question_answering(
            question=data["text"],
            context=data["context"] if "context" in data else "",
            model=model_id,
        )
        return result

    if task == "sentence-similarity":
        r = client.sentence_similarity(
            sentence=data["text"],
            other_sentences=(
                data["other_sentences"] if "other_sentences" in data else []
            ),
            model=model_id,
        )
        result = {"predicted": r}
        return result

    if task in ["translation"]:
        inputs = data["text"]
        result = client.translation(text=inputs, model=model_id)
        return result

    if task in ["summarization"]:
        if "file" in data:
            file = data["file"]
            with open(file, "r", encoding="utf-8") as f:
                inputs = f.read()
        else:
            inputs = data["text"]
        result = client.summarization(text=inputs, model=model_id)
        return result

    if task in [
        "text-classification",
        "text-generation",
    ]:
        inputs = data["text"]
        r = client.text_classification(text=inputs, model=model_id)
        result = {}
        if task == "text-classification":
            result["predicted"] = r[0].pop("label")
            if model_id == "mshenoda/roberta-spam":
                result["predicted"] = (
                    "spam" if result["predicted"] == "LABEL_1" else "ham"
                )
        if "id2label" in model_desc[model_id]["metadata"]:
            result["predicted"] = model_desc[model_id]["metadata"]["id2label"][result["predicted"]]
        return result

    if task == "token-classification":
        inputs = data["text"]
        r = client.token_classification(text=inputs, model=model_id)
        result = {"predicted": []}
        for item in r:
            result["predicted"].append(
                {
                    "word": item["word"],
                    "entity_group": item["entity_group"],
                }
            )
        return result

    if task in [
        "automatic-speech-recognition",
        "audio-classification",
    ]:
        audio = data["audio"]
        if isinstance(audio, str):
            base64_pattern = r"^[A-Za-z0-9+/=]+$"
            audio_url_or_base64 = audio
            if re.match(base64_pattern, audio_url_or_base64):
                audio_data = base64.b64decode(audio_url_or_base64)
            else:
                audio_data = audio_to_bytes(audio_url_or_base64)
        elif isinstance(audio, BytesIO):
            audio.seek(0)
            audio_data = audio.read()
        elif isinstance(audio, np.ndarray):
            audio_data = audio.tobytes()
        else:
            audio_data = audio
        client.headers["Content-Type"] = "audio/mpeg"
        if task == "automatic-speech-recognition":
            response = client.automatic_speech_recognition(audio=audio_data, model=model_id)
            result = response
        elif task == "audio-classification":
            response = client.audio_classification(audio=audio_data, model=model_id)
            result = response
        return result

    raise ValueError(f"Unsupported task: {task}")


def local_model_inference(model_id, data, task):
    task_url = f"{LOCAL_INFERENCE_ENDPOINT_URL}/models?model_id={model_id}"

    files = {}

    if task in ["image-classification", "object-detection", "image-to-text"]:
        print(type(data["image"]))
        img_data = None
        # Handle different types of input
        if isinstance(data["image"], str):
            base64_pattern = r"^[A-Za-z0-9+/=]+$"
            img_url_or_base64 = data["image"]
            if re.match(base64_pattern, img_url_or_base64):
                # If input is base64, decode to bytes
                img_data = base64.b64decode(img_url_or_base64)
            else:
                # If input is URL, download bytes from URL
                img_data = image_to_bytes(img_url_or_base64)
        elif isinstance(data["image"], Image.Image):
            # If input is PIL Image, save to bytes
            img_obj = data["image"]
            img_data = BytesIO()
            img_obj.save(img_data, format="PNG")
            img_data = img_data.getvalue()
        elif isinstance(data["image"], (np.ndarray, torch.Tensor, list)):
            # If input is numpy array or torch tensor
            img_array = data["image"]
            if isinstance(img_array, list):
                # Convert list to NumPy array
                img_array = np.array(img_array, dtype=np.float32)
            if isinstance(img_array, torch.Tensor):
                img_array = img_array.cpu().numpy()  # Convert tensor to NumPy
            if img_array.shape[-1] == 1:  # Handle grayscale
                img_array = img_array.squeeze(-1)
            # Convert to uint8 if needed
            if img_array.dtype != np.uint8:
                img_array = (img_array * 255).astype(np.uint8)
            img_obj = Image.fromarray(img_array)
            img_data = BytesIO()
            img_obj.save(img_data, format="PNG")
            img_data = img_data.getvalue()
        else:
            # If input is bytes
            img_data = data["image"]

        files["image"] = ("image.png", img_data, "image/png")
        data = {}

    elif task in [
        "automatic-speech-recognition",
        "audio-classification",
    ]:
        audio = data["audio"]
        if isinstance(audio, str):
            base64_pattern = r"^[A-Za-z0-9+/=]+$"
            audio_url_or_base64 = audio
            if re.match(base64_pattern, audio_url_or_base64):
                audio_data = base64.b64decode(audio_url_or_base64)
            else:
                audio_data = audio_to_bytes(audio_url_or_base64)
        else:
            audio_data = audio
        files["audio"] = ("audio.flac", audio_data, "audio/flac")
        data = {}

    response = requests.post(task_url, data={"data": json.dumps(data)}, files=files)
    print(response.json())
    response = response.json()
    if response["code"] == "000":
        return response["data"]
    else:
        raise ValueError(f"Error: {response['message']}")


def model_inference(model_id: str, input_data: dict, hosted_on: str = None, task: str = ""):

    try:
        if hosted_on == "huggingface":
            try:
                return huggingface_model_inference(model_id, input_data, task)
            except Exception as e:
                return local_model_inference(model_id, input_data, task)
        elif hosted_on == "local":
            return local_model_inference(model_id, input_data, task)
        else:
            return local_model_inference(model_id, input_data, task)
            # raise ValueError(f"Unsupported hosted_on: {hosted_on}")
    except Exception as e:
        raise e


if __name__ == "__main__":

    result = model_inference(
        model_id="anton-l/wav2vec2-base-lang-id",
        input_data={
            "audio": "tasks/node-level/lang_audio_classification/audios/4.wav"
        },
        hosted_on="local",
        task="audio-classification",
    )

    print(result)
