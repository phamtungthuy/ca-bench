import numpy as np
from keras.layers import TFSMLayer
from keras import Sequential
import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file
import os
from server.settings import device


class TabularClassificationPipeline:
    def __init__(
        self, model_path: str, feature_order: list[str], label_map: dict[int, str]
    ):
        self.model = Sequential(
            [TFSMLayer(model_path, call_endpoint="serving_default")]
        )
        self.feature_order = feature_order
        self.label_map = label_map
        self.device = device  # Store device info for consistency
        print(f"[ info ] TabularClassificationPipeline loaded (TensorFlow model, device: {self.device})")

    def __call__(self, inputs: dict):
        # Chuyển dict -> DataFrame -> np.array theo thứ tự đặc trưng
        df = pd.DataFrame([inputs])
        df = df[self.feature_order]
        input_array = df.to_numpy()
        preds = self.model.predict(input_array)

        if isinstance(preds, dict):
            preds = list(preds.values())[0]

        # Xử lý theo shape
        if len(preds.shape) == 1:
            label_idx = np.argmax(preds)
        else:
            label_idx = np.argmax(preds, axis=1)[0]

        label = self.label_map[label_idx]
        return [{"label": label, "score": float(preds[0][label_idx])}]
    
    def to(self, device):
        """TensorFlow models handle device automatically, but keep for API consistency"""
        self.device = device
        print(f"[ info ] TabularClassificationPipeline device set to {device} (TensorFlow)")
        return self


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        act = torch.relu(self.fc1(x))
        act = torch.relu(self.fc2(act))
        act = torch.relu(self.fc3(act))
        return self.fc4(act)


class TabularRegressionPipeline:
    def __init__(self, model_path: str, feature_order: list[str]):
        self.feature_order = feature_order
        self.device = device  # Store device for later use
        
        from sklearn.datasets import fetch_california_housing

        housing = fetch_california_housing(as_frame=True)

        X = housing["data"]

        self.X_means, self.X_stds = pd.Series(X.mean(axis=0)), pd.Series(X.std(axis=0))

        self.model = MLP()
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # Move model to device
        self.model = self.model.to(self.device)
        print(f"[ info ] TabularRegressionPipeline model moved to {self.device}")

    def __call__(self, inputs: dict):
        df = pd.DataFrame([inputs])
        df = df[self.feature_order]
        df = (df - self.X_means) / self.X_stds
        input_tensor = torch.tensor(df.values, dtype=torch.float32)
        
        # Move input tensor to same device as model
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            preds = self.model(input_tensor)

        return [{"prediction": float(preds[0].item())}]
    
    def to(self, device):
        """Move PyTorch model to specified device"""
        self.device = device
        self.model = self.model.to(device)
        print(f"[ info ] TabularRegressionPipeline moved to {device}")
        return self
