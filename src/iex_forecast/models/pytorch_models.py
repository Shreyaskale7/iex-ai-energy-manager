"""PyTorch based Deep Learning models for RTM forecasting."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    class LSTMForecaster(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            # x shape: (batch_size, seq_len, input_dim)
            out, _ = self.lstm(x)
            # Take the output of the last time step
            out = self.fc(out[:, -1, :]) 
            return out
else:
    class LSTMForecaster:
        pass

class PyTorchLSTMModel(BaseEstimator):
    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, epochs: int = 10, lr: float = 0.005):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not HAS_TORCH:
            raise ImportError("PyTorch is not installed")
            
        X_arr = X.values.astype(np.float32)
        y_arr = y.values.astype(np.float32)
        
        # We treat each tabular row as a sequence of length 1
        X_tensor = torch.tensor(X_arr).unsqueeze(1).to(self.device)
        y_tensor = torch.tensor(y_arr).unsqueeze(1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        self.model = LSTMForecaster(input_dim=X.shape[1], hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        self.model.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted")
            
        self.model.eval()
        X_arr = X.values.astype(np.float32)
        X_tensor = torch.tensor(X_arr).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor)
            
        return preds.cpu().numpy().flatten()
