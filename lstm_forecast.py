"""
Time Series Forecasting with LSTM for Stock Prices

This script trains an LSTM model to forecast stock closing prices using
historical OHLCV data. It includes:
- Data loading from CSV or yfinance download
- Train/validation split with sliding window sequences
- LSTM model definition, training loop, and checkpointing
- Inference function for next-N step prediction
- Matplotlib plots comparing actual vs predicted prices

Usage examples:
1) Train from Yahoo Finance and plot:
   python lstm_forecast.py --ticker AAPL --epochs 10 --seq-len 60 --pred-steps 5

2) Use a local CSV:
   python lstm_forecast.py --csv data/stock.csv --target Close --epochs 10

3) Inference only from a saved model:
   python lstm_forecast.py --ticker AAPL --load-model lstm_model.pth --pred-steps 10 --inference-only

Requirements:
- torch, numpy, pandas, matplotlib, scikit-learn, yfinance (optional)
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)


@dataclass
class Config:
    seq_len: int = 60
    pred_steps: int = 1
    epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PriceDataset(torch.utils.data.Dataset):
    def __init__(self, series: np.ndarray, seq_len: int, pred_steps: int):
        self.series = series.astype(np.float32)
        self.seq_len = seq_len
        self.pred_steps = pred_steps
        self.X, self.y = self._create_sequences()

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(self.series) - self.seq_len - self.pred_steps + 1):
            X.append(self.series[i : i + self.seq_len])
            y.append(self.series[i + self.seq_len : i + self.seq_len + self.pred_steps])
        return np.array(X)[..., None], np.array(y)  # add feature dim

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, pred_steps: int = 1):
        # x: (B, T, 1)
        out, (h, c) = self.lstm(x)
        last = out[:, -1, :]  # (B, H)
        next_step = self.fc(last)  # (B, 1)
        preds = [next_step]
        # Autoregressive forecasting
        ht, ct = h, c
        cur = next_step.unsqueeze(1)  # (B,1,1)
        for _ in range(pred_steps - 1):
            out, (ht, ct) = self.lstm(cur, (ht, ct))
            cur = self.fc(out[:, -1, :]).unsqueeze(1)
            preds.append(cur.squeeze(1))
        return torch.cat(preds, dim=1)


def load_series_from_csv(csv_path: str, target: str = "Close") -> pd.Series:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in CSV columns: {df.columns.tolist()}")
    return df[target].dropna().reset_index(drop=True)


def load_series_from_yfinance(ticker: str, period: str = "5y") -> pd.Series:
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install with: pip install yfinance")
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        raise RuntimeError(f"No data downloaded for ticker {ticker}")
    return data["Close"].dropna().reset_index(drop=True)


def train(model, train_loader, val_loader, cfg: Config):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    best_val = float("inf")
    best_path = "lstm_model.pth"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X = X.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad()
            preds = model(X, pred_steps=y.shape[1])
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(cfg.device)
                y = y.to(cfg.device)
                preds = model(X, pred_steps=y.shape[1])
                loss = criterion(preds, y)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{cfg.epochs} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved new best model to {best_path}")


def prepare_loaders(series: pd.Series, cfg: Config) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, MinMaxScaler]:
    # Scale to [0,1]
    scaler = MinMaxScaler()
    values = scaler.fit_transform(series.values.reshape(-1, 1)).squeeze(1)

    # Split
    n_total = len(values)
    n_train = int(n_total * 0.8)
    train_vals = values[:n_train]
    val_vals = values[n_train - cfg.seq_len :]  # overlap to allow first val sequence

    train_ds = PriceDataset(train_vals, cfg.seq_len, cfg.pred_steps)
    val_ds = PriceDataset(val_vals, cfg.seq_len, cfg.pred_steps)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, scaler


def plot_predictions(actual: np.ndarray, pred: np.ndarray, title: str = "LSTM Forecast"):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label="Actual", linewidth=2)
    plt.plot(range(len(actual) - len(pred), len(actual)), pred, label="Predicted", linewidth=2)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lstm_predictions.png", dpi=150)
    print("Saved plot to lstm_predictions.png")
    try:
        plt.show()
    except Exception:
        pass


def infer_next_steps(model: LSTMForecaster, last_seq: np.ndarray, cfg: Config) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(last_seq.astype(np.float32)).unsqueeze(0).unsqueeze(-1).to(cfg.device)
        pred = model(x, pred_steps=cfg.pred_steps).squeeze(0).cpu().numpy()
    return pred


def main():
    parser = argparse.ArgumentParser(description="LSTM Time Series Forecaster for Stocks")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ticker", type=str, help="Yahoo Finance ticker (e.g., AAPL)")
    src.add_argument("--csv", type=str, help="Path to CSV with price column")

    parser.add_argument("--target", type=str, default="Close", help="Target column name for CSV mode")
    parser.add_argument("--period", type=str, default="5y", help="Yahoo Finance download period")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--pred-steps", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--load-model", type=str, default=None, help="Path to a saved .pth model to load")
    parser.add_argument("--inference-only", action="store_true", help="Only run inference without training")

    args = parser.parse_args()

    set_seed(42)
    cfg = Config(
        seq_len=args.seq_len,
        pred_steps=args.pred_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    # Load data
    if args.csv:
        series = load_series_from_csv(args.csv, target=args.target)
    else:
        series = load_series_from_yfinance(args.ticker, period=args.period)

    # Prepare loaders
    train_loader, val_loader, scaler = prepare_loaders(series, cfg)

    # Model
    model = LSTMForecaster(cfg.hidden_size, cfg.num_layers, cfg.dropout).to(cfg.device)

    # Load model if provided
    if args.load_model and os.path.exists(args.load_model):
        model.load_state_dict(torch.load(args.load_model, map_location=cfg.device))
        print(f"Loaded model from {args.load_model}")

    # Train unless inference only
    if not args.inference_only:
        train(model, train_loader, val_loader, cfg)

    # Inference on the last sequence of the full series
    all_scaled = scaler.transform(series.values.reshape(-1, 1)).squeeze(1)
    last_seq = all_scaled[-cfg.seq_len:]
    pred_scaled = infer_next_steps(model, last_seq, cfg)
    # For visualization on scaled values
    actual_scaled = all_scaled.copy()
    # Append prediction for plotting alignment
    viz_series = np.concatenate([actual_scaled, pred_scaled])
    plot_predictions(viz_series, pred_scaled, title=f"{args.ticker or os.path.basename(args.csv)} LSTM Forecast")

    # Optionally save model
    torch.save(model.state_dict(), "lstm_model.pth")
    print("Saved trained model to lstm_model.pth")


if __name__ == "__main__":
    main()
