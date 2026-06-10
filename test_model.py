from pathlib import Path

import numpy as np
import torch
from torch import nn



class DrivingTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        seq_len,
        d_model=64,
        nhead=4,
        num_layers=2,
        feedforward_dim=128,
        dropout=0.2,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.predictor_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_embedding[:, :x.shape[1], :]
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        return self.predictor_head(x).squeeze(-1)


class DrivingModelInference:
    def __init__(
        self,
        model_path="best_driving_transformer.pth",
        normalization_path="normalization_stats.npz",
        device=None,
    ):
        self.model_path = Path(model_path)
        self.normalization_path = Path(normalization_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        stats = np.load(self.normalization_path, allow_pickle=True)
        self.mean = stats["mean"].astype(np.float32)
        self.std = stats["std"].astype(np.float32)
        self.feature_columns = [str(column) for column in stats["feature_columns"]]
        self.seq_len = int(stats["seq_len"])

        self.model = DrivingTransformer(
            input_dim=len(self.feature_columns),
            seq_len=self.seq_len,
        ).to(self.device)

        state_dict = torch.load(self.model_path, map_location=self.device)
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_probability(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.shape != (self.seq_len, len(self.feature_columns)):
            raise ValueError(
                "Expected sequence shape "
                f"({self.seq_len}, {len(self.feature_columns)}), got {sequence.shape}"
            )

        normalized = (sequence[None, :, :] - self.mean) / self.std
        tensor = torch.from_numpy(normalized).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probability = torch.sigmoid(logits)[0].item()

        return float(probability)
