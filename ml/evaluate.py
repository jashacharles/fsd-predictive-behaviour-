import glob
import numpy as np
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from ml.config import (
    EVAL_DATA_DIR,
    MODEL_PATH,
    STATS_PATH,
    SEQ_LEN,
    PREDICT_HORIZON,
    BATCH_SIZE,
    NUM_CLASSES,
    DEVICE,
)

from ml.data_utils import (
    get_csv_files,
    build_sequences_from_csv,
    apply_normalization,
    DrivingDataset,
)

from ml.model import DrivingTransformer


def evaluate_model():
    stats = np.load(STATS_PATH, allow_pickle=True)

    mean = stats["mean"]
    std = stats["std"]
    feature_columns = stats["feature_columns"].tolist()

    eval_files = get_csv_files(EVAL_DATA_DIR)
    print(f"Found evaluation CSVs: {len(eval_files)}")
    for f in eval_files:
        print(f)

    all_X, all_y = [], []
    for file_path in eval_files:
        X_eval, y_eval = build_sequences_from_csv(
            file_path=file_path,
            feature_columns=feature_columns,
            seq_len=SEQ_LEN,
            predict_horizon=PREDICT_HORIZON,
        )
        all_X.append(X_eval)
        all_y.append(y_eval)

    X_eval = np.concatenate(all_X, axis=0)
    y_eval = np.concatenate(all_y, axis=0)

    X_eval_norm = apply_normalization(X_eval, mean, std)

    eval_loader = DataLoader(
        DrivingDataset(X_eval_norm, y_eval),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = DrivingTransformer(
        input_dim=len(feature_columns),
        seq_len=SEQ_LEN,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in eval_loader:
            batch_X = batch_X.to(DEVICE)

            logits = model(batch_X)
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("Evaluation sequences:", X_eval.shape)

    print("Actual stay samples:", int(np.sum(all_labels == 0)))
    print("Actual move_left samples:", int(np.sum(all_labels == 1)))
    print("Actual move_right samples:", int(np.sum(all_labels == 2)))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["stay", "move_left", "move_right"]
    ))


if __name__ == "__main__":
    evaluate_model()