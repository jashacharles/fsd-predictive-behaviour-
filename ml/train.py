import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from ml.config import (
    TRAIN_DATA_DIR,
    MODEL_PATH,
    STATS_PATH,
    SEQ_LEN,
    PREDICT_HORIZON,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_CLASSES,
    DEVICE,
)

from ml.data_utils import (
    get_csv_files,
    get_feature_columns,
    build_dataset_from_multiple_csvs,
    normalize_train_data,
    split_train_test,
    DrivingDataset,
)

from ml.model import DrivingTransformer


def train_model():
    print("Device:", DEVICE)

    csv_files = get_csv_files(TRAIN_DATA_DIR)

    print("Found training CSVs:", len(csv_files))
    for file in csv_files:
        print(file)

    feature_columns = get_feature_columns(csv_files[0])

    print("Feature count:", len(feature_columns))
    print("First features:", feature_columns[:10])
    print("Last features:", feature_columns[-10:])

    X_seq, y_seq = build_dataset_from_multiple_csvs(
        csv_files=csv_files,
        feature_columns=feature_columns,
        seq_len=SEQ_LEN,
        predict_horizon=PREDICT_HORIZON,
    )

    print("X shape:", X_seq.shape)
    print("y shape:", y_seq.shape)

    X_seq_norm, mean, std = normalize_train_data(X_seq)

    X_train, X_test, y_train, y_test = split_train_test(
        X_seq_norm,
        y_seq,
        test_size=0.2
    )

    train_loader = DataLoader(
        DrivingDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        DrivingDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = DrivingTransformer(
        input_dim=len(feature_columns),
        seq_len=SEQ_LEN,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    class_counts = np.bincount(y_train.astype(int), minlength=NUM_CLASSES)
    total_samples = len(y_train)

    class_weights_array = np.array(
        [
            total_samples / max(count, 1)
            for count in class_counts
        ],
        dtype=np.float32
    )

    class_weights = torch.tensor(
        class_weights_array,
        dtype=torch.float32
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print("Class counts:", class_counts)
    print("Class weights:", class_weights.cpu().numpy())

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_X)
            loss = loss_fn(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                logits = model(batch_X)
                loss = loss_fn(logits, batch_y)

                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

    np.savez(
        STATS_PATH,
        mean=mean,
        std=std,
        feature_columns=np.array(feature_columns),
        seq_len=np.array(SEQ_LEN)
    )

    print("Training complete.")
    print("Best val loss:", best_val_loss)
    print("Saved model to:", MODEL_PATH)
    print("Saved normalization stats to:", STATS_PATH)


if __name__ == "__main__":
    train_model()