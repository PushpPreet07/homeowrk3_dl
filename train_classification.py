
import copy
import torch
import torch.nn as nn

from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return correct / total if total > 0 else 0.0


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Adjust these arguments if your dataset loader uses a different API.
    train_loader = load_data(
        "classification_data/train",
        transform_pipeline="aug",
        batch_size=128,
        num_workers=0,
        shuffle=True,
    )

    val_loader = load_data(
        "classification_data/val",
        transform_pipeline="default",
        batch_size=128,
        num_workers=0,
        shuffle=False,
    )

    model = Classifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    num_epochs = 30
    best_val_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / max(total_samples, 1)
        val_accuracy = evaluate(model, val_loader, device)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {train_loss:.4f} "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    # Restore the best validation checkpoint before saving.
    model.load_state_dict(best_state)
    save_model(model)

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print("Saved trained weights to homework/classifier.th")


if __name__ == "__main__":
    main()
