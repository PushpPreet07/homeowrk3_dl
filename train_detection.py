
import copy
import torch
import torch.nn as nn

from homework.models import Detector, save_model
from homework.datasets.road_dataset import load_data


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()

    # Confusion matrix: rows = true classes, columns = predicted classes.
    confusion = torch.zeros(3, 3, dtype=torch.long, device=device)

    depth_abs_error = 0.0
    depth_count = 0
    lane_abs_error = 0.0
    lane_count = 0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["track"].to(device).long()
            depths = batch["depth"].to(device).float()

            logits, predicted_depth = model(images)
            predictions = logits.argmax(dim=1)

            valid = (masks >= 0) & (masks < 3)
            encoded = 3 * masks[valid] + predictions[valid]
            confusion += torch.bincount(
                encoded, minlength=9
            ).reshape(3, 3)

            errors = (predicted_depth - depths).abs()
            depth_abs_error += errors.sum().item()
            depth_count += errors.numel()

            # Lane boundaries are classes 1 and 2.
            lane_pixels = (masks > 0) & valid
            lane_abs_error += errors[lane_pixels].sum().item()
            lane_count += lane_pixels.sum().item()

    true_positive = confusion.diag().float()
    union = (
        confusion.sum(dim=1)
        + confusion.sum(dim=0)
        - true_positive
    )
    iou = true_positive / union.clamp_min(1)
    mean_iou = iou.mean().item()

    depth_mae = depth_abs_error / max(depth_count, 1)
    lane_mae = lane_abs_error / max(lane_count, 1)

    return mean_iou, depth_mae, lane_mae


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    segmentation_criterion = nn.CrossEntropyLoss()
    depth_criterion = nn.L1Loss()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["track"].to(device).long()
        depths = batch["depth"].to(device).float()

        optimizer.zero_grad(set_to_none=True)

        logits, predicted_depth = model(images)

        segmentation_loss = segmentation_criterion(logits, masks)
        depth_loss = depth_criterion(predicted_depth, depths)

        loss = segmentation_loss + depth_loss

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Run this script from the main directory containing drive_data/.
    train_loader = load_data(
        "drive_data/train",
        transform_pipeline="default",
        batch_size=32,
        num_workers=0,
        shuffle=True,
    )

    val_loader = load_data(
        "drive_data/val",
        transform_pipeline="default",
        batch_size=32,
        num_workers=0,
        shuffle=False,
    )

    model = Detector().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    num_epochs = 30
    best_miou = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device
        )

        miou, depth_mae, lane_mae = evaluate(
            model, val_loader, device
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss: {train_loss:.4f} | "
            f"mIoU: {miou:.4f} | "
            f"Depth MAE: {depth_mae:.4f} | "
            f"Lane MAE: {lane_mae:.4f}"
        )

        if miou > best_miou:
            best_miou = miou
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    save_model(model)

    print(f"Best validation mIoU: {best_miou:.4f}")
    print("Saved weights to homework/detector.th")


if __name__ == "__main__":
    main()
