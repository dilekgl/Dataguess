import torch
import torchvision
import time
import os
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Category IDs to be used (e.g., 11: white-pawn, 5: black-pawn)
TARGET_CATEGORIES = {11, 5}

# Custom transform class to convert PIL images to tensors
class CocoTransform:
    def __call__(self, image, target):
        image = F.to_tensor(image)  
        return image, target

# Load COCO-format dataset with custom transformation
def get_coco_dataset(img_dir, ann_file):
    return CocoDetection(
        root=img_dir,
        annFile=ann_file,
        transforms=CocoTransform()
    )

# Load training and validation datasets
train_dataset = get_coco_dataset(
    img_dir="fastrcnn/chess-pieces-fastrcnn/train/images",
    ann_file="chess-pieces-coco/train/_annotations.coco.json"
)

val_dataset = get_coco_dataset(
    img_dir="fastrcnn/chess-pieces-fastrcnn/valid/images",
    ann_file="chess-pieces-coco/valid/_annotations.coco.json"
)

# Data loaders with batch size and custom collate function to handle variable target sizes
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))   

# Load pretrained Faster R-CNN and replace the classifier head for custom number of classes
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Set number of classes: background + white-pawn + black-pawn
num_classes = 3
model = get_model(num_classes)

# Move model to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define optimizer and learning rate scheduler
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Training function for one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        processed_targets = []
        valid_images = []

        # Process each target annotation to extract bounding boxes and labels
        for i, target in enumerate(targets):
            boxes, labels = [], []
            for obj in target:
                x, y, w, h = obj["bbox"]
                if w > 0 and h > 0 and obj["category_id"] in TARGET_CATEGORIES:
                    boxes.append([x, y, x + w, y + h])
                    labels.append(1 if obj["category_id"] == 11 else 2)

            # Only add valid targets (with at least one box)
            if boxes:
                processed_targets.append({
                    "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                })
                valid_images.append(images[i])

        # Skip batch if no valid targets
        if not processed_targets:
            continue

        images = valid_images
        loss_dict = model(images, processed_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch [{epoch}] Loss: {losses.item():.4f}")

# Evaluation function for a saved model
def evaluate_model(model, data_loader, device, model_path):
    # Load model weights from file
    model.load_state_dict(torch.load(model_path))
    model.eval()

    metric = MeanAveragePrecision()
    inference_times = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]

            ground_truths = []
            valid_indices = []

            # Convert COCO annotations to expected format
            for i, target in enumerate(targets):
                boxes, labels = [], []
                for obj in target:
                    if obj["category_id"] in TARGET_CATEGORIES:
                        x, y, w, h = obj["bbox"]
                        boxes.append([x, y, x + w, y + h])
                        labels.append(1 if obj["category_id"] == 11 else 2)
                
                if boxes:
                    ground_truths.append({
                        "boxes": torch.tensor(boxes, dtype=torch.float32).to(device),
                        "labels": torch.tensor(labels, dtype=torch.int64).to(device),
                    })
                    valid_indices.append(i)

            # Skip batch if no valid targets
            if not valid_indices:
                continue

            valid_images = [images[i] for i in valid_indices]

            # Measure inference time
            start_time = time.time()
            outputs = model(valid_images)
            end_time = time.time()

            predictions = []
            for output in outputs:
                predictions.append({
                    "boxes": output["boxes"].to(device),
                    "labels": output["labels"].to(device),
                    "scores": output["scores"].to(device),
                })

            # Update metric with predictions and ground truths
            metric.update(predictions, ground_truths)
            inference_times.append((end_time - start_time) * 1000)  # ms

    # Compute final mAP metrics
    map_results = metric.compute()
    avg_inference_time = sum(inference_times) / len(inference_times)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"mAP: {map_results['map']:.4f}")
    print(f"mAP@50: {map_results['map_50']:.4f}")
    print(f"mAP@75: {map_results['map_75']:.4f}")
    print(f"Average Inference Time: {avg_inference_time:.2f} ms")
    print(f"Model Size: {model_size_mb:.2f} MB")

# Train and save the model for each epoch
num_epochs = 5
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch)
    lr_scheduler.step()
    
    model_path = f"fasterrcnn_resnet50_epoch_{epoch + 1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    # Evaluate the model after each epoch
    print(f"\nEvaluating Model: {model_path}")
    evaluate_model(model, val_loader, device, model_path)
    print("\n" + "=" * 50 + "\n")
