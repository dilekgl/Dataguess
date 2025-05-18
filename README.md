# Chess Piece Detection with Faster R-CNN

This project aims to detect chess pieces using Faster R-CNN. The model is trained on COCO-formatted annotations and focuses solely on detecting `white-pawn` and `black-pawn` classes.

## 📈 Project Summary

* **Model**: Faster R-CNN (ResNet50-FPN)
* **Framework**: PyTorch & Torchvision
* **Data Format**: COCO (with separate train, val, and test sets)
* **Target Classes**: `white-pawn (id=11)`, `black-pawn (id=5)`

---

## 🔧 Installation

```bash
pip install torch torchvision torchmetrics pycocotools
```

---

## ⚙️ Technologies Used

* **Python**
* **PyTorch**: Deep learning framework
* **Torchvision**: Pre-built detection models like Faster R-CNN
* **Torchmetrics**: For evaluation metrics such as mAP
* **pycocotools**: For handling COCO-style annotations

---

## ⚙️ Dataset Preparation

* The dataset is in COCO format and includes three sets: `train`, `val`, and `test`.
* Annotations are filtered to include only two classes: `white-pawn` and `black-pawn`.
* Category IDs have been manually adjusted to fit model requirements.

---

## ⚙️ Training Parameters

* Number of epochs: 5
* Batch size: 2
* Learning rate: 0.005
* Optimizer: SGD (momentum=0.9, weight\_decay=0.0005)
* Learning rate scheduler: StepLR

```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

---

## 🔄 Model Training

```bash
python model_training.py
```

* Training script reads annotations, filters relevant categories, and trains the model on the dataset.
* The trained model is saved after each epoch in the `output/` directory.

---

## 🧪 Testing & Inference

```bash
python test.py
```

* Performs inference on test images
* Displays bounding boxes with labels and confidence scores
* Optionally saves output images to disk

---

## 🔬 Performance Evaluation

* Metric used: **Mean Average Precision (mAP)**
* Evaluation script reports:

  * mAP
  * mAP\@50
  * mAP\@75
  * Average inference time
  * Model size

---

## 💡 Algorithm Choice and Methods

### Why Faster R-CNN?

* Balances speed and accuracy effectively
* Especially strong at detecting small and medium-sized objects like chess pieces
* Region Proposal Network (RPN) ensures precise localization

### Why ResNet50 + FPN?

* ResNet50 ensures strong feature extraction
* Feature Pyramid Network (FPN) enhances detection across multiple scales

---

## 🏭 Real-World Applications

### 1. Industrial Vision Systems

* Differentiate between parts based on shape or color

### 2. Quality Control

* Detect faulty or missing components on production lines

### 3. Inventory Management

* Automatically recognize and count inventory items

---

## 💾 Model Saving

* Trained model is saved in `.pth` format after each epoch
* Saved under the `output/` directory

---
