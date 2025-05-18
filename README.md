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
* **PyTorch**: Chosen for its dynamic computation graph and ease of use, making it suitable for research and production.
* **Torchvision**: Pre-built detection models like Faster R-CNN
* **Torchmetrics**: For evaluation metrics such as mAP
* **pycocotools**: For handling COCO-style annotations
* **Faster R-CNN**: A two-stage object detection model that first proposes regions and then classifies them. It's effective for detecting objects with varying sizes and aspect ratios.
* **Transfer Learning**: Utilizing a pre-trained ResNet-50 backbone to leverage learned features, reducing training time and improving performance.
* **Mean Average Precision (mAP)**: Used as the evaluation metric to assess the precision and recall of the model across different thresholds.





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

* Automated detection of defects or anomalies in products on assembly lines, ensuring consistent product quality.

### 3. Inventory Management

* Real-time tracking of products and materials using object detection to maintain optimal inventory levels.

---

## 💾 Model Saving

* Trained model is saved in `.pth` format after each epoch
* Saved under the `output/` directory

---
