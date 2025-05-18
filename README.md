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
pip install torch torchvision torchmetrics
```

---

## ⚖️ Technologies Used

* **Python**
* **PyTorch**: Deep learning framework
* **Torchvision**: Pre-built detection models like Faster R-CNN
* **Torchmetrics**: For evaluation metrics such as mAP

---

## ⚙️ Training Parameters

* Number of epochs: 5
* Number of classes: 3 (background + 2 pieces)
* Optimizer: SGD (momentum=0.9, weight\_decay=0.0005)

```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

---

## 🔄 Model Training

```bash
python train.py
```

The code filters annotations to include only the target categories. COCO format annotations are adjusted accordingly.

---

## 🔬 Performance Evaluation

* Metric used: **Mean Average Precision (mAP)**
* Detailed scores:

  * mAP
  * mAP\@50
  * mAP\@75
* Average inference time and model size are also computed.

---

## 💡 Algorithm Choice and Methods

### Why Faster R-CNN?

* High accuracy
* Excellent performance on small objects (e.g., pawns)
* Widely used in industrial settings

### Why ResNet50 + FPN?

* Combines multi-scale features to improve detection performance

---

## 🏭 Real-World Applications

### 1. Industrial Vision Systems

* Differentiating between products with varying colors/shapes

### 2. Quality Control

* Detecting defective or faulty components

### 3. Autonomous Robotics

* Detecting objects to enable appropriate actions

### 4. Inventory Management

* Recognizing different items to update stock levels automatically

### 5. Assembly Line Automation

* Controlling robotic arms through part recognition

---

## 📁 Saving the Model

* The model is saved in `.pth` format at the end of each epoch.
* Saved under the "models" directory.

---

