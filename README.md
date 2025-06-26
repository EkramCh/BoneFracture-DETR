# BoneFracture-DETR 🦴📦  
**Bone Fracture Detection using Detection Transformer (DETR)**

This project applies Facebook's **DETR (Detection Transformer)** for automated detection of bone fractures from X-ray images. The model is fine-tuned on a custom dataset using COCO-style annotations.

> 🔍 **Goal:** Detect bone fracture regions in X-ray images using transformer-based object detection.

---

## 📁 Project Structure

- `BoneFracture_DETR.ipynb`: Main notebook for training and inference using DETR.
- `datasets/`: Folder to store downloaded image dataset and COCO annotations.
- `outputs/`: (Optional) Trained model weights and sample predictions.

---

## 📦 Dataset

- **X-ray Images**:  
  [Kaggle Dataset – pkdarabi/bone-fracture-detection-computer-vision-project](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)

- **COCO-style Annotations**:  
  [Kaggle Dataset – banddaniel/bone-fracture-detection-detection-coco-annots](https://www.kaggle.com/datasets/banddaniel/bone-fracture-detection-detection-coco-annots)

> Download both datasets and organize them as:
```
datasets/
├── test/
│   ├── annotations.json
│   ├── image1.jpg...
│   ├── ...
├── train/
│   ├── annotations.json
│   ├── image1.jpg...
│   ├── ...
├── valid/
│   ├── annotations.json
│   ├── image1.jpg...
│   ├── ...

---

## 📚 References

- 📄 **Original DETR Paper**:  
  [End-to-End Object Detection with Transformers (Carion et al., 2020)](https://arxiv.org/abs/2005.12872)

- 🧠 **DETR GitHub Repository**:  
  [facebookresearch/detr](https://github.com/facebookresearch/detr)

- 📝 **Notebook Template** (Used as reference):  
  [Fine-tuning DETR on custom dataset (balloon)](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb)

---

## 🚀 How to Use

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bone-fracture-detection-computer-vision-project.git
cd bone-fracture-detection-computer-vision-project
```

2. Install required libraries:
```bash
pip install torch torchvision transformers pytorch-lightning cython pycocotools supervision
```

3. Run the notebook:
> Open `BoneFracture_DETR.ipynb` and execute all cells to train and visualize predictions.

---

## 📌 Notes

- The model is trained using transfer learning with pretrained `facebook/detr-resnet-50`.
- COCO format annotations make the integration with `DetrForObjectDetection` seamless.
- Predictions include bounding boxes and class probabilities for fracture detection.

---

## 📸 Sample Output 

<img src="Predection Fracture Detection using DETR.jpg" alt="Fracture Detection" width="600"/>

<p align="center"><em>Figure 1: DETR prediction with bounding boxes on fractured bone</em></p>

---
