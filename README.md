# Brain Tumor MRI Classification

This project presents a deep learning system for classifying brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The system includes data preprocessing, baseline CNN modeling, transfer learning using MobileNetV2 and ResNet50, model evaluation, and a Streamlit web application for prediction and Grad-CAM explainability.

## Project Overview

The main goal of this project is to compare different deep learning approaches for brain tumor MRI classification and identify the best-performing model.

The project includes:

- A baseline CNN model trained from scratch
- MobileNetV2 transfer learning experiments
- ResNet50 transfer learning experiments
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves
- Grad-CAM visualization to explain model predictions
- A Streamlit application for uploading MRI images and displaying prediction results

## Project Pipeline

1. **Data Preparation**  
   MRI images are organized into training, validation, and testing sets.

2. **Data Exploration**  
   The dataset is inspected by visualizing sample images, checking class distribution, and understanding image characteristics.

3. **Data Preprocessing**  
   Images are resized, converted to RGB, transformed into tensors, and normalized using ImageNet statistics.

4. **Baseline CNN**  
   A simple CNN model is trained as a starting reference point.

5. **Transfer Learning**  
   Pretrained MobileNetV2 and ResNet50 models are fine-tuned to improve classification performance.

6. **Model Evaluation**  
   All models are evaluated on the test set using accuracy, precision, recall, F1-score, ROC-AUC, confusion matrices, and ROC curves.

7. **Explainability**  
   Grad-CAM is used to visualize the image regions that influenced the model prediction.

8. **Deployment Application**  
   A Streamlit application is provided to upload an MRI image, predict its class, display confidence scores, and show Grad-CAM focus areas.

## Best Model

Based on the final evaluation, the best-performing model is:

**ResNet50 Deeper Fine-Tuning**

This model achieved the highest performance among the tested models and was selected for the Streamlit application.

## Project Structure

```text
brain-tumor-mri-classification/
│
├── data/
│   └── raw/
│
├── models/
│
├── notebooks/
│
├── src/
│
├── app.py
├── requirements.txt
└── README.md
```

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/amrasmar27/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

After running the command, open the local Streamlit link in the browser.

The application allows the user to:

- Upload a brain MRI image
- View the predicted tumor class
- See top-3 prediction confidence scores
- View the uncertainty score
- Display Grad-CAM explanation for the prediction

## Notes

The processed data files are not tracked in Git because they can be generated during preprocessing. The raw dataset and preprocessing notebook are used to recreate the processed splits when needed.

## Conclusion

This project demonstrates that transfer learning improves brain tumor MRI classification compared with a CNN trained from scratch. ResNet50 Deeper Fine-Tuning achieved the strongest evaluation performance and was integrated into the final prediction application.
