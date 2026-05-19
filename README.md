## Brain Tumor MRI Classification

This project uses deep learning to classify brain MRI images into four categories:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor

The project includes data preprocessing, CNN models, and transfer learning approaches.

## Project Pipeline

1. Data Collection
   - Load MRI brain tumor dataset
   - Organize training and testing images

2. Data Exploration
   - Visualize MRI image samples
   - Analyze class distribution
   - Inspect image sizes and quality

3. Data Preprocessing
   - Resize images
   - Normalize pixel values
   - Convert images into arrays
   - Apply data augmentation

4. Model Development
   - Build a baseline CNN model
   - Train and validate the model

5. Transfer Learning
   - Use pretrained models such as MobileNetV2 or ResNet50
   - Fine-tune the model for MRI classification

6. Model Evaluation
   - Accuracy and loss analysis
   - Confusion matrix
   - Classification report

7. Prediction & Inference
   - Predict tumor class for unseen MRI images

8. Result Analysis
   - Compare CNN and transfer learning performance
   - Discuss strengths and limitations
