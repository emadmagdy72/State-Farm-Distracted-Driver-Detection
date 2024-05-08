# Project Overview

This project aims to develop  deep learning models for classifying distracted behaviors in drivers from dashboard camera images. The project encompasses the following key components:

## Baseline Models

- Developed baseline Dense layers and Convolutional Neural Network (CNN) models to classify distracted behaviors exhibited by drivers. These models serve as the foundation for further experimentation and optimization.

## Data Augmentation

- Implemented data augmentation techniques to augment the dataset, thereby enhancing model generalization and robustness to variations in input data.

## Transfer Learning

Explored transfer learning techniques utilizing pre-trained models such as VGG16. By leveraging pre-trained models, model training time is expedited, and classification accuracy is potentially improved.

After applying transfer learning using pre-trained models such as VGG16, we observed the following improvements in model performance:

- ![deep learning projct](https://github.com/emadmagdy72/State-Farm-Distracted-Driver-Detection/assets/67216285/3b87d577-793a-4209-9ccb-5f359db44bcc)

The plot above illustrates the training and validation accuracy over 25 epochs. The model achieved higher accuracy on both the training and validation sets as training progressed, indicating successful transfer learning.

## Tools Used

- **NumPy:** Used for numerical computing and array operations.
- **Scikit-Learn:** Employed for machine learning tasks such as data preprocessing and evaluation.
- **Pandas:** Utilized for data manipulation and analysis, especially for handling structured data.
- **TensorFlow:** Deep learning framework used for building and training neural network models.
- **Keras:** High-level neural networks API for rapid prototyping and experimentation with deep learning models.
- **OpenCV:** Library for computer vision tasks, including image processing and manipulation.

Feel free to explore the project repository for more details and contributions!
