# Med-VQA System for Analyzing Polyps in Gastrointestinal Endoscopy Images [(Paper)](./Paper.pdf)

## Overview
The **Med-VQA System** is an innovative Visual Question Answering (VQA) platform designed to enhance the analysis of polyps in gastrointestinal (GI) endoscopy images. The system leverages cutting-edge deep learning techniques to assist medical professionals in diagnosing and assessing polyps with improved accuracy and efficiency.

## Key Features
- **Advanced Image Processing:** Utilizes ResNet to extract critical features from GI endoscopy images.
- **Multimodal Learning:** Integrates visual and textual data, enabling the system to address complex medical inquiries effectively.
- **Feature Fusion Techniques:** Employs element-wise multiplication to highlight feature interactions, achieving better performance compared to simple concatenation methods.
- **Robust Classification Framework:** Features a Multilayer Perceptron (MLP) architecture with dropout regularization and Leaky ReLU activation to ensure stable and efficient training.

## Methodology
- **Dataset:** The system is trained and validated using the ImageCLEFmed-Med-VQA-GI-2023 dataset.
- **Preprocessing:** Implements image resizing to a standardized dimension of 800x800 pixels and applies pixel normalization across color channels to enhance model training stability.
- **Model Evaluation:** Performance is assessed using metrics like accuracy and BLEU-1 score, evaluating both the system's answers and its ability to generalize across different inputs.

## Experiments and Results
- **Regularization Studies:** Analyzes the impact of weight decay regularization on model performance, finding significant improvements in prediction stability and accuracy.
- **Performance Optimizations:** Demonstrates that feature interaction through element-wise multiplication boosts Med-VQA system performance, surpassing conventional approaches.

## References
This study builds on a foundation of extensive research and methodologies from the fields of deep learning and medical image analysis. Key contributions include advancements in CNN-based image processing, ensemble learning for medical diagnostics, and comprehensive evaluations in medical visual question answering.
