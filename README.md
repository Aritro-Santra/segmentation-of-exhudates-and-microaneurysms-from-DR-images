# Image Processing with Eigenvalue Analysis and Medical Image Segmentation  

## Overview  
This project combines advanced image analysis techniques with medical image segmentation to extract meaningful insights from visual data. The application is divided into two primary components:  

1. **Image Eigenvalue Analyzer**  
   - Analyzes eigenvalues of image channels (grayscale, red, green, and blue).  
   - Computes Euclidean distances between eigenvalue sets of different channels to evaluate their similarity.  
   - Provides insights into the relationship between color channels and grayscale representation.  

2. **Medical Image Segmentation**  
   - Implements a U-Net-based segmentation model for detecting:  
     - **Exudates**  
     - **Microaneurysms**  
   - Designed for medical image analysis, particularly in identifying features in retinal scans.  

---

## Features  

### 1. **Image Eigenvalue Analyzer**  
- Loads and preprocesses an image.  
- Resizes the image to a uniform dimension of 256×256.  
- Computes eigenvalues for grayscale and RGB channels.  
- Measures Euclidean distances between eigenvalues of grayscale and RGB channels.  
- Identifies and displays:  
  - Channel pairs with minimum and maximum similarity.  

### 2. **Segmentation with U-Net**  
- Utilizes a deep learning U-Net architecture for pixel-wise segmentation.  
- Detects medical features such as exudates and microaneurysms from input medical images.  
- Supports training, validation, and inference on custom datasets.
