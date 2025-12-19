# Low-Light Image Enhancement (MRPNet & Combined Architecture)

This project contains the complete code for training a Deep Learning model to perform Low-Light Image Enhancement. The solution combines a denoising network (MRPNet) and a super-resolution generator to restore clean, high-quality images from noisy, low-light inputs.

## 🚀 Key Features
The pipeline is engineered for high-fidelity image restoration within a Kaggle environment, utilizing a combined architecture for simultaneous denoising and enhancement.

  * Dual-Stage Architecture: Combines a custom MRPNet (Denoise) and a Generator (Super-Resolution) into a single end-to-end differentiable pipeline.

  * Residual Learning: The Generator utilizes residual blocks and skip connections to preserve high-frequency details while upscaling and enhancing features.

  * Mixed Precision Training: Implements torch.amp (Automatic Mixed Precision) to accelerate training and reduce GPU memory usage without compromising model        accuracy.

  * Robust Data Handling: Features custom Dataset classes for paired training (noisy/clean) and efficient data loading with DataLoader.


## 📈 Methodology
The core of this solution is a sequential refinement process where the image is first denoised and then enhanced for perceptual quality.

### 1. Model Architecture (Combined Model)

The model consists of two distinct sub-networks trained jointly:

  * **MRPNet (Denoiser)**: A compact Encoder-Decoder network.

    * **Encoder**: Convolutional layers with ReLU to extract features and suppress noise.
    * **Decoder**: Transposed convolutions to reconstruct the denoised spatial representation.

  * **Generator (Enhancer)**: A deeper network inspired by Super-Resolution architectures.

    * **Residual Blocks**: Conv2d + BatchNorm + ReLU blocks with skip connections (x+residual) to prevent gradient vanishing.
    * **PixelShuffle Upsampling**: Increases resolution/feature clarity effectively without the checkerboard artifacts often caused by standard deconvolution.

### 2. Training Strategy

  * Loss Function: Mean Squared Error (MSE) to minimize the pixel-wise difference between the enhanced output and the ground truth (GT) image.

  * Optimizer: Adam with a learning rate of $10^{-4}$

  * Checkpointing: Automatically saves the best model weights (best_model.pth) whenever the validation loss improves.

  * Resume Capability: The script supports loading pre-trained weights to continue training or fine-tune existing models.

### 3. Inference & Submission

  * Batch Inference: Predicts on the test set using a dedicated test_loader and saves the enhanced images to a temporary directory.

  * Post-Processing: Converts the enhanced images into grayscale, flattens them, and samples pixels (stride 8) to match the specific CSV format required by the     competition.


## 🛠️ Tech Stack

* Core: Python 3
* Deep Learning: PyTorch (nn, optim, autograd)
* Computer Vision: Torchvision, PIL (Pillow)
* Data Handling: NumPy, Pandas, Tqdm

## 🏃 Running the Project

### 1. Dependencies

This script is designed to run in a Kaggle Notebook environment. Ensure the dataset path matches the competition directory structure.

```
pip install torch torchvision numpy pandas pillow kaggle
```

### 2. Dataset

This model was trained on a low-light image enhancement dataset as part of a university challenge. The data consists of pairs of noisy, low-light images and their corresponding clean, ground-truth (GT) images. Due to privacy and access restrictions, the dataset is not publicly available and is not included in this repository.

Therefore, the script cannot be run out-of-the-box without downloading the specific competition data separately and placing it in the correct directory structure (e.g., train/train for input and train/gt for target images).

### 3. Notebook Review

The provided code serves as an end-to-end pipeline:

* **Data Loading**: Initializes datasets and loaders for Train, Validation, and Test splits.

* **Model Build**: Instantiates the CombinedModel and moves it to the GPU.

* **Training**: Runs the training loop with mixed precision and saves the best checkpoints.

* **Inference**: Generates predictions on the test set.



