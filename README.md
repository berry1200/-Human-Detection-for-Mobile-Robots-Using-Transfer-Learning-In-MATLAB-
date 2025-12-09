# Human Detection for Mobile Robots Using CNN (MATLAB)

This project implements a simple **human vs nonhuman image classifier** using
**Convolutional Neural Networks (CNNs)** and **transfer learning** in MATLAB.

It is designed for mobile robots or camera systems that only need to know:
> “Is there a human in this image or not?”

The final model reaches **94.17% accuracy** on the test set.

---

## 1. Requirements

- MATLAB (R2022a or later recommended)
- Deep Learning Toolbox
- Image Processing Toolbox
- (Optional) GPU support for faster training

---

## 2. Project Structure

```text
.
├─ humanTransferNet_aug.mat        # Trained CNN model (human vs nonhuman)
├─ demo_image_file.m               # Simple demo: classify a saved image
├─ create_brightness_aug.m         # (Optional) Generate augmented low-light images
├─ prepare_data_aug.m              # Prepare augmented datastores from Dataset/
├─ train_transfer_aug.m            # Train transfer-learning CNN
├─ evaluate_improved_aug.m         # Evaluate improved model
├─ step12_evaluate_full.m          # Full evaluation (metrics, confusion matrix, ROC)
├─ show_random_predictions.m       # Show random prediction results
└─ Dataset/
   ├─ human/                       # Put your human images here
   └─ nonhuman/                    # Put your non-human images here

** 3. Quick Start (Use Pretrained Model) **

If you only want to test the system, you do not need to retrain.

Step 1 – Open MATLAB in this folder

In MATLAB, set the current folder to the project root.

Step 2 – Put a test image

Add an image file in the project folder, for example:

test_image.jpg

It can be any JPG image that either contains a person or not.

Step 3 – Run the demo script

In the MATLAB Command Window:

** demo_image_file **


The script will:

Load humanTransferNet_aug.mat

Read test_image.jpg

Resize it to the correct input size

Classify it as human or nonhuman

Show the image with the prediction and confidence score in the title
