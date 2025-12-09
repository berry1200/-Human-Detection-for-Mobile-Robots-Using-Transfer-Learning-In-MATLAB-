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
