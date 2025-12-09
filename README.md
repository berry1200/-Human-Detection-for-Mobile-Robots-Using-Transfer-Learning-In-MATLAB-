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

##https://drive.google.com/file/d/1eM6iUeSCc9ax_TdZWzIWGV8puRMVdyz1/view?usp=sharing
##Note: The trained model (humanTransferNet_aug.mat) is large (~80 MB).
##If it cannot be downloaded  from GitHub, use the Google Drive link.

#_3. Quick Start (Recommended) — Use the Pretrained Model_

If you want to test the system immediately, you only need:

humanTransferNet_aug.mat

demo_image_file.m

Step 1 — Open MATLAB in this project folder

Set the current MATLAB directory to the GitHub project location.

Step 2 — Add a test image

Place an image such as:

test_image.jpg


in the project folder.
This image may contain a human or a nonhuman scene.

Step 3 — Run the demo

In the MATLAB Command Window:

demo_image_file


The script will:

Load the trained model (humanTransferNet_aug.mat)

Read test_image.jpg

Resize it to the required CNN input size

Classify it as human or nonhuman

Display the image with prediction and confidence score

This demo works 100% reliably and is recommended for beginners.

#4. Full Training Workflow (Optional)

If you want to train the model from scratch, follow all steps below.

Step 1 — Prepare the Dataset

Create the following folder structure:

Dataset/
   human/
      (your images containing humans)
   nonhuman/
      (your images without humans)


Use at least 50–100 images per class for good results.

Step 2 — (Optional) Generate Low-Light & Augmented Images

To improve robustness in dark and complex environments:

create_brightness_aug


This generates:

Dataset_aug/


containing extra rotated, scaled, and low-light images.

Step 3 — Prepare Datastores

Run:

prepare_data_aug


This creates:

datastores_aug.mat


containing training and testing sets ready for MATLAB’s CNN functions.

Step 4 — Train the Transfer-Learning Model

Train the CNN using:

train_transfer_aug


This script:

Loads a pretrained CNN (ResNet-18 / AlexNet)

Replaces the final layers for binary classification

Trains the model using your dataset

Saves the result as:

humanTransferNet_aug.mat

Step 5 — Evaluate the Model

Run:

step12_evaluate_full


This generates:

Accuracy (%)

Precision, Recall, F1-score

Confusion Matrix

ROC Curve (if supported)

Runtime speed (ms/frame + FPS)

You can also visualize random predictions using:

show_random_predictions



#Google drive link for entire code if github is not working:   **https://drive.google.com/drive/folders/14RfRH9-jKCEfSbth5O1ivdapDADY2AN6?usp=sharing**
