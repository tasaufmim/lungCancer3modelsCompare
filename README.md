## Paper Title: Deep CNN Models for Reliable Lung Cancer Classification: A Comparative Study on IQ-OTH/NCCD Dataset

**Abstract:**
Due in large part to delayed detection and diffculties correctly differentiating between benign, malignant, and healthy tissues in CT scans, lung cancer remains the leading cause of cancer-related deaths globally. This study uses the publicly available IQ-OTH/NCCD CT scan dataset to present a deep learning-based approach for the multiclass classification of lung cancer. In order to address class imbalance, we evaluate the performance of three well-known CNN architectures— VGG16, VGG19, and ResNet50—using a consistent training environment and applied data augmentation. To improve model performance, the dataset—which included 1,097 CT scans divided into three classes— was preprocessed, augmented, and resized. The scans were classified as benign, malignant, and normal. According to the experimental data, all three models had outstanding classification accuracy; however, ResNet50 outperformed the others, achieving 99.33% accuracy, 99.00% recall, and 98.66% precision. The suggested model’s accuracy and stability are confirmed by a comparative benchmarking with state-of-the-art methods using the same data corpus. These results highlight the effectiveness of deep CNNs—ResNet50 in particular—for reliable and automated lung cancer diagnosis, which will help clinical decision support systems develop.

---

**lung Cancer Detection using Deep CNN Architecture**

I haved utilited 3 pretrained model to classify lung Cancer.

Models Used:  i. VGG16
              ii. ResNet50
              iii. VGG19

Dataset Used: https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset

Classes:      i, Benign cases
              ii. Malignant cases
              iii. Normal cases


Dataset Strucgture: The dataset contains CT scan images categorized into three directories like below:
/The IQ-OTHNCCD lung cancer dataset/
```bash
    ├── Benign cases/
    ├── Malignant cases/
    └── Normal cases/
```
d
Dataset Distribution (Initial):
Benign cases: 120
Malignant cases: 561
Normal cases: 416

Image Preprocessing:
Before going to the training process, all images are:
  - Converted to RGB mode
  - Resized to 512 x 512 px for the augmentation process
  - For VGG and ResNet's input requirements again resized to 224 x 224 px

Data Augmentation:
To increase the robustness of the model and balance the dataset, PIL.Image is used to apply the data augmentation. Synthetic images were generated using random transfomations:

| Transformation        | Details                    |
| --------------------- | -------------------------- |
| Horizontal Flip       | `Image.FLIP_LEFT_RIGHT`    |
| Vertical Flip         | `Image.FLIP_TOP_BOTTOM`    |
| Random Rotation       | Angle between -25° and 25° |
| Contrast Enhancement  | Factor between 1.2 and 1.8 |
| Color Enhancement     | Factor between 1.2 and 2.0 |
| Sharpness Enhancement | Factor between 1.5 and 2.5 |

After augmentation, Target count: 600 images each class.

Dataset Distribution after augmentation:
```bash
Benign cases: 600
Malignant cases: 600
Normal cases: 600
```

The dataset was then split:
```bash
Training Set: 80% (1440 images)
Validation Set: 20% (60 images)
```

Normalization and Data loading:
```bash
torchvision.datasets.ImageFolder is used to load images and normalized with
transforms.Normalize([0.485, 0.456, 0.406],  # Mean (ImageNet)
                     [0.229, 0.224, 0.225])  # Std  (ImageNet)
```

Supported Formats:
Only image with extensions .png, .jpg, .jpeg were used

Outputs:
Plots:
```bash
  Accuracy and loss over epochs
  ROC curves
  Confusion matrices
```
Reports:
```bash
  Classification reports
  Evaluation tables (accuracy, precision, recall, F1)
```
