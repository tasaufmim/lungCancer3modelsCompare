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
    ├── Benign cases/
    ├── Malignant cases/
    └── Normal cases/

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
Benign cases: 600
Malignant cases: 600
Normal cases: 600

The dataset was then split:
Training Set: 80% (1440 images)
Validation Set: 20% (60 images)

Normalization and Data loading:
torchvision.datasets.ImageFolder is used to load images and normalized with
transforms.Normalize([0.485, 0.456, 0.406],  # Mean (ImageNet)
                     [0.229, 0.224, 0.225])  # Std  (ImageNet)

Supported Formats:
Only image with extensions .png, .jpg, .jpeg were used

Outputs:
Plots:
  Accuracy and loss over epochs
  ROC curves
  Confusion matrices
Reports:
  Classification reports
  Evaluation tables (accuracy, precision, recall, F1)
