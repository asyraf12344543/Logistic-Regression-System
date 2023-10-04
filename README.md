# Binary and Multiclass Classification Project
This project involves training a logistic regression binary classifier and a multiclass classifier on respective datasets. The project is divided into two main tasks: binary classification and multiclass classification. Each task is explained in detail below.

# Task 1: Binary Classifier
# Dataset Preparation
1. The binary classification dataset is used for this task.
2. The dataset is divided into three subsets:
a) Testing (TST): 20% of the overall dataset. Not normalized or standardized.
b) Training (TRN): 35% of the normalized/standardized dataset.
c) Validation (VLD): 65% of the normalized/standardized dataset.

# Logistic Regression with Gradient Descent
1. A logistic regression classifier is trained using gradient descent (GD) on the TRN dataset.
2. At each iteration, the cost is calculated based on both the TRN and VLD datasets. (VLD is not used for GD parameter updates).
3. The training process involves optimizing the model's parameters.
4. A plot is generated showing the training and validation losses/costs on a single chart.
   
# Performance Metrics
1. A confusion matrix is generated for the validation set.
2. The following performance metrics are calculated:
a) Accuracy
b) Precision and Recall
c) F-Score
d) True Positive Rate (TPR)
e) False Positive Rate (FPR)

# Classifier Improvement
1. After obtaining the initial metrics, attempts are made to improve the classifier.
2. Changes can only be made to hyperparameters or something from the dataset preparation step onwards (e.g., data preprocessing).
3. Classifier Deployment and Testing
4. Both the initial and improved classifiers are deployed on the TST set.
5. TST is normalized based on the parameters from the training set but not on itself.
6. The following metrics are calculated for both classifiers on the TST set:
a) Accuracy
b) Precision and Recall
c) F-Score
d) True Positive Rate (TPR)
e) False Positive Rate (FPR)

# Task 2: Multiclass Classifier
# Dataset Selection
A multiclass dataset from the previous lab is chosen for this task.

# Training Multiclass Classifiers
The necessary number of classifiers is trained to enable logistic regression for multiclass classification.
During the presentation session, input instances will be used to evaluate classifier performance.

# Overall Accuracy
The overall accuracy is calculated for each multiclass classifier.

# Testing Set Preparation
Each class in the dataset is divided into 80% training and 20% testing instances.
The instances mentioned during the presentation session will be selected from the testing set.

This project aims to demonstrate binary and multiclass classification using logistic regression and provides insights into optimizing classifier performance. It involves dataset preparation, model training, evaluation, and testing for both binary and multiclass classification tasks.
