# mobile-plan-classifier
This project focuses on building a machine learning classification model to help the mobile carrier Megaline recommend newer mobile plans — Smart or Ultra — to its subscribers. Many users are still on legacy plans, and Megaline aims to use behavioral data to guide customers toward the most suitable modern plan.

## 📝 Project Overview
Megaline has provided behavioral data on subscribers who have already switched to either the Smart or Ultra plan. Using this data, the goal is to:

- Develop a classification model that predicts whether a user should be on the Smart (0) or Ultra (1) plan.
- Achieve an accuracy of at least 75% on the test dataset.
- Evaluate different machine learning algorithms and tune their hyperparameters for optimal performance.

## 📂 Dataset Information
File Path: datasets/users_behavior.csv

Observations: Each row represents monthly usage behavior of a single user.

Features:

calls: Number of calls made

minutes: Total duration of calls (in minutes)

messages: Number of SMS messages sent

mb_used: Internet usage in MB

is_ultra: Target variable (1 = Ultra plan, 0 = Smart plan)

## ✅ Project Goal
- Load and explore the dataset to understand the distribution and nature of the data.

- Split the data into training, validation, and test sets.

- Train multiple classification models and experiment with different hyperparameters.

- Evaluate the models using the validation set and select the best-performing one.

- Test the final model on the test set and ensure that it meets the 0.75 accuracy threshold.

- Perform a sanity check to validate that the model is reasonable and not overfitting.
