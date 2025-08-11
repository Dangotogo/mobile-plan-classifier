# Project overview
Megaline, a mobile carrier, wants to recommend updated mobile plans — Smart or Ultra — to users still on legacy plans. To achieve this, we’ll build a classification model using user behavior data to predict the most suitable plan.

### The objective is to:

- Predict the correct plan (Smart or Ultra) based on usage data.
- Achieve at least 75% accuracy on the test set.
- Evaluate different models and tune hyperparameters.
- Validate the model with a sanity check to ensure logical performance.
- This project uses preprocessed data and focuses on model development and evaluation.

  
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
```


```python
# Load data
df = pd.read_csv('users_behavior.csv')

# Features/target
features = df.drop(['is_ultra'], axis=1)
target = df['is_ultra']

# Split train+val/test
features_train_val, features_test, target_train_val, target_test = train_test_split(
    features, target, test_size=0.2, random_state=12345)

# Split train/val
features_train, features_valid, target_train, target_valid = train_test_split(
    features_train_val, target_train_val, test_size=0.25, random_state=12345)

```


```python
# Models
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=12345),
    'Random Forest': RandomForestClassifier(random_state=12345),
    'Logistic Regression': LogisticRegression(random_state=12345, max_iter=2000)
}

for name, model in models.items():
    model.fit(features_train, target_train)
    predictions = model.predict(features_valid)
    accuracy = accuracy_score(target_valid, predictions)
    print(f"{name}: {accuracy:.4f}")
```

    Decision Tree: 0.7123
    Random Forest: 0.7916
    Logistic Regression: 0.7263



```python
# Decision Tree tuning
best_accuracy = 0
best_depth = 0
for depth in range(1, 21):
    model = DecisionTreeClassifier(max_depth=depth, random_state=12345)
    model.fit(features_train, target_train)
    accuracy = accuracy_score(target_valid, model.predict(features_valid))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
    print(f"Depth {depth}: {accuracy:.4f}")

print(f"Best Decision Tree depth: {best_depth}, accuracy: {best_accuracy:.4f}")

```

    Depth 1: 0.7387
    Depth 2: 0.7574
    Depth 3: 0.7652
    Depth 4: 0.7636
    Depth 5: 0.7589
    Depth 6: 0.7574
    Depth 7: 0.7745
    Depth 8: 0.7667
    Depth 9: 0.7621
    Depth 10: 0.7714
    Depth 11: 0.7589
    Depth 12: 0.7558
    Depth 13: 0.7496
    Depth 14: 0.7574
    Depth 15: 0.7527
    Depth 16: 0.7496
    Depth 17: 0.7387
    Depth 18: 0.7418
    Depth 19: 0.7356
    Depth 20: 0.7294
    Best Decision Tree depth: 7, accuracy: 0.7745



```python
# Random Forest tuning
best_rf_accuracy = 0
best_params = {}
best_rf_model = None

for n_estimators in [10, 50, 100]:
    for max_depth in [5, 10, None]:
        rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=12345)
        rf_model.fit(features_train, target_train)
        accuracy = accuracy_score(target_valid, rf_model.predict(features_valid))
        if accuracy > best_rf_accuracy:
            best_rf_accuracy = accuracy
            best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
            best_rf_model = rf_model
        print(f"n_estimators={n_estimators}, max_depth={max_depth}: {accuracy:.3f}")

print(f"Best RF params: {best_params}, accuracy: {best_rf_accuracy:.4f}")

```

    n_estimators=10, max_depth=5: 0.778
    n_estimators=10, max_depth=10: 0.790
    n_estimators=10, max_depth=None: 0.788
    n_estimators=50, max_depth=5: 0.779
    n_estimators=50, max_depth=10: 0.798
    n_estimators=50, max_depth=None: 0.788
    n_estimators=100, max_depth=5: 0.779
    n_estimators=100, max_depth=10: 0.796
    n_estimators=100, max_depth=None: 0.792
    Best RF params: {'n_estimators': 50, 'max_depth': 10}, accuracy: 0.7978



```python
# Evaluate best RF on test set
your_predictions = best_rf_model.predict(features_test)
your_rf_accuracy = accuracy_score(target_test, your_predictions)

print("\nFinal Random Forest Evaluation:")
print(f"Test accuracy: {your_rf_accuracy:.4f}")

# Baseline
dummy_model = DummyClassifier(strategy='most_frequent', random_state=12345)
dummy_model.fit(features_train, target_train)
dummy_predictions = dummy_model.predict(features_test)
dummy_accuracy = accuracy_score(target_test, dummy_predictions)

print(f"Dummy classifier accuracy: {dummy_accuracy:.4f}")
print(f"Improvement over baseline: {your_rf_accuracy - dummy_accuracy:.4f}")

```

    
    Final Random Forest Evaluation:
    Test accuracy: 0.7994
    Dummy classifier accuracy: 0.6952
    Improvement over baseline: 0.1042



```python
# Class distribution check
print("\nActual class distribution in test set:")
print(target_test.value_counts(normalize=True))
print("\nPredicted class distribution:")
unique, counts = np.unique(your_predictions, return_counts=True)
for class_val, count in zip(unique, counts):
    print(f"Class {class_val}: {count/len(your_predictions):.3f}")

```

    
    Actual class distribution in test set:
    is_ultra
    0    0.695179
    1    0.304821
    Name: proportion, dtype: float64
    
    Predicted class distribution:
    Class 0: 0.787
    Class 1: 0.213



```python
# Sample predictions
comparison = pd.DataFrame({
    'calls': features_test['calls'].values[:10],
    'minutes': features_test['minutes'].values[:10], 
    'messages': features_test['messages'].values[:10],
    'mb_used': features_test['mb_used'].values[:10],
    'actual': target_test.values[:10],
    'predicted': your_predictions[:10]
})
print("\nSample predictions:")
print(comparison)
```

    
    Sample predictions:
       calls  minutes  messages   mb_used  actual  predicted
    0   82.0   507.89      88.0  17543.37       1          0
    1   50.0   375.91      35.0  12388.40       0          0
    2   83.0   540.49      41.0   9127.74       0          0
    3   79.0   562.99      19.0  25508.19       1          0
    4   78.0   531.29      20.0   9217.25       0          0
    5   53.0   478.18      78.0  20152.53       0          0
    6   73.0   582.47      33.0  12095.91       0          0
    7   31.0   172.10      25.0  31077.59       0          1
    8   28.0   222.21      30.0  22986.30       0          0
    9   68.0   523.56      14.0  18910.66       0          0

## Summary of Findings
The objective of this project was to build a classification model to predict whether a mobile user should be on the Smart or Ultra plan based on usage data. Multiple models were evaluated, including Decision Tree, Random Forest, and Logistic Regression.

The Random Forest classifier (50 estimators, max depth=10) achieved the best performance, with a validation accuracy of ~79.8% and a test accuracy of 79.94%. This represents a 10.4 percentage point improvement over the baseline dummy classifier, which achieved 69.52% by predicting the most frequent class.

Class distribution analysis revealed a dataset imbalance: 69.5% Smart users versus 30.5% Ultra users. The model overpredicted Smart (78.7%) and underpredicted Ultra (21.3%), indicating a bias toward the majority class. Several Ultra users with high data and minute usage were misclassified as Smart, suggesting feature overlap between plans in the mid-to-high usage range.

Overall, the Random Forest model met the target accuracy threshold but showed reduced sensitivity toward Ultra plan predictions. Future work should address class imbalance through reweighting or resampling, explore additional feature engineering, and consider advanced models such as gradient boosting to improve minority class detection.
