# 1. Project overview
Megaline, a mobile carrier, wants to recommend updated mobile plans — Smart or Ultra — to users still on legacy plans. To achieve this, we’ll build a classification model using user behavior data to predict the most suitable plan.

### The objective is to:

- Predict the correct plan (Smart or Ultra) based on usage data.
- Achieve at least 75% accuracy on the test set.
- Evaluate different models and tune hyperparameters.
- Validate the model with a sanity check to ensure logical performance.
- This project uses preprocessed data and focuses on model development and evaluation.

# 2.Loading Data
Load the data set from datasets/users_behavior.csv and import nescessary library 


```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```


```python
df = pd.read_csv('users_behavior.csv')
display(df)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>calls</th>
      <th>minutes</th>
      <th>messages</th>
      <th>mb_used</th>
      <th>is_ultra</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>40.0</td>
      <td>311.90</td>
      <td>83.0</td>
      <td>19915.42</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>85.0</td>
      <td>516.75</td>
      <td>56.0</td>
      <td>22696.96</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>77.0</td>
      <td>467.66</td>
      <td>86.0</td>
      <td>21060.45</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>106.0</td>
      <td>745.53</td>
      <td>81.0</td>
      <td>8437.39</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>66.0</td>
      <td>418.74</td>
      <td>1.0</td>
      <td>14502.75</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3209</th>
      <td>122.0</td>
      <td>910.98</td>
      <td>20.0</td>
      <td>35124.90</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3210</th>
      <td>25.0</td>
      <td>190.36</td>
      <td>0.0</td>
      <td>3275.61</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3211</th>
      <td>97.0</td>
      <td>634.44</td>
      <td>70.0</td>
      <td>13974.06</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3212</th>
      <td>64.0</td>
      <td>462.32</td>
      <td>90.0</td>
      <td>31239.78</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3213</th>
      <td>80.0</td>
      <td>566.09</td>
      <td>6.0</td>
      <td>29480.52</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>3214 rows × 5 columns</p>
</div>


# 3.Data Prepocessing
Apply cleaning, transformation, or checks 


```python
features = df.drop(['is_ultra'],axis = 1)
target = df['is_ultra']
```

# 4. Spliting Dataset
Split the dataset into training , validation, and test set. Justifice your split sizes

- Split your data into three sets:
  - Training set (60%): To train your models
  - Validation set (20%): To tune hyperparameters
  - Test set (20%): Final evaluation (don't touch until the end!)


```python
# Step 1: Split into train+validation (80%) and test (20%)
features_train_val, features_test, target_train_val, target_test = train_test_split(
    features, target, test_size=0.2, random_state=12345)

# Step 2: Split train+validation into train (60% of total) and validation (20% of total)
features_train, features_valid, target_train, target_valid = train_test_split(
    features_train_val, target_train_val, test_size=0.25, random_state=12345)

```

# 5. Model Selection and Training 
Train multiple classification models and tune hyperparameters


```python
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=12345),
    'Random Forest': RandomForestClassifier(random_state=12345),
    'Logistic Regression': LogisticRegression(random_state=12345)}

# Train and evaluate each model
for name, model in models.items():
    model.fit(features_train, target_train)
    predictions = model.predict(features_valid)
    accuracy = accuracy_score(target_valid, predictions)
    print(f"{name}: {accuracy:.4f}")
```

    Decision Tree: 0.7123
    Random Forest: 0.7916
    Logistic Regression: 0.7263


    /opt/anaconda3/lib/python3.12/site-packages/sklearn/linear_model/_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(



```python
best_accuracy = 0
best_depth = 0

for depth in range(1, 21):  # Try different max_depth values
    model = DecisionTreeClassifier(max_depth=depth, random_state=12345)
    model.fit(features_train, target_train)
    predictions = model.predict(features_valid)
    accuracy = accuracy_score(target_valid, predictions)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth
    
    print(f"Depth {depth}: {accuracy:.4f}")

print(f"\nBest depth: {best_depth}, Best accuracy: {best_accuracy:.4f}")
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
    
    Best depth: 7, Best accuracy: 0.7745



```python
# Random Forest hyperparameter tuning
best_rf_accuracy = 0
best_params = {}

for n_estimators in [10, 50, 100]:
    for max_depth in [5, 10, None]:
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=12345
        )
        model.fit(features_train, target_train)
        predictions = model.predict(features_valid)
        accuracy = accuracy_score(target_valid, predictions)
        
        if accuracy > best_rf_accuracy:
            best_rf_accuracy = accuracy
            best_params = {'n_estimators': n_estimators, 'max_depth': max_depth}
        
        print(f"n_estimators={n_estimators}, max_depth={max_depth}: {accuracy:.3f}")

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


# 6. Model evaluation on Validation Set
Compare model performance on the validation set and select the best one.


```python
model.fit(features, target)

train_predictions = model.predict(features)
test_predictions = model.predict(features_test)

print('Accuracy')
print('Training set:', accuracy_score(target, train_predictions))
print('Test set:', accuracy_score(target_test, test_predictions))
```

    Accuracy
    Training set: 1.0
    Test set: 1.0



```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# Create a dummy classifier that always predicts the most frequent class
dummy_model = DummyClassifier(strategy='most_frequent', random_state=12345)
dummy_model.fit(features_train, target_train)
dummy_predictions = dummy_model.predict(features_test)
dummy_accuracy = accuracy_score(target_test, dummy_predictions)

print(f"Dummy classifier accuracy: {dummy_accuracy:.4f}")
print(f"Your Random Forest accuracy: {your_rf_accuracy:.4f}")
print(f"Improvement over baseline: {your_rf_accuracy - dummy_accuracy:.4f}")
```

    Dummy classifier accuracy: 0.6952



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[9], line 11
          8 dummy_accuracy = accuracy_score(target_test, dummy_predictions)
         10 print(f"Dummy classifier accuracy: {dummy_accuracy:.4f}")
    ---> 11 print(f"Your Random Forest accuracy: {your_rf_accuracy:.4f}")
         12 print(f"Improvement over baseline: {your_rf_accuracy - dummy_accuracy:.4f}")


    NameError: name 'your_rf_accuracy' is not defined



```python
# Check if your model is just predicting one class
import numpy as np

print("Actual class distribution in test set:")
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



    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[10], line 8
          5 print(target_test.value_counts(normalize=True))
          7 print("\nPredicted class distribution:")
    ----> 8 unique, counts = np.unique(your_predictions, return_counts=True)
          9 for class_val, count in zip(unique, counts):
         10     print(f"Class {class_val}: {count/len(your_predictions):.3f}")


    NameError: name 'your_predictions' is not defined



```python
# Look at some predictions and see if they make sense
import pandas as pd

# Create a comparison dataframe
comparison = pd.DataFrame({
    'calls': features_test['calls'].values[:10],
    'minutes': features_test['minutes'].values[:10], 
    'messages': features_test['messages'].values[:10],
    'mb_used': features_test['mb_used'].values[:10],
    'actual': target_test.values[:10],
    'predicted': your_predictions[:10]
})

print("Sample predictions:")
print(comparison)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[11], line 11
          2 import pandas as pd
          4 # Create a comparison dataframe
          5 comparison = pd.DataFrame({
          6     'calls': features_test['calls'].values[:10],
          7     'minutes': features_test['minutes'].values[:10], 
          8     'messages': features_test['messages'].values[:10],
          9     'mb_used': features_test['mb_used'].values[:10],
         10     'actual': target_test.values[:10],
    ---> 11     'predicted': your_predictions[:10]
         12 })
         14 print("Sample predictions:")
         15 print(comparison)


    NameError: name 'your_predictions' is not defined


# 7.Final Model Testing 
Evaluate the chosen model on the test set and report the final accuracy.


```python

```

# 8. Sanity Check 
Perform a sanity check to ensure the model predictions make logical sense.


```python

```

# 9.Summary of Finding 
Discuss model performance and insights


```python

```
