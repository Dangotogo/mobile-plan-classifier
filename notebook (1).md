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

df = pd.read_csv('users_behavior.csv')
df.info()
display(df)
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3214 entries, 0 to 3213
    Data columns (total 5 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   calls     3214 non-null   float64
     1   minutes   3214 non-null   float64
     2   messages  3214 non-null   float64
     3   mb_used   3214 non-null   float64
     4   is_ultra  3214 non-null   int64  
    dtypes: float64(4), int64(1)
    memory usage: 125.7 KB



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

```

# 4. Spliting Dataset
Split the dataset into training , validation, and test set. Justifice your split sizes


```python

```

# 5. Model Selection and Training 
Train multiple classification models and tune hyperparameters


```python

```

# 6. Model evaluation on Validation Set
Compare model performance on the validation set and select the best one.


```python

```

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
