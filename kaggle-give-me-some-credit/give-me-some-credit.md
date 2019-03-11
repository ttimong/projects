

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
pd.set_option("display.max_columns", 100)
```


```python
train = pd.read_csv('./cs-training.csv', index_col=0)
```


```python
test = pd.read_csv('./cs-test.csv', index_col=0)
```


```python
test.head(2)
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
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>0.885519</td>
      <td>43</td>
      <td>0</td>
      <td>0.177513</td>
      <td>5700.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>0.463295</td>
      <td>57</td>
      <td>0</td>
      <td>0.527237</td>
      <td>9141.0</td>
      <td>15</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.head(2)
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
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.766127</td>
      <td>45</td>
      <td>2</td>
      <td>0.802982</td>
      <td>9120.0</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0.957151</td>
      <td>40</td>
      <td>0</td>
      <td>0.121876</td>
      <td>2600.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.shape
```




    (101503, 11)




```python
train.shape
```




    (150000, 11)




```python
combine = pd.concat([train, test], axis=0)
```


```python
combine.head(2)
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
      <th>SeriousDlqin2yrs</th>
      <th>RevolvingUtilizationOfUnsecuredLines</th>
      <th>age</th>
      <th>NumberOfTime30-59DaysPastDueNotWorse</th>
      <th>DebtRatio</th>
      <th>MonthlyIncome</th>
      <th>NumberOfOpenCreditLinesAndLoans</th>
      <th>NumberOfTimes90DaysLate</th>
      <th>NumberRealEstateLoansOrLines</th>
      <th>NumberOfTime60-89DaysPastDueNotWorse</th>
      <th>NumberOfDependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.766127</td>
      <td>45</td>
      <td>2</td>
      <td>0.802982</td>
      <td>9120.0</td>
      <td>13</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.957151</td>
      <td>40</td>
      <td>0</td>
      <td>0.121876</td>
      <td>2600.0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine.columns = ['target', 'unsecuredline_utilization', 'age', 'count_of_30-59_days_past_due_not_worse', 'debt_ratio', 'monthly_income', 'count_open_creditlines_and_loans', 'count_90_days_late',
                'count_real_estate_loans_or_lines', 'count_of_60-89_days_past_due_not_worse', 'count_dependents']
```


```python
combine.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 251503 entries, 1 to 101503
    Data columns (total 11 columns):
    target                                    150000 non-null float64
    unsecuredline_utilization                 251503 non-null float64
    age                                       251503 non-null int64
    count_of_30-59_days_past_due_not_worse    251503 non-null int64
    debt_ratio                                251503 non-null float64
    monthly_income                            201669 non-null float64
    count_open_creditlines_and_loans          251503 non-null int64
    count_90_days_late                        251503 non-null int64
    count_real_estate_loans_or_lines          251503 non-null int64
    count_of_60-89_days_past_due_not_worse    251503 non-null int64
    count_dependents                          244953 non-null float64
    dtypes: float64(5), int64(6)
    memory usage: 23.0 MB
    


```python
target_dist = (
    train
    .groupby("target")
    .agg({"age": "count"})
    .reset_index()
)
```


```python
target_dist
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
      <th>target</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>139974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10026</td>
    </tr>
  </tbody>
</table>
</div>




```python
missing_dpt = combine[combine.count_dependents.isnull()]
missing_dpt_target_dist = (
    missing_dpt
    .groupby("target")
    .agg({"age": "count"})
    .reset_index()
)
```


```python
missing_dpt_target_dist
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
      <th>target</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>3745</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>179</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine.count_dependents.describe()
```




    count    244953.000000
    mean          0.761995
    std           1.123905
    min           0.000000
    25%           0.000000
    50%           0.000000
    75%           1.000000
    max          43.000000
    Name: count_dependents, dtype: float64




```python
# impute null values in count_dependents column with median value, i.e. 0
combine.count_dependents.fillna(0, inplace=True)
```


```python
missing_income = combine[combine.monthly_income.isnull()]
missing_income_target_dist = (
    missing_income
    .groupby("target")
    .agg({"age": "count"})
    .reset_index()
)
missing_income_target_dist
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
      <th>target</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>28062</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1669</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine.age.sort_values().unique()
```




    array([  0,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
            33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
            46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
            59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,
            72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,
            85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
            98,  99, 100, 101, 102, 103, 104, 105, 107, 109], dtype=int64)




```python
combine.query("monthly_income == 0").debt_ratio.describe()
```




    count      2654.000000
    mean       1713.872645
    std        5386.139965
    min           0.000000
    25%          93.000000
    50%         858.500000
    75%        2175.250000
    max      202990.000000
    Name: debt_ratio, dtype: float64




```python
combine.query("monthly_income > 0").debt_ratio.describe()
```




    count    199015.000000
    mean          5.009825
    std         164.243734
    min           0.000000
    25%           0.141114
    50%           0.292142
    75%           0.472765
    max       61106.500000
    Name: debt_ratio, dtype: float64




```python
combine.query("monthly_income > 0").debt_ratio.quantile(q=0.99)
```




    2.9688210751599997




```python
combine.monthly_income.fillna(-1, inplace=True)
```


```python
combine.loc[combine.query("monthly_income == -1 and debt_ratio > 3").index.values, 'monthly_income'] = 0
```


```python
len(combine.query("monthly_income == -1"))
```




    3080




```python
combine.query("monthly_income == -1 and (target == 1 or target == 0 )").age.unique()
```




    array([ 62,  63,  28,  29,  83,  97,  91,  82,  86,  34,  90,  70,  84,
            57,  89,  69,  68,  92,  48,  30,  81,  60,  25,  22,  41,  85,
            24,  55,  61,  46,  56,  31,  40,  75,  66,  42,  59,  79,  54,
            50,  88,  73,  38,  80,  76,  71,  78,  44,  74,  52,  51,  64,
            77,  93,  37,  65,  39,  23,  45,  21,  43,  36,  67,  35,  26,
            27,  98,  87,  49,  94,  72,  58,  53,  95,  33,  99,  32,  47,
           105,  96, 102], dtype=int64)


