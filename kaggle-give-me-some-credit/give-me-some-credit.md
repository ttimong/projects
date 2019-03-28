

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
combine.columns = ['target', 'unsecured_line_utilization', 'age', 'count_of_30-59_days_past_due_not_worse', 'debt_ratio', 'monthly_income', 'count_open_creditlines_and_loans', 'count_90_days_late',
                'count_real_estate_loans', 'count_of_60-89_days_past_due_not_worse', 'count_dependents']
```


```python
combine.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 251503 entries, 1 to 101503
    Data columns (total 11 columns):
    target                                    150000 non-null float64
    unsecured_line_utilization                251503 non-null float64
    age                                       251503 non-null int64
    count_of_30-59_days_past_due_not_worse    251503 non-null int64
    debt_ratio                                251503 non-null float64
    monthly_income                            201669 non-null float64
    count_open_creditlines_and_loans          251503 non-null int64
    count_90_days_late                        251503 non-null int64
    count_real_estate_loans                   251503 non-null int64
    count_of_60-89_days_past_due_not_worse    251503 non-null int64
    count_dependents                          244953 non-null float64
    dtypes: float64(5), int64(6)
    memory usage: 23.0 MB
    


```python
combine.describe()
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
      <th>unsecured_line_utilization</th>
      <th>age</th>
      <th>count_of_30-59_days_past_due_not_worse</th>
      <th>debt_ratio</th>
      <th>monthly_income</th>
      <th>count_open_creditlines_and_loans</th>
      <th>count_90_days_late</th>
      <th>count_real_estate_loans</th>
      <th>count_of_60-89_days_past_due_not_worse</th>
      <th>count_dependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150000.000000</td>
      <td>251503.000000</td>
      <td>251503.000000</td>
      <td>251503.000000</td>
      <td>251503.000000</td>
      <td>2.016690e+05</td>
      <td>251503.000000</td>
      <td>251503.000000</td>
      <td>251503.000000</td>
      <td>251503.000000</td>
      <td>244953.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.066840</td>
      <td>5.750415</td>
      <td>52.339694</td>
      <td>0.434245</td>
      <td>349.562468</td>
      <td>6.744818e+03</td>
      <td>8.453064</td>
      <td>0.278370</td>
      <td>1.016155</td>
      <td>0.252466</td>
      <td>0.761995</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.249746</td>
      <td>229.633980</td>
      <td>14.775120</td>
      <td>4.335643</td>
      <td>1884.792016</td>
      <td>2.571761e+04</td>
      <td>5.145194</td>
      <td>4.312539</td>
      <td>1.121935</td>
      <td>4.299204</td>
      <td>1.123905</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.029977</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.174330</td>
      <td>3.400000e+03</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.153575</td>
      <td>52.000000</td>
      <td>0.000000</td>
      <td>0.365612</td>
      <td>5.400000e+03</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.561293</td>
      <td>63.000000</td>
      <td>0.000000</td>
      <td>0.861754</td>
      <td>8.212000e+03</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>50708.000000</td>
      <td>109.000000</td>
      <td>98.000000</td>
      <td>329664.000000</td>
      <td>7.727000e+06</td>
      <td>85.000000</td>
      <td>98.000000</td>
      <td>54.000000</td>
      <td>98.000000</td>
      <td>43.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine.unsecured_line_utilization.quantile(q=0.99)
```




    1.0912907162




```python
# highly likely that unsecured_line_utilization to be more than 1.1
# assume that any value above 1.1 is erroneous 
print("count of erroneous unsecured_line_utilization: " + str(len(combine.query("unsecured_line_utilization > 1.1"))))
```

    count of erroneous unsecured_line_utilization: 2406
    

erroneous unsecuredline_utilization is about 1% of the entire data. I will replace these erroneous values with the median value of unsecured_line_utilization


```python
# replacing errorneous values for unsecured_line_utilization
combine.loc[combine.query("unsecured_line_utilization >= 1.1").index.values, 'unsecured_line_utilization'] = combine.query("unsecured_line_utilization < 1.1").unsecured_line_utilization.median()
```


```python
# replacing age = 0 with the mean age of data set
combine.loc[combine.query("age == 0").index.values, 'age'] = combine.query("age != 0").age.median()
```


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
combine['age_group'] = ['21- 40' if (i>=21 and i<41) else '41-60' if (i>=41 and i<61) else '61-80' if (i>=61 and i<81) else '>81' for i in combine.age]
```


```python
combine.
```


```python
_ = sns.boxplot(x='age_group', y='monthly_income', data=combine)
```


![png](give-me-some-credit_files/give-me-some-credit_19_0.png)



```python
_ = sns.boxplot(x='age_group', y='monthly_income', data=combine.query("monthly_income < 200000"))
```


![png](give-me-some-credit_files/give-me-some-credit_20_0.png)



```python
combine.query("monthly_income > 0").monthly_income.quantile(q=0.99)
```




    25325.57999999926




```python
combine = (
    combine
    .pipe(lambda x: x.assign(monthly_income=np.where(combine[combine.monthly_income > 25000]
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
      <th>unsecured_line_utilization</th>
      <th>age</th>
      <th>count_of_30-59_days_past_due_not_worse</th>
      <th>debt_ratio</th>
      <th>monthly_income</th>
      <th>count_open_creditlines_and_loans</th>
      <th>count_90_days_late</th>
      <th>count_real_estate_loans</th>
      <th>count_of_60-89_days_past_due_not_worse</th>
      <th>count_dependents</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.907239</td>
      <td>49.0</td>
      <td>1</td>
      <td>0.024926</td>
      <td>63588.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>267</th>
      <td>0.0</td>
      <td>0.007983</td>
      <td>43.0</td>
      <td>1</td>
      <td>0.009691</td>
      <td>208333.0</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>275</th>
      <td>0.0</td>
      <td>0.292645</td>
      <td>57.0</td>
      <td>0</td>
      <td>0.205928</td>
      <td>26484.0</td>
      <td>11</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>412</th>
      <td>0.0</td>
      <td>0.265705</td>
      <td>56.0</td>
      <td>0</td>
      <td>0.259103</td>
      <td>31666.0</td>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>562</th>
      <td>0.0</td>
      <td>0.162077</td>
      <td>38.0</td>
      <td>0</td>
      <td>0.024885</td>
      <td>70000.0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>604</th>
      <td>0.0</td>
      <td>0.019154</td>
      <td>49.0</td>
      <td>0</td>
      <td>0.201026</td>
      <td>33333.0</td>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>4.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>0.0</td>
      <td>0.369172</td>
      <td>82.0</td>
      <td>0</td>
      <td>0.048229</td>
      <td>26000.0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>&gt;81</td>
    </tr>
    <tr>
      <th>1379</th>
      <td>0.0</td>
      <td>0.081354</td>
      <td>59.0</td>
      <td>0</td>
      <td>0.028554</td>
      <td>60200.0</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>1538</th>
      <td>0.0</td>
      <td>0.510293</td>
      <td>59.0</td>
      <td>0</td>
      <td>0.316978</td>
      <td>61000.0</td>
      <td>13</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>1762</th>
      <td>0.0</td>
      <td>0.000172</td>
      <td>71.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>30000.0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>1772</th>
      <td>0.0</td>
      <td>0.705197</td>
      <td>56.0</td>
      <td>0</td>
      <td>0.262722</td>
      <td>55000.0</td>
      <td>16</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>2090</th>
      <td>0.0</td>
      <td>0.202233</td>
      <td>70.0</td>
      <td>2</td>
      <td>0.657355</td>
      <td>41666.0</td>
      <td>22</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>2283</th>
      <td>0.0</td>
      <td>0.001874</td>
      <td>44.0</td>
      <td>0</td>
      <td>0.022127</td>
      <td>55000.0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>2351</th>
      <td>0.0</td>
      <td>0.156709</td>
      <td>60.0</td>
      <td>0</td>
      <td>0.047499</td>
      <td>32000.0</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>2538</th>
      <td>0.0</td>
      <td>0.663486</td>
      <td>37.0</td>
      <td>0</td>
      <td>0.023033</td>
      <td>37250.0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>2550</th>
      <td>0.0</td>
      <td>0.423640</td>
      <td>39.0</td>
      <td>0</td>
      <td>0.157977</td>
      <td>33333.0</td>
      <td>10</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>2555</th>
      <td>0.0</td>
      <td>0.320600</td>
      <td>56.0</td>
      <td>0</td>
      <td>0.260669</td>
      <td>28000.0</td>
      <td>26</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>3153</th>
      <td>0.0</td>
      <td>0.113344</td>
      <td>66.0</td>
      <td>1</td>
      <td>0.081545</td>
      <td>58249.0</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>3191</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>72.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>50000.0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>3360</th>
      <td>0.0</td>
      <td>0.089873</td>
      <td>62.0</td>
      <td>0</td>
      <td>0.227385</td>
      <td>28000.0</td>
      <td>14</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>3428</th>
      <td>0.0</td>
      <td>0.404801</td>
      <td>56.0</td>
      <td>0</td>
      <td>0.038005</td>
      <td>110775.0</td>
      <td>12</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>3431</th>
      <td>0.0</td>
      <td>0.432811</td>
      <td>55.0</td>
      <td>1</td>
      <td>0.274929</td>
      <td>60000.0</td>
      <td>11</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>3492</th>
      <td>0.0</td>
      <td>0.190440</td>
      <td>53.0</td>
      <td>1</td>
      <td>0.070462</td>
      <td>67000.0</td>
      <td>11</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>3711</th>
      <td>1.0</td>
      <td>0.845686</td>
      <td>52.0</td>
      <td>1</td>
      <td>0.257959</td>
      <td>25600.0</td>
      <td>9</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>4044</th>
      <td>0.0</td>
      <td>0.007390</td>
      <td>94.0</td>
      <td>0</td>
      <td>0.000039</td>
      <td>203500.0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>&gt;81</td>
    </tr>
    <tr>
      <th>4302</th>
      <td>0.0</td>
      <td>0.507791</td>
      <td>44.0</td>
      <td>2</td>
      <td>0.149314</td>
      <td>60000.0</td>
      <td>9</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>4374</th>
      <td>0.0</td>
      <td>0.118937</td>
      <td>37.0</td>
      <td>0</td>
      <td>0.038279</td>
      <td>36416.0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>4390</th>
      <td>0.0</td>
      <td>0.174713</td>
      <td>44.0</td>
      <td>0</td>
      <td>0.160347</td>
      <td>26660.0</td>
      <td>12</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>4406</th>
      <td>0.0</td>
      <td>0.030375</td>
      <td>53.0</td>
      <td>0</td>
      <td>0.201707</td>
      <td>28000.0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>4587</th>
      <td>0.0</td>
      <td>0.265371</td>
      <td>67.0</td>
      <td>1</td>
      <td>0.038434</td>
      <td>60700.0</td>
      <td>21</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>98337</th>
      <td>NaN</td>
      <td>0.155251</td>
      <td>39.0</td>
      <td>0</td>
      <td>0.200187</td>
      <td>31000.0</td>
      <td>11</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>98361</th>
      <td>NaN</td>
      <td>0.149271</td>
      <td>69.0</td>
      <td>0</td>
      <td>0.026247</td>
      <td>128699.0</td>
      <td>11</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>1.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>98410</th>
      <td>NaN</td>
      <td>0.008773</td>
      <td>57.0</td>
      <td>0</td>
      <td>0.000333</td>
      <td>30000.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>98515</th>
      <td>NaN</td>
      <td>0.000000</td>
      <td>63.0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>50000.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>98617</th>
      <td>NaN</td>
      <td>0.635255</td>
      <td>33.0</td>
      <td>3</td>
      <td>0.181676</td>
      <td>33333.0</td>
      <td>8</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>98641</th>
      <td>NaN</td>
      <td>0.121488</td>
      <td>50.0</td>
      <td>0</td>
      <td>0.321673</td>
      <td>26700.0</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>98774</th>
      <td>NaN</td>
      <td>0.066603</td>
      <td>49.0</td>
      <td>0</td>
      <td>0.908486</td>
      <td>26771.0</td>
      <td>19</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>99234</th>
      <td>NaN</td>
      <td>0.004156</td>
      <td>69.0</td>
      <td>0</td>
      <td>0.063198</td>
      <td>35000.0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>99348</th>
      <td>NaN</td>
      <td>0.026034</td>
      <td>54.0</td>
      <td>0</td>
      <td>0.002818</td>
      <td>33000.0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>99451</th>
      <td>NaN</td>
      <td>0.060062</td>
      <td>39.0</td>
      <td>0</td>
      <td>0.008918</td>
      <td>73333.0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>99460</th>
      <td>NaN</td>
      <td>0.673957</td>
      <td>38.0</td>
      <td>0</td>
      <td>1.573905</td>
      <td>27000.0</td>
      <td>14</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>99527</th>
      <td>NaN</td>
      <td>0.689402</td>
      <td>54.0</td>
      <td>0</td>
      <td>0.047766</td>
      <td>35924.0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>99622</th>
      <td>NaN</td>
      <td>0.149271</td>
      <td>64.0</td>
      <td>0</td>
      <td>0.322834</td>
      <td>32000.0</td>
      <td>12</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>99626</th>
      <td>NaN</td>
      <td>0.016578</td>
      <td>65.0</td>
      <td>0</td>
      <td>0.306052</td>
      <td>32500.0</td>
      <td>19</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>99722</th>
      <td>NaN</td>
      <td>0.745913</td>
      <td>45.0</td>
      <td>0</td>
      <td>0.037766</td>
      <td>41200.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>99882</th>
      <td>NaN</td>
      <td>0.017367</td>
      <td>50.0</td>
      <td>0</td>
      <td>0.221759</td>
      <td>30000.0</td>
      <td>16</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>99963</th>
      <td>NaN</td>
      <td>0.116618</td>
      <td>46.0</td>
      <td>2</td>
      <td>0.152472</td>
      <td>29611.0</td>
      <td>13</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>100327</th>
      <td>NaN</td>
      <td>0.124346</td>
      <td>54.0</td>
      <td>0</td>
      <td>0.351797</td>
      <td>29300.0</td>
      <td>12</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>2.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>100443</th>
      <td>NaN</td>
      <td>0.146878</td>
      <td>33.0</td>
      <td>0</td>
      <td>0.685683</td>
      <td>40000.0</td>
      <td>18</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>100488</th>
      <td>NaN</td>
      <td>0.012783</td>
      <td>58.0</td>
      <td>0</td>
      <td>0.000760</td>
      <td>50000.0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>100521</th>
      <td>NaN</td>
      <td>0.450754</td>
      <td>72.0</td>
      <td>0</td>
      <td>0.177540</td>
      <td>33000.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>100576</th>
      <td>NaN</td>
      <td>0.186636</td>
      <td>29.0</td>
      <td>1</td>
      <td>0.038410</td>
      <td>55427.0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>21- 40</td>
    </tr>
    <tr>
      <th>100962</th>
      <td>NaN</td>
      <td>0.128575</td>
      <td>44.0</td>
      <td>2</td>
      <td>0.142225</td>
      <td>35485.0</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>101077</th>
      <td>NaN</td>
      <td>0.103166</td>
      <td>41.0</td>
      <td>1</td>
      <td>0.294460</td>
      <td>26950.0</td>
      <td>9</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>101134</th>
      <td>NaN</td>
      <td>0.105178</td>
      <td>58.0</td>
      <td>0</td>
      <td>0.002940</td>
      <td>50000.0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>101262</th>
      <td>NaN</td>
      <td>0.165953</td>
      <td>66.0</td>
      <td>0</td>
      <td>0.070271</td>
      <td>29300.0</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>101331</th>
      <td>NaN</td>
      <td>0.066047</td>
      <td>56.0</td>
      <td>0</td>
      <td>0.043614</td>
      <td>26000.0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>101344</th>
      <td>NaN</td>
      <td>0.554144</td>
      <td>56.0</td>
      <td>3</td>
      <td>0.181828</td>
      <td>36000.0</td>
      <td>15</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>101413</th>
      <td>NaN</td>
      <td>0.010570</td>
      <td>52.0</td>
      <td>0</td>
      <td>0.001569</td>
      <td>43333.0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>101435</th>
      <td>NaN</td>
      <td>0.932771</td>
      <td>43.0</td>
      <td>1</td>
      <td>0.004089</td>
      <td>67749.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>41-60</td>
    </tr>
  </tbody>
</table>
<p>2021 rows × 12 columns</p>
</div>




```python
len(combine.query("monthly_income >25000"))
```




    2021




```python
len(combine)
```




    251503




```python

```


![png](give-me-some-credit_files/give-me-some-credit_25_0.png)



```python

```




    83005.32800001078




```python
combine.query("monthly_income == 83000")
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
      <th>unsecured_line_utilization</th>
      <th>age</th>
      <th>count_of_30-59_days_past_due_not_worse</th>
      <th>debt_ratio</th>
      <th>monthly_income</th>
      <th>count_open_creditlines_and_loans</th>
      <th>count_90_days_late</th>
      <th>count_real_estate_loans</th>
      <th>count_of_60-89_days_past_due_not_worse</th>
      <th>count_dependents</th>
      <th>age_group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>68281</th>
      <td>0.0</td>
      <td>1.000000</td>
      <td>45.0</td>
      <td>0</td>
      <td>0.136818</td>
      <td>83000.0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>129390</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>62.0</td>
      <td>0</td>
      <td>0.064855</td>
      <td>83000.0</td>
      <td>10</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0.0</td>
      <td>61-80</td>
    </tr>
    <tr>
      <th>48601</th>
      <td>NaN</td>
      <td>0.157289</td>
      <td>54.0</td>
      <td>0</td>
      <td>0.008217</td>
      <td>83000.0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3.0</td>
      <td>41-60</td>
    </tr>
    <tr>
      <th>80397</th>
      <td>NaN</td>
      <td>0.762257</td>
      <td>41.0</td>
      <td>0</td>
      <td>0.059686</td>
      <td>83000.0</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>4.0</td>
      <td>41-60</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = sns.distplot(combine[combine.monthly_income.notnull()].query("monthly_income< 25000").monthly_income)
_ = plt.xticks(rotation=75)
```

    C:\Users\timothy.ong\AppData\Local\Continuum\anaconda3\lib\site-packages\scipy\stats\stats.py:1706: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](give-me-some-credit_files/give-me-some-credit_28_1.png)



```python
len(combine[combine.monthly_income.notnull()].query("monthly_income< 200000"))/len(combine[combine.monthly_income.notnull()])
```




    0.9997669448452663




```python
combine[combine.monthly_income.notnull()].query("monthly_income > 25000").groupby("target").agg({"age":"count"}).reset_index()
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
      <td>1106</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>62</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = sns.distplot(combine.query("monthly_income >= 0 and monthly_income <={}".format(combine.query("monthly_income != 0").monthly_income.quantile(q=0.99))).monthly_income)
```

    C:\Users\timothy.ong\AppData\Local\Continuum\anaconda3\lib\site-packages\scipy\stats\stats.py:1706: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
      return np.add.reduce(sorted[indexer] * weights, axis=axis) / sumval
    


![png](give-me-some-credit_files/give-me-some-credit_31_1.png)



```python
len(combine.query("monthly_income == 0"))
```




    2654




```python
combine.monthly_income.quantile(q=0.99)
```




    25050.360000000335




```python
combine.query("monthly_income == 0").groupby("target").agg({"age": "count"}).reset_index()
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
      <td>1568</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>66</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(combine.query("monthly_income == 0 and debt_ratio <3"))
```




    200




```python
combine.query("monthly_income > 0").debt_ratio.quantile(q=0.99)
```




    2.9688210751599997




```python
len(combine[combine.monthly_income.isnull()].query("debt_ratio>3"))
```




    46215



We can see that at 99 percentile, debt ratio is less than 3 when monthly income is more than 0


```python
# filling null values in monthly_income column with -1
combine.monthly_income.fillna(-1, inplace=True)
```


```python
# from the above observations, it is very likely that a person's monthly_income would be zero if his/her debt_ratio is above 3
combine.loc[combine.query("monthly_income == -1 and debt_ratio > 3").index.values, 'monthly_income'] = 0
```


```python
# how many null values of monthly_income left
len(combine.query("monthly_income == -1"))
```




    3080




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
combine.query("age == 0")
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
      <th>unsecured_line_utilization</th>
      <th>age</th>
      <th>count_of_30-59_days_past_due_not_worse</th>
      <th>debt_ratio</th>
      <th>monthly_income</th>
      <th>count_open_creditlines_and_loans</th>
      <th>count_90_days_late</th>
      <th>count_real_estate_loans</th>
      <th>count_of_60-89_days_past_due_not_worse</th>
      <th>count_dependents</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65696</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>1</td>
      <td>0.436927</td>
      <td>6000.0</td>
      <td>6</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# creating new column age_group so that we can better ascertain the missing monthly_income based on age
combine['age_group'] = ['21- 30' if (i>=21 and i<31) else '31-40' if (i>=31 and i<41) else '41-50' if (i>=41 and i<50) else '51-60' if (i>=51 and i<60) else '61-70' if (i>=61 and i<70)\
                       else '71-80' if (i>=71 and i<80) else '81-90' if (i>=81 and i<90) else '91-100' if (i>=91 and i<100) else '>101' for i in combine.age]
```


```python
# filling missing monthly_income values with the median income for that specific age_group
combine.loc[combine.query("monthly_income == -1 and age_group == '31-40'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '31-40'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '41-50'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '41-50'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '51-60'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '51-60'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '61-70'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '61-70'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '71-80'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '71-80'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '81-90'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '81-90'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '91-100'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '91-100'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '>101'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '>101'").monthly_income.median()
```


```python
combine.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 251503 entries, 1 to 101503
    Data columns (total 12 columns):
    target                                    150000 non-null float64
    unsecured_line_utilization                251503 non-null float64
    age                                       251503 non-null float64
    count_of_30-59_days_past_due_not_worse    251503 non-null int64
    debt_ratio                                251503 non-null float64
    monthly_income                            251503 non-null float64
    count_open_creditlines_and_loans          251503 non-null int64
    count_90_days_late                        251503 non-null int64
    count_real_estate_loans                   251503 non-null int64
    count_of_60-89_days_past_due_not_worse    251503 non-null int64
    count_dependents                          251503 non-null float64
    age_group                                 251503 non-null object
    dtypes: float64(6), int64(5), object(1)
    memory usage: 29.9+ MB
    
