

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
%matplotlib inline
```


```python
transactions = pd.read_csv('./user_transactions.csv')
user_base = pd.read_csv('./user_base_part1_edit.csv')
user_label = pd.read_csv('./user_label_branch_edit.csv')
```

# Data Cleaning


```python
# divide monetary value by 10000 so that it is easier to read
combine = (
    transactions
    .merge(user_base, on='user_id', how='left')
    .merge(user_label, on='user_id', how='left')
    [['user_id', 'branch_code', 'default_flag', 'number_of_cards', 'outstanding', 'credit_limit', 'bill', 'total_cash_usage',
       'total_retail_usage', 'total_usage', 'total_usage_per_limit', 'total_3mo_usage_per_limit',
       'total_6mo_usage_per_limit', 'remaining_bill', 'payment_ratio', 'payment_ratio_3month', 'payment_ratio_6month','overlimit_percentage',
       'delinquency_score', 'years_since_card_issuing',
       'remaining_bill_per_number_of_cards', 'remaining_bill_per_limit', 'utilization_3month', 'utilization_6month']]
    .pipe(lambda x: x.assign(outstanding=x.outstanding/10000))
    .pipe(lambda x: x.assign(credit_limit=x.credit_limit/10000))
    .pipe(lambda x: x.assign(bill=x.bill/10000))
    .pipe(lambda x: x.assign(total_cash_usage=x.total_cash_usage/10000))
    .pipe(lambda x: x.assign(total_retail_usage=x.total_retail_usage/10000))
    .pipe(lambda x: x.assign(total_usage=x.total_usage/10000))
    .pipe(lambda x: x.assign(remaining_bill=x.remaining_bill/10000))
    .pipe(lambda x: x.assign(remaining_bill_per_number_of_cards=x.remaining_bill_per_number_of_cards/10000))
)
```


```python
combine[['total_cash_usage', 'total_retail_usage', 'total_usage']].describe()
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
      <th>total_cash_usage</th>
      <th>total_retail_usage</th>
      <th>total_usage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15600.000000</td>
      <td>15645.000000</td>
      <td>15645.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.457618</td>
      <td>202.480595</td>
      <td>209.928120</td>
    </tr>
    <tr>
      <th>std</th>
      <td>61.778729</td>
      <td>790.934462</td>
      <td>807.325021</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>-1566.720000</td>
      <td>-1566.720000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>20.000000</td>
      <td>24.380000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>139.620400</td>
      <td>150.747800</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2884.057200</td>
      <td>28500.000000</td>
      <td>31400.000000</td>
    </tr>
  </tbody>
</table>
</div>



Since there are 45 null values in the total_cash_usage, I would need to find a way to fill in those missing values. 

Logically, it seems that total_cash_usage + total_retail_usage = total_usage. If this were true, I would be able to use total_retail_usage and total_usage to calculate total_cash_usage.


```python
# assuming total_cash_usage + total_retail_usage = total_usage
# check out assumption
check_assumption1 = (
    combine
    [['total_cash_usage', 'total_retail_usage', 'total_usage']]
    .fillna('x')
    .query("total_cash_usage != 'x'")
    .pipe(lambda x: x.assign(total_usage_check=x.total_cash_usage+x.total_retail_usage-x.total_usage))
)
```


```python
print("% of rows that does not tally: " + str(round(len(check_assumption1.query("total_usage_check !=0"))/check_assumption1.shape[0] * 100, 1)) + '%')
```

    % of rows that does not tally: 0.3%



```python
check_assumption1.query("total_usage_check !=0")
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
      <th>total_cash_usage</th>
      <th>total_retail_usage</th>
      <th>total_usage</th>
      <th>total_usage_check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>70</td>
      <td>110.6768</td>
      <td>180.6768</td>
      <td>2.84217e-14</td>
    </tr>
    <tr>
      <th>173</th>
      <td>80</td>
      <td>25.2665</td>
      <td>105.2665</td>
      <td>1.42109e-14</td>
    </tr>
    <tr>
      <th>594</th>
      <td>300</td>
      <td>426.9589</td>
      <td>726.9589</td>
      <td>1.13687e-13</td>
    </tr>
    <tr>
      <th>1325</th>
      <td>200</td>
      <td>109.3785</td>
      <td>309.3785</td>
      <td>5.68434e-14</td>
    </tr>
    <tr>
      <th>1464</th>
      <td>20</td>
      <td>44.5006</td>
      <td>64.5006</td>
      <td>-1.42109e-14</td>
    </tr>
    <tr>
      <th>2555</th>
      <td>88</td>
      <td>66.3470</td>
      <td>154.3470</td>
      <td>-2.84217e-14</td>
    </tr>
    <tr>
      <th>2749</th>
      <td>50</td>
      <td>127.4036</td>
      <td>177.4036</td>
      <td>-2.84217e-14</td>
    </tr>
    <tr>
      <th>2795</th>
      <td>30</td>
      <td>36.1350</td>
      <td>66.1350</td>
      <td>-1.42109e-14</td>
    </tr>
    <tr>
      <th>2828</th>
      <td>350</td>
      <td>88.5350</td>
      <td>438.5350</td>
      <td>-5.68434e-14</td>
    </tr>
    <tr>
      <th>3085</th>
      <td>150</td>
      <td>461.3427</td>
      <td>611.3427</td>
      <td>-1.13687e-13</td>
    </tr>
    <tr>
      <th>3260</th>
      <td>500</td>
      <td>260.6582</td>
      <td>760.6582</td>
      <td>1.13687e-13</td>
    </tr>
    <tr>
      <th>3357</th>
      <td>30</td>
      <td>21.1989</td>
      <td>51.1989</td>
      <td>-7.10543e-15</td>
    </tr>
    <tr>
      <th>3449</th>
      <td>20</td>
      <td>117.8676</td>
      <td>137.8676</td>
      <td>-2.84217e-14</td>
    </tr>
    <tr>
      <th>3519</th>
      <td>90</td>
      <td>70.0000</td>
      <td>159.9823</td>
      <td>0.0177</td>
    </tr>
    <tr>
      <th>3858</th>
      <td>15</td>
      <td>27.0716</td>
      <td>42.0716</td>
      <td>7.10543e-15</td>
    </tr>
    <tr>
      <th>4469</th>
      <td>450</td>
      <td>1800.0000</td>
      <td>2250.5130</td>
      <td>-0.513</td>
    </tr>
    <tr>
      <th>4517</th>
      <td>100</td>
      <td>249.9061</td>
      <td>349.9061</td>
      <td>5.68434e-14</td>
    </tr>
    <tr>
      <th>5216</th>
      <td>30</td>
      <td>125.5963</td>
      <td>155.5963</td>
      <td>-2.84217e-14</td>
    </tr>
    <tr>
      <th>6056</th>
      <td>45</td>
      <td>45.5200</td>
      <td>90.5200</td>
      <td>1.42109e-14</td>
    </tr>
    <tr>
      <th>6285</th>
      <td>70</td>
      <td>43.9600</td>
      <td>113.9600</td>
      <td>1.42109e-14</td>
    </tr>
    <tr>
      <th>6395</th>
      <td>180</td>
      <td>109.3785</td>
      <td>289.3785</td>
      <td>5.68434e-14</td>
    </tr>
    <tr>
      <th>6497</th>
      <td>40</td>
      <td>70.0000</td>
      <td>110.0135</td>
      <td>-0.0135</td>
    </tr>
    <tr>
      <th>6666</th>
      <td>505.37</td>
      <td>1520.9865</td>
      <td>2026.3565</td>
      <td>-2.27374e-13</td>
    </tr>
    <tr>
      <th>6688</th>
      <td>90</td>
      <td>1007.8950</td>
      <td>1100.0000</td>
      <td>-2.105</td>
    </tr>
    <tr>
      <th>6882</th>
      <td>50</td>
      <td>1151.7024</td>
      <td>1200.0000</td>
      <td>1.7024</td>
    </tr>
    <tr>
      <th>6917</th>
      <td>80</td>
      <td>81.0930</td>
      <td>161.0930</td>
      <td>2.84217e-14</td>
    </tr>
    <tr>
      <th>8073</th>
      <td>10</td>
      <td>28.5996</td>
      <td>38.5996</td>
      <td>-7.10543e-15</td>
    </tr>
    <tr>
      <th>9095</th>
      <td>2884.06</td>
      <td>28500.0000</td>
      <td>31400.0000</td>
      <td>-15.9428</td>
    </tr>
    <tr>
      <th>9723</th>
      <td>40</td>
      <td>49.2900</td>
      <td>89.2900</td>
      <td>-1.42109e-14</td>
    </tr>
    <tr>
      <th>9927</th>
      <td>380</td>
      <td>140.7439</td>
      <td>520.7439</td>
      <td>-1.13687e-13</td>
    </tr>
    <tr>
      <th>11151</th>
      <td>1000</td>
      <td>197.0519</td>
      <td>1200.0000</td>
      <td>-2.9481</td>
    </tr>
    <tr>
      <th>11947</th>
      <td>350</td>
      <td>1200.0000</td>
      <td>1553.2304</td>
      <td>-3.2304</td>
    </tr>
    <tr>
      <th>12303</th>
      <td>2873.34</td>
      <td>28500.0000</td>
      <td>31400.0000</td>
      <td>-26.6588</td>
    </tr>
    <tr>
      <th>12411</th>
      <td>200</td>
      <td>81.6937</td>
      <td>281.6937</td>
      <td>5.68434e-14</td>
    </tr>
    <tr>
      <th>12723</th>
      <td>600</td>
      <td>502.0000</td>
      <td>1100.0000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12774</th>
      <td>40</td>
      <td>53.8002</td>
      <td>93.8002</td>
      <td>-1.42109e-14</td>
    </tr>
    <tr>
      <th>12925</th>
      <td>80</td>
      <td>457.0592</td>
      <td>537.0592</td>
      <td>-1.13687e-13</td>
    </tr>
    <tr>
      <th>13058</th>
      <td>20</td>
      <td>48.0400</td>
      <td>68.0400</td>
      <td>-1.42109e-14</td>
    </tr>
    <tr>
      <th>13149</th>
      <td>450</td>
      <td>69.1678</td>
      <td>519.1678</td>
      <td>-1.13687e-13</td>
    </tr>
    <tr>
      <th>13703</th>
      <td>250</td>
      <td>204.9100</td>
      <td>454.9100</td>
      <td>-5.68434e-14</td>
    </tr>
    <tr>
      <th>14005</th>
      <td>150</td>
      <td>102.4480</td>
      <td>252.4480</td>
      <td>-2.84217e-14</td>
    </tr>
    <tr>
      <th>14251</th>
      <td>70</td>
      <td>465.6144</td>
      <td>535.6144</td>
      <td>-1.13687e-13</td>
    </tr>
    <tr>
      <th>14506</th>
      <td>350</td>
      <td>426.9589</td>
      <td>776.9589</td>
      <td>1.13687e-13</td>
    </tr>
    <tr>
      <th>14726</th>
      <td>90</td>
      <td>65.5160</td>
      <td>155.5160</td>
      <td>2.84217e-14</td>
    </tr>
    <tr>
      <th>14760</th>
      <td>90</td>
      <td>204.9100</td>
      <td>294.9100</td>
      <td>-5.68434e-14</td>
    </tr>
    <tr>
      <th>15000</th>
      <td>20</td>
      <td>29.6540</td>
      <td>49.6540</td>
      <td>-7.10543e-15</td>
    </tr>
    <tr>
      <th>15102</th>
      <td>125</td>
      <td>176.7150</td>
      <td>301.7150</td>
      <td>5.68434e-14</td>
    </tr>
  </tbody>
</table>
</div>



After doing a check, it seems that my assumption is correct. For most of the rows that do not fit the assumption, it just seems to be a rounding issue. I will proceed to fill in the null values of total_cash_usage with total_usage - total_retail_usage. Furthermore, I would think that it is impossible to have negative values for total_usage and total_retail_usage. In order to resolve the negative values, I will mulitply those values with -1.


```python
print("% of data points where total_retail_usage or total_usage is negative: " + str(round(len(combine.query("total_retail_usage < 0 or total_usage < 0"))/combine.shape[0] * 100, 1)) + '%')
```

    % of data points where total_retail_usage or total_usage is negative: 0.3%



```python
# assumption is correct
# fill missing null values of total_cash_usage with total_usage - total_retail_usage 
# correct rows where total_usage != total_cash_usage + total_retail_usage
# mulitply -1 with values where total_usage or total_retail_usage is negaive
combine = (
    combine
    .fillna({"total_cash_usage": -999})
    .pipe(lambda x: x.assign(total_retail_usage=np.where(x.total_retail_usage<0, -1*x.total_retail_usage, x.total_retail_usage)))
    .pipe(lambda x: x.assign(total_usage=np.where(x.total_usage<0, -1*x.total_usage, x.total_usage)))
    .pipe(lambda x: x.assign(total_cash_usage=np.where(x.total_cash_usage==-999, x.total_usage-x.total_retail_usage, x.total_cash_usage)))
    .pipe(lambda x: x.assign(total_usage_check=x.total_usage-x.total_cash_usage-x.total_retail_usage))
    .pipe(lambda x: x.assign(total_usage=np.where(x.total_usage_check!=0, x.total_cash_usage+x.total_retail_usage, x.total_usage)))
    .drop('total_usage_check', axis=1)
)
```


```python
combine[['total_usage_per_limit', 'total_3mo_usage_per_limit', 'total_6mo_usage_per_limit']].describe()
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
      <th>total_usage_per_limit</th>
      <th>total_3mo_usage_per_limit</th>
      <th>total_6mo_usage_per_limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15645.000000</td>
      <td>15645.000000</td>
      <td>15645.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.108977</td>
      <td>0.167157</td>
      <td>0.202598</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.199740</td>
      <td>0.192212</td>
      <td>0.274314</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.632000</td>
      <td>-0.126000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.034767</td>
      <td>0.032600</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.021700</td>
      <td>0.108000</td>
      <td>0.117000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.127143</td>
      <td>0.239000</td>
      <td>0.285000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>4.600000</td>
      <td>3.490000</td>
      <td>8.110000</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, I would think that it is improbable for the columns total_usage_per_limit and total_3mo_usage_per_limit to have negative values. To correct these values, I would make the assumption that total_usage_per_limit = total_usage/credit_limit. 


```python
# assume total_usage_per_limit is calculated as total_usage/credit_limit
check_assumption2 = (
    combine
    [['credit_limit', 'total_usage', 'total_usage_per_limit']]
    .pipe(lambda x: x.assign(total_usage_per_limit_recal=x.total_usage/x.credit_limit))
    .pipe(lambda x: x.assign(total_usage_per_limit_check=round(x.total_usage_per_limit_recal-abs(x.total_usage_per_limit), 1)))
)
```


```python
check_assumption2.query("total_usage_per_limit_check != 0")
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
      <th>credit_limit</th>
      <th>total_usage</th>
      <th>total_usage_per_limit</th>
      <th>total_usage_per_limit_recal</th>
      <th>total_usage_per_limit_check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4819</th>
      <td>300.0</td>
      <td>259.3022</td>
      <td>0.602</td>
      <td>0.864341</td>
      <td>0.3</td>
    </tr>
  </tbody>
</table>
</div>



All, but one entry agreeds with my assumption in total_usage_per_limit calculation logic. I will proceed to re-calculate the total_usage_per_limit to remove erroneous data like the above entry as well as negative values. As for resolving negative total_3mo_usage_per_limit, I will multiply -1 to the negative values.


```python
# other than index 4819, the assumption on the calculation of total_usage_per_limit seems right
# re-calculate total_usage_per_limit using total_usage/credit_limit to get rid of negative values of total_usage_per_limit
# multiply -1 to negative total_3mo_usage_per_limit
combine = (
    combine
    .pipe(lambda x: x.assign(total_usage_per_limit=x.total_usage/x.credit_limit))
    .pipe(lambda x: x.assign(total_3mo_usage_per_limit=np.where(x.total_3mo_usage_per_limit<0, -1*x.total_3mo_usage_per_limit, x.total_3mo_usage_per_limit)))
)
```


```python
combine[['payment_ratio', 'payment_ratio_3month', 'payment_ratio_6month']].describe()
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
      <th>payment_ratio</th>
      <th>payment_ratio_3month</th>
      <th>payment_ratio_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15645.000000</td>
      <td>15645.000000</td>
      <td>15645.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>64.438215</td>
      <td>50.496321</td>
      <td>81.544782</td>
    </tr>
    <tr>
      <th>std</th>
      <td>790.320478</td>
      <td>1320.816758</td>
      <td>1460.736051</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-18138.000000</td>
      <td>-57792.340000</td>
      <td>-77056.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>20.200000</td>
      <td>21.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>26.900000</td>
      <td>50.000000</td>
      <td>65.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>100.000000</td>
      <td>91.200000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>68983.000000</td>
      <td>75575.000000</td>
      <td>54899.000000</td>
    </tr>
  </tbody>
</table>
</div>



Assuming that payment is calculated as payment_ratio * bill, the payment for each month is outrageous. The median payment_ratio is approximately 27, meaning to say that 50% of the users are making payment of more than 27 times their last month bill. As such, I would assume that the unit of payment_ratio, payment_ratio_3month and payment_ratio_6month columns are in percentages. I will divide these columns by 100 to obtain the actual ratio. Furthermore, I will multiply -1 to negative payment ratios as it is logically impossible to have negative payment ratios.


```python
# divide payment ratios columns by 100 to get the real payment_ratio
# for payment ratios that are negative, I will mulitply by -1
combine = (
    combine
    .pipe(lambda x: x.assign(payment_ratio=x.payment_ratio/100))
    .pipe(lambda x: x.assign(payment_ratio=np.where(x.payment_ratio<0, -1*x.payment_ratio, x.payment_ratio)))
    .pipe(lambda x: x.assign(payment_ratio_3month=x.payment_ratio_3month/100))
    .pipe(lambda x: x.assign(payment_ratio_3month=np.where(x.payment_ratio_3month<0, -1*x.payment_ratio_3month, x.payment_ratio_3month)))
    .pipe(lambda x: x.assign(payment_ratio_6month=x.payment_ratio_6month/100))
    .pipe(lambda x: x.assign(payment_ratio_6month=np.where(x.payment_ratio_6month<0, -1*x.payment_ratio_6month, x.payment_ratio_6month)))
)
```


```python
combine[['payment_ratio', 'payment_ratio_3month', 'payment_ratio_6month']].describe()
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
      <th>payment_ratio</th>
      <th>payment_ratio_3month</th>
      <th>payment_ratio_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15645.000000</td>
      <td>15645.000000</td>
      <td>15645.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.701148</td>
      <td>1.092672</td>
      <td>1.191463</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.898371</td>
      <td>13.172573</td>
      <td>14.581506</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.207000</td>
      <td>0.222000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.271000</td>
      <td>0.501000</td>
      <td>0.660000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>0.921000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>689.830000</td>
      <td>755.750000</td>
      <td>770.560000</td>
    </tr>
  </tbody>
</table>
</div>



From the table above, we can see that there are some outliers in payment ratio columns. The maximum payment ratios is approximately 700 times the 75th percentile and it is probably impossible to make a payment 700x the customer's bill. As such, I would think that there are some erroneous data in these columns. In order to pick them out, I will use IQR to decide on my upper limit and I will treat everything above it as outliers.


```python
# using iqr to spot outliers
def upper_limit(df, column):
    iqr = df['{}'.format(column)].quantile(q=0.75) - df['{}'.format(column)].quantile(q=0.25)
    upper_limit = df['{}'.format(column)].quantile(q=0.75) + 1.5*iqr
    print("upper limit of {} based on 1.5*IQR: ".format(column) + str(upper_limit))
```


```python
upper_limit(combine, 'payment_ratio')
upper_limit(combine, 'payment_ratio_3month')
upper_limit(combine, 'payment_ratio_6month')
```

    upper limit of payment_ratio based on 1.5*IQR: 2.5
    upper limit of payment_ratio_3month based on 1.5*IQR: 1.992
    upper limit of payment_ratio_6month based on 1.5*IQR: 2.167



```python
print("% of data points where payment_ratio > 2.5: " + str(round(len(combine.query("payment_ratio > 2.5"))/len(combine)*100, 1)) + '%')
print("% of data points where payment_ratio_3month > 2.5: " + str(round(len(combine.query("payment_ratio_3month > 1.992"))/len(combine)*100, 1)) + '%')
print("% of data points where payment_ratio_6month > 2.5: " + str(round(len(combine.query("payment_ratio_6month > 2.167"))/len(combine)*100, 1)) + '%')
```

    % of data points where payment_ratio > 2.5: 0.7%
    % of data points where payment_ratio_3month > 2.5: 1.3%
    % of data points where payment_ratio_6month > 2.5: 1.6%


We can see that a small portion of the data are outliers. I will choose to drop them instead of replacing those values.


```python
# since the % of outliers is low, I will drop these rows
print("% of data points that will be dropped: " + str(round(len(combine.query("payment_ratio > 2.5 or payment_ratio_3month > 1.992 or payment_ratio_6month > 2.167"))/len(combine)*100, 1)) + '%')
combine = combine.query("payment_ratio <= 2.5 and payment_ratio_3month <= 1.992 and payment_ratio_6month <= 2.167")
```

    % of data points that will be dropped: 2.8%



```python
print("% of data points where payment_ratio equals to 0: " + str(round(len(combine.query("payment_ratio == 0"))/len(combine)*100, 1)) + '%')
print("% of data points where payment_ratio equals to 0: " + str(round(len(combine.query("payment_ratio_3month == 0"))/len(combine)*100, 1)) + '%')
print("% of data points where payment_ratio equals to 0: " + str(round(len(combine.query("payment_ratio_6month == 0"))/len(combine)*100, 1)) + '%')
```

    % of data points where payment_ratio equals to 0: 31.0%
    % of data points where payment_ratio equals to 0: 3.0%
    % of data points where payment_ratio equals to 0: 5.1%



```python
combine.query("payment_ratio != 0")[['payment_ratio', 'payment_ratio_3month', 'payment_ratio_6month']].corr()
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
      <th>payment_ratio</th>
      <th>payment_ratio_3month</th>
      <th>payment_ratio_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>payment_ratio</th>
      <td>1.000000</td>
      <td>0.720974</td>
      <td>0.489230</td>
    </tr>
    <tr>
      <th>payment_ratio_3month</th>
      <td>0.720974</td>
      <td>1.000000</td>
      <td>0.717471</td>
    </tr>
    <tr>
      <th>payment_ratio_6month</th>
      <td>0.489230</td>
      <td>0.717471</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



Taking a deeper look into payment_ratio, we can see that approximately 1/3 of the data have a value of 0. It is quite unlikely for such a sizeable portion of the data to be 0 as such a phenomenon cannot be observed in payment_ratio_3month and payment_ratio_6month. I would assume that these data in payment_ratio are erroneous. Since payment_ratio and payment_ratio_3month have a strong correlation, I will be replacing the 0 values in payment_ratio with payment_ratio_3month values.


```python
# replacing 0s in payment_ratio with payment_ratio_3month values
combine = (
    combine
    .pipe(lambda x: x.assign(payment_ratio=np.where(x.payment_ratio==0, x.payment_ratio_3month, x.payment_ratio)))
)
```


```python
combine[['user_id', 'overlimit_percentage', 'delinquency_score']].describe()
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
      <th>user_id</th>
      <th>overlimit_percentage</th>
      <th>delinquency_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15202.000000</td>
      <td>15176.000000</td>
      <td>15115.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7818.466452</td>
      <td>3.291069</td>
      <td>0.033741</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4518.125737</td>
      <td>9.030692</td>
      <td>0.360736</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3908.250000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7805.500000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11735.750000</td>
      <td>1.030000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15645.000000</td>
      <td>190.000000</td>
      <td>5.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
combine.delinquency_score.value_counts()
```




    0.0    14958
    5.0       42
    3.0       33
    4.0       33
    1.0       29
    2.0       20
    Name: delinquency_score, dtype: int64




```python
combine[combine.delinquency_score.isnull()].groupby("default_flag").agg({"user_id": "count"}).reset_index()
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
      <th>default_flag</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>87</td>
    </tr>
  </tbody>
</table>
</div>



Since all of the data points where delinquency_score is null are non-defaulters, I will impute the null values with 0 as it is the mode of the column


```python
# imputing delinquency_score null values with mode
combine.loc[combine[combine.delinquency_score.isnull()].index.values, 'delinquency_score'] = 0
```

In order to impute the null values in overlimit_percentage, I shall make the assumption that overlimit_percentage = outstanding/credit_limit. If this assumption is true, I would be able to use this logic to impute overlimit_percentage null values.


```python
# assume that overlimit_percentage is calculated using outstanding divide by credit_limit
# check assumption
check_assumption3 = (
    combine
    [combine.overlimit_percentage.notnull()]
    [['overlimit_percentage', 'credit_limit', 'outstanding', 'default_flag']]
    .pipe(lambda x: x.assign(overlimit_percentage_recal=np.where(x.outstanding>x.credit_limit, (x.outstanding-x.credit_limit)/x.credit_limit * 100, 0)))
    .query("overlimit_percentage != overlimit_percentage_recal")
)
```


```python
check_assumption3.shape
```




    (4276, 5)



Approximately 1/3 of the data does not agreed with my above assumption. I will refrain from using my assumed logic to impute missing values in overlimit_percentage. Instead I will drop these rows. I will also divide the values in the column by 100 so that it would be of the same scale as the other columns


```python
# dropping rows where there are missing values
# dividing overlimit_percentage by 100
combine.dropna(inplace=True)
combine = (
    combine
    .pipe(lambda x: x.assign(overlimit_percentage=x.overlimit_percentage/100))
)
```


```python
combine[['utilization_3month', 'utilization_6month']].describe()
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
      <th>utilization_3month</th>
      <th>utilization_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15176.000000</td>
      <td>15176.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.571800</td>
      <td>0.534719</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.408624</td>
      <td>0.424357</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000288</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.191968</td>
      <td>0.156066</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.578000</td>
      <td>0.502000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.918000</td>
      <td>0.865000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.125671</td>
      <td>9.730000</td>
    </tr>
  </tbody>
</table>
</div>




```python
upper_limit(combine, 'utilization_3month')
upper_limit(combine, 'utilization_6month')
```

    upper limit of utilization_3month based on 1.5*IQR: 2.007047995125
    upper limit of utilization_6month based on 1.5*IQR: 1.928400357625



```python
print("% of data where utilization_3month or utilization_6month are outliers: " + str(round(len(combine.query("utilization_3month > 2 or utilization_6month > 1.928"))/len(combine)*100, 1)) + '%')
```

    % of data where utilization_3month or utilization_6month are outliers: 0.4%


Again, we can see that there are some outliers in the data and I will choose to drop them instead of replacing their values as the proportion is very small.


```python
# dropping these rows that has outliers
combine = combine.query("utilization_3month <= 2 and utilization_6month <= 1.928")
```


```python
combine.shape
```




    (15121, 24)




```python
combine.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 15121 entries, 0 to 15642
    Data columns (total 24 columns):
    user_id                               15121 non-null int64
    branch_code                           15121 non-null object
    default_flag                          15121 non-null int64
    number_of_cards                       15121 non-null int64
    outstanding                           15121 non-null float64
    credit_limit                          15121 non-null float64
    bill                                  15121 non-null float64
    total_cash_usage                      15121 non-null float64
    total_retail_usage                    15121 non-null float64
    total_usage                           15121 non-null float64
    total_usage_per_limit                 15121 non-null float64
    total_3mo_usage_per_limit             15121 non-null float64
    total_6mo_usage_per_limit             15121 non-null float64
    remaining_bill                        15121 non-null float64
    payment_ratio                         15121 non-null float64
    payment_ratio_3month                  15121 non-null float64
    payment_ratio_6month                  15121 non-null float64
    overlimit_percentage                  15121 non-null float64
    delinquency_score                     15121 non-null float64
    years_since_card_issuing              15121 non-null float64
    remaining_bill_per_number_of_cards    15121 non-null float64
    remaining_bill_per_limit              15121 non-null float64
    utilization_3month                    15121 non-null float64
    utilization_6month                    15121 non-null float64
    dtypes: float64(20), int64(3), object(1)
    memory usage: 2.9+ MB



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
      <th>user_id</th>
      <th>default_flag</th>
      <th>number_of_cards</th>
      <th>outstanding</th>
      <th>credit_limit</th>
      <th>bill</th>
      <th>total_cash_usage</th>
      <th>total_retail_usage</th>
      <th>total_usage</th>
      <th>total_usage_per_limit</th>
      <th>total_3mo_usage_per_limit</th>
      <th>total_6mo_usage_per_limit</th>
      <th>remaining_bill</th>
      <th>payment_ratio</th>
      <th>payment_ratio_3month</th>
      <th>payment_ratio_6month</th>
      <th>overlimit_percentage</th>
      <th>delinquency_score</th>
      <th>years_since_card_issuing</th>
      <th>remaining_bill_per_number_of_cards</th>
      <th>remaining_bill_per_limit</th>
      <th>utilization_3month</th>
      <th>utilization_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.00000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
      <td>15121.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7809.062033</td>
      <td>0.088817</td>
      <td>2.506514</td>
      <td>1138.332563</td>
      <td>2062.952186</td>
      <td>795.065258</td>
      <td>7.265117</td>
      <td>196.51146</td>
      <td>203.776577</td>
      <td>0.107609</td>
      <td>0.163087</td>
      <td>0.194305</td>
      <td>802.539513</td>
      <td>0.583380</td>
      <td>0.542882</td>
      <td>0.608554</td>
      <td>0.031843</td>
      <td>0.025792</td>
      <td>6.625007</td>
      <td>293.883776</td>
      <td>0.469097</td>
      <td>0.566323</td>
      <td>0.526435</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4509.277341</td>
      <td>0.284489</td>
      <td>1.013716</td>
      <td>2315.642473</td>
      <td>2831.500701</td>
      <td>1630.221261</td>
      <td>61.550189</td>
      <td>758.39025</td>
      <td>775.544559</td>
      <td>0.195220</td>
      <td>0.179391</td>
      <td>0.236645</td>
      <td>1785.041507</td>
      <td>0.425452</td>
      <td>0.368445</td>
      <td>0.403279</td>
      <td>0.084034</td>
      <td>0.307957</td>
      <td>4.660911</td>
      <td>581.627239</td>
      <td>0.461281</td>
      <td>0.389670</td>
      <td>0.391664</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>2.004300</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.750000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000288</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3908.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>201.255200</td>
      <td>500.000000</td>
      <td>83.034000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034400</td>
      <td>0.032300</td>
      <td>0.000000</td>
      <td>0.178000</td>
      <td>0.202000</td>
      <td>0.214000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.920000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.191000</td>
      <td>0.156000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7797.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>471.464900</td>
      <td>900.000000</td>
      <td>316.070400</td>
      <td>0.000000</td>
      <td>20.05760</td>
      <td>24.578900</td>
      <td>0.021972</td>
      <td>0.106000</td>
      <td>0.115565</td>
      <td>280.124700</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>0.635000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.666667</td>
      <td>122.359400</td>
      <td>0.354000</td>
      <td>0.576000</td>
      <td>0.498000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11719.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>1065.561900</td>
      <td>2200.000000</td>
      <td>743.775000</td>
      <td>0.000000</td>
      <td>138.56570</td>
      <td>150.000000</td>
      <td>0.126093</td>
      <td>0.234000</td>
      <td>0.278000</td>
      <td>721.443200</td>
      <td>1.000000</td>
      <td>0.880000</td>
      <td>1.000000</td>
      <td>0.009900</td>
      <td>0.000000</td>
      <td>9.330000</td>
      <td>312.601000</td>
      <td>0.931701</td>
      <td>0.915786</td>
      <td>0.862000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>15643.000000</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>79805.857400</td>
      <td>100000.000000</td>
      <td>41900.000000</td>
      <td>2884.057200</td>
      <td>28500.00000</td>
      <td>31384.057200</td>
      <td>4.603448</td>
      <td>2.460000</td>
      <td>4.550000</td>
      <td>44400.000000</td>
      <td>2.460000</td>
      <td>1.990000</td>
      <td>2.160000</td>
      <td>1.900000</td>
      <td>5.000000</td>
      <td>34.416667</td>
      <td>13100.000000</td>
      <td>2.200000</td>
      <td>1.980000</td>
      <td>1.920000</td>
    </tr>
  </tbody>
</table>
</div>



After cleaning the data, I am left with 15,121 rows.

# EDA

The data provided revolves around historical credit card transactions, usage and payment details. However, it does not seem that the default_flag refers to credit card default as the payment_ratio is not 0 for defaulters.

Taking into account that GoFin provides credit loan to users, I would assume that the default_flag refers to users who took up credit loan from GoFin default or not. The credit card details would assist GoFin in making the decision to lend money to users. My end goal in mind is to come up with a **predictive model to ascertain whether a user will default on his/her loan to GoFin**. The by-product of the model will also inform me on the **importance of each feature to determine defaulters**.

I would also assume that the branch_code refers to GoFin branches that gives out loans to users.

### Summary
- Data set is highly skewed, with baseline default rate to be at 9.7%
- Branch D and G have lower default rate than baseline while Branch E is higher than baseline. Personnel in Branch D and G are filtering out the right users to lend money to. Best practices should be shared across the branches
- Certain features are highly correlated and I have removed them from further analysis as the analysis would be similar
- Users who owns 2 or 3 credit cards have lower default rates as compared to the rest
- Users who owns 2 or 3 credit cards also tend to have lower credit limit. I have inferred that this group of users have lower income (due to lower credit limit) and hence, they are more conscious in spending. As such, they do not default as much. This is a point GoFin could take note off when a user of a high income apply for a loan.
- **delinquency_score is a great feature to ascertain whether a not a user is going to default**. If a user has a delinquency_score > 0, he/she is likely to default 80% of the time. GoFin should not loan money to anyone who has a delinquency_score > 0
- It is interesting to note that there are defaulters even though outstanding amount is very low. This group of people seems are the **poor and needy** and they require loans to get by their daily lives. They have low spending history and have high payment_ratio than other defaulters. Despite knowing that this group might default, GoFin should nevertheless **help them financially and should spread their repayment plan across a longer period of time, while keeping interest rate low**.
- First sign of financial duress can be first spotted 3 months ago when user spend drop slightly as compared to his/her longer term spend. GoFin could use this information to **target potential borrowers** and assist them in better financial planning to ensure that they do not default.
- There is a group of users who spend as per normal despite having financial problems. This group of people have higher credit limit, which implies higher income. GoFin should **charge them higher interest** as this will not only disuade them from overborrowing, GoFin is also able to maximise profits as this higher income group are able to make the higher interest repayment
- As expected, payment ratios of defaulters are lower than non-defaulters. GoFin could potentially **use these payment ratio metrics as simple method to gauge potential defaulters**. For example, if the interest rate of a loan from GoFin is above their current payment_ratio, then GoFin should not loan money out.
- Users who took loans from Branch D and G have relatively higher total_3mo_usage_per_limit and payment_ratio as compared to other branches. This could be the reason why Branch D and G have lower default rate than baseline. 


```python
combine.groupby("default_flag").agg({"user_id": "count"}).reset_index()
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
      <th>default_flag</th>
      <th>user_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>13778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1343</td>
    </tr>
  </tbody>
</table>
</div>




```python
# the data is skewed towards non-defaulters
print("baseline default rate: " + str(round(1343/13778*100, 1)) + '%')
```

    baseline default rate: 9.7%



```python
def cat_var_processing(df, column):
    processed_df = (
        df
        .groupby(["{}".format(column), "default_flag"])
        .agg({"user_id": "count"})
        .reset_index()
        .pivot_table(columns='default_flag', index='{}'.format(column), values='user_id', aggfunc=np.sum)
        .reset_index()
        .rename(columns={0: "non_default", 1: "default"})
        .fillna(0)
        .pipe(lambda x: x.assign(total=x.non_default+x.default))
        .pipe(lambda x: x.assign(total_pctg=100))
        .pipe(lambda x: x.assign(non_default_pctg=round(x.non_default/(x.total)*100, 1)))
        .pipe(lambda x: x.assign(default_pctg=x.total_pctg-x.non_default_pctg))
    )
    
    return processed_df
```


```python
def plot_cat_var(df, column):  
    overall_plot = sns.barplot(x = df['{}'.format(column)], y = df.total_pctg, color = "red", label='default')
    bottom_plot = sns.barplot(x = df['{}'.format(column)], y = df.non_default_pctg, color = "green", label='non_default')
    _ = plt.title('Breakdown of default status by {}'.format(column))
    _ = plt.legend(loc=4, ncol = 1)
    _ = plt.ylabel('percentage')
```


```python
branch_defaulters = cat_var_processing(combine, 'branch_code')
```


```python
branch_defaulters = (
    combine
    .groupby(["branch_code", "default_flag"])
    .agg({"user_id": "count"})
    .reset_index()
    .pivot_table(columns='default_flag', index='branch_code', values='user_id', aggfunc=np.sum)
    .reset_index()
    .rename(columns={0: "non_default", 1: "default"})
    .pipe(lambda x: x.assign(total=x.non_default+x.default))
    .pipe(lambda x: x.assign(total_pctg=100))
    .pipe(lambda x: x.assign(non_default_pctg=round(x.non_default/(x.total)*100, 1)))
    .pipe(lambda x: x.assign(default_pctg=x.total_pctg-x.non_default_pctg))
)
```


```python
branch_defaulters[['branch_code', 'total', 'default_pctg']]
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
      <th>default_flag</th>
      <th>branch_code</th>
      <th>total</th>
      <th>default_pctg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>8644</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1309</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>323</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>201</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>599</td>
      <td>14.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>f</td>
      <td>1619</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>g</td>
      <td>543</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>h</td>
      <td>360</td>
      <td>8.9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>i</td>
      <td>964</td>
      <td>10.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>j</td>
      <td>387</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>k</td>
      <td>172</td>
      <td>9.9</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_cat_var(branch_defaulters, 'branch_code')
```


![png](assignment2-3-4_files/assignment2-3-4_61_0.png)


Most loans are given out by Branch A. Branch E has the worse default rates out of all the branches and Branch D and G are the better performing ones.

It would be good for personnel in Branch D and G to share their best practice on how to spot potential defaulters, so that default rate will be lowered.

Let's take a deeper look into the data on how Branch D and G spot potential defaulters. 

Before I continue with data exploration, I will first look at the correlation between each variables. I will remove either of the highly correlated variables as there are currently too many variables to look at and highly correlated variables will tell the same story.


```python
# plotting a heatmap of a correlation matrix of all the continuous variables
cont_var_df = combine[['number_of_cards',
       'outstanding', 'credit_limit', 'total_cash_usage',
       'total_retail_usage', 'total_usage', 'total_usage_per_limit',
       'total_3mo_usage_per_limit', 'total_6mo_usage_per_limit',
       'bill', 'remaining_bill', 'payment_ratio', 'payment_ratio_3month',
       'payment_ratio_6month', 'overlimit_percentage', 'delinquency_score',
       'years_since_card_issuing', 'remaining_bill_per_number_of_cards',
       'remaining_bill_per_limit', 'utilization_3month', 'utilization_6month']]

_ = plt.subplots(figsize = (15,10))

corr = cont_var_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(600, 20, as_cmap=True)

_ = sns.heatmap(corr, mask=mask, annot=True, cmap =cmap)
```


![png](assignment2-3-4_files/assignment2-3-4_63_0.png)


We can observation strong correlation between:
- outstanding and bill, outstanding and remaining_bill. I will drop bill and remaining_bill
- total_usage and total_retail_usage. I will drop total_usage as total_cash_usage is a subset of total_usage
- payment_ratio and payment_ratio_3month. I will drop payment_ratio_3_month. Although I should note that I have replace 1/3 of payment_ratio with payment_ratio_3month
- utilization_3month and utilization_6month, utilization_3month and remaining_bill_per_limit. I will drop utilization_3month


```python
# dropping highly correlated variables
combine2 = combine.drop(['bill', 'remaining_bill', 'total_usage', 'payment_ratio_3month', 'utilization_3month'], axis=1)
```

## Categorical Features


```python
no_cards_defaulters = cat_var_processing(combine, 'number_of_cards')
```


```python
no_cards_defaulters
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
      <th>default_flag</th>
      <th>number_of_cards</th>
      <th>non_default</th>
      <th>default</th>
      <th>total</th>
      <th>total_pctg</th>
      <th>non_default_pctg</th>
      <th>default_pctg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>63.0</td>
      <td>9.0</td>
      <td>72.0</td>
      <td>100</td>
      <td>87.5</td>
      <td>12.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>9585.0</td>
      <td>899.0</td>
      <td>10484.0</td>
      <td>100</td>
      <td>91.4</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2477.0</td>
      <td>249.0</td>
      <td>2726.0</td>
      <td>100</td>
      <td>90.9</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>993.0</td>
      <td>109.0</td>
      <td>1102.0</td>
      <td>100</td>
      <td>90.1</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>409.0</td>
      <td>41.0</td>
      <td>450.0</td>
      <td>100</td>
      <td>90.9</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>134.0</td>
      <td>15.0</td>
      <td>149.0</td>
      <td>100</td>
      <td>89.9</td>
      <td>10.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>59.0</td>
      <td>14.0</td>
      <td>73.0</td>
      <td>100</td>
      <td>80.8</td>
      <td>19.2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>24.0</td>
      <td>3.0</td>
      <td>27.0</td>
      <td>100</td>
      <td>88.9</td>
      <td>11.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>13.0</td>
      <td>3.0</td>
      <td>16.0</td>
      <td>100</td>
      <td>81.2</td>
      <td>18.8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>100</td>
      <td>100.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>100</td>
      <td>100.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>100</td>
      <td>100.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>100</td>
      <td>100.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>100</td>
      <td>100.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>16</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>100</td>
      <td>50.0</td>
      <td>50.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_cat_var(no_cards_defaulters, 'number_of_cards')
```


![png](assignment2-3-4_files/assignment2-3-4_69_0.png)


I will disregard observations made on customers having 10 or more cards as the sample size is way too small.

We can observe that most customer only have 2 or 3 credit cards and default rate (8.6% and 9.1% respectively) of these group of people are below the baseline default rate of 9.7%. 

Customers having 5 or more cards tend to have a higher default rate. However, it is worth to note that the sample size of these data points are quite small as well.

Interestingly, the default rate of customers having 1 credit card is above the average default rate. It could be that due to poor credit scoring, they were unable to obtain a second credit card.

Since having 2 or 3 credit cards is the norm and the default rate of these groups are below the baseline and the fact that the data points in other groups are sparse, I will engineer a new feature to split number_of_cards to two groups, having_2_or_3_cards or not. 


```python
combine2 = (
    combine2
    .pipe(lambda x: x.assign(credit_limit_per_card=x.credit_limit/x.number_of_cards))
    .pipe(lambda x: x.assign(having_2_or_3_cards=np.where(((x.number_of_cards == 2)|(x.number_of_cards == 3)), 1, 0)))
)
```


```python
_ = sns.boxplot(x='having_2_or_3_cards', y='credit_limit_per_card', data=combine2)
_.set_yscale('log')
```


![png](assignment2-3-4_files/assignment2-3-4_72_0.png)


We can see that the credit_limit_per_card of users having 2 or 3 cards is lower that those who have 1 credit card or more than 3 credit cards. Normally credit limit is determine by a person's income. Having a lower credit limit would imply that the user's income is lower as well. Furthermore, we know what the default rate of users having 2 or 3 credit cards are lower than baseline, while the rest are higher. We can infer that perhaps users who are of a lower income group are more down-to-earth and have tigher spends as such, they do not spend excessively that will result in a default.


```python
delinquency_defaulters = cat_var_processing(combine, 'delinquency_score')
```


```python
delinquency_defaulters
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
      <th>default_flag</th>
      <th>delinquency_score</th>
      <th>non_default</th>
      <th>default</th>
      <th>total</th>
      <th>total_pctg</th>
      <th>non_default_pctg</th>
      <th>default_pctg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>13752.0</td>
      <td>1239.0</td>
      <td>14991.0</td>
      <td>100</td>
      <td>91.7</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>22.0</td>
      <td>7.0</td>
      <td>29.0</td>
      <td>100</td>
      <td>75.9</td>
      <td>24.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>18.0</td>
      <td>20.0</td>
      <td>100</td>
      <td>10.0</td>
      <td>90.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>31.0</td>
      <td>33.0</td>
      <td>100</td>
      <td>6.1</td>
      <td>93.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>100</td>
      <td>0.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>100</td>
      <td>0.0</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_cat_var(delinquency_defaulters, 'delinquency_score')
```


![png](assignment2-3-4_files/assignment2-3-4_76_0.png)


Delinquency_score has an obvious correlation to default rate. It is obvious that GoFin should not be giving out loans to users who have 1 or more.


```python
delinquency_more_than_0_defaulters = (
    combine2
    .pipe(lambda x: x.assign(delinquency_score_more_than_0=np.where(x.delinquency_score>=1, 1, 0)))
    .groupby(['delinquency_score_more_than_0', 'branch_code'])
    .agg({"user_id": "count"})
    .reset_index()
    .pivot_table(columns='delinquency_score_more_than_0', index='branch_code', values='user_id', aggfunc=np.sum)
    .reset_index()
    .rename(columns={0: "delinquency_score_0", 1: "delinquency_score_more_than_0"})
    .pipe(lambda x: x.assign(score_more_than_0_pctg=round(x.delinquency_score_more_than_0/(x.delinquency_score_more_than_0+x.delinquency_score_0)*100, 1)))
)
```


```python
delinquency_more_than_0_defaulters.merge(branch_defaulters, on='branch_code', how='left')[['branch_code', 'score_more_than_0_pctg', 'default_pctg']]
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
      <th>branch_code</th>
      <th>score_more_than_0_pctg</th>
      <th>default_pctg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>0.3</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>0.2</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>0.6</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>d</td>
      <td>6.5</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>e</td>
      <td>5.5</td>
      <td>14.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>f</td>
      <td>0.2</td>
      <td>8.6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>g</td>
      <td>0.2</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>h</td>
      <td>0.3</td>
      <td>8.9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>i</td>
      <td>0.3</td>
      <td>10.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>j</td>
      <td>8.3</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>k</td>
      <td>9.9</td>
      <td>9.9</td>
    </tr>
  </tbody>
</table>
</div>



If Branch D, J & K were to pay more attention to delinquency_score, they would have 0 or close to 0 default rate.

Not paying attention to delinquency_score is also the main reason why Branch E have such a high default rate.

As the majority users (~99%) have a delinquency_score of 0, it would be pointless to have this feature in my model. I will drop this feature moving forward.


```python
combine2.drop('delinquency_score', axis=1, inplace=True)
```

## Continuous Features


```python
def plot_cont_var(df, column, boxplot=True, log=True, ax=1, x='default_flag'):
    if boxplot:
        if log:
            _ = sns.boxplot(x='{}'.format(x), y='{}'.format(column), data=df)
            _.set_yscale('symlog')
            _ = plt.title("Boxplot of {} (symlog scale)".format(column))
        else:
            _ = sns.boxplot(x='{}'.format(x), y='{}'.format(column), data=df)
            _ = plt.title("Boxplot of {}".format(column))
    else:
        if log:
            _ = sns.distplot(np.log(df.query("{} == 0".format(x))['{}'.format(column)]), ax=ax, label='non_default')
            _ = sns.distplot(np.log(df.query("{} == 1".format(x))['{}'.format(column)]), ax=ax, label='default')
            _ = plt.title("Distribution of {}".format(column))
            _ = ax.legend()
        else:
            _ = sns.distplot(df.query("{} == 0".format(x))['{}'.format(column)], ax=ax, label='non_default')
            _ = sns.distplot(df.query("{} == 0".format(x))['{}'.format(column)], ax=ax, label='default')
            _ = plt.title("Distribution of {}".format(column))
            _ = ax.legend()
```


```python
_ = plt.figure(figsize=(25,15))
start = 441
for idx, col in enumerate(['outstanding', 'credit_limit', 'total_cash_usage', 'total_retail_usage',
       'total_usage_per_limit', 'total_3mo_usage_per_limit',
       'total_6mo_usage_per_limit']):
    _ = plt.subplot(start + idx)
    _ = sns.boxplot(x='default_flag', y='{}'.format(col), data=combine2)
    _ = plt.title("Boxplot of {}".format(col))

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
```


![png](assignment2-3-4_files/assignment2-3-4_84_0.png)


Due to data skew, it is hard to see the distribution of each feature. I will re-plot the graphs again using symlog scale or removing skewed data points. 


```python
# re-plotting using symlog scale
_ = plt.figure(figsize=(25,15))
ax1 = plt.subplot(441)
plot_cont_var(combine2, 'outstanding', boxplot=True, log=True, ax=ax1)

ax2 = plt.subplot(442)
plot_cont_var(combine2, 'credit_limit', boxplot=True, log=True, ax=ax2)

ax3 = plt.subplot(443)
plot_cont_var(combine2, 'credit_limit_per_card', boxplot=True, log=True, ax=ax3)
```


![png](assignment2-3-4_files/assignment2-3-4_86_0.png)


The symlog scale will set a range around zero within the plot to be linear instead of logarithmic. This allows me to handle zero values.

We can see that distribution spread of outstanding amount for defaulters is quite wide. Defaulters are skewed towards having higher outstanding amount than non-defaulters. It is interesting to note that **there are defaulters even when the outstanding amount is very low**. I would think that GoFin should pay special attention to these group of people. It seems that **these are the poor and needy** and had to take a loan with GoFin probably to get by their daily lives. From the below table, we can see that these group of people have very **low spending history and their payment_ratio is higher than those defaulters who have high outstanding amount**. We can also observe that although the payment_ratio is higher than their counterpart, payment_ratio_6month is lower. This could mean that their source of income is unstable and could only make payment when they have an inflow of cash. Perhaps, GoFin could **spread their repayment plan across a longer period of time than norm while keeping interest rate low, so to help them financially as well as to not incur defaultment**.

As for credit limit, not much observation can be made. Similar to the above inference regarding credit_limit_per_card, the credit limit for defaulter is slightly higher than non-defaulters.


```python
# poor and needy group
(
    combine
    .query("default_flag == 1 and outstanding < {}".format(combine2.outstanding.quantile(q=0.25)))
    [['outstanding', 'credit_limit', 'total_retail_usage', 'total_3mo_usage_per_limit', 'total_6mo_usage_per_limit', 'payment_ratio', 
      'payment_ratio_6month', 'remaining_bill_per_number_of_cards', 'remaining_bill_per_limit', 'utilization_6month']]
    .describe()
)
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
      <th>outstanding</th>
      <th>credit_limit</th>
      <th>total_retail_usage</th>
      <th>total_3mo_usage_per_limit</th>
      <th>total_6mo_usage_per_limit</th>
      <th>payment_ratio</th>
      <th>payment_ratio_6month</th>
      <th>remaining_bill_per_number_of_cards</th>
      <th>remaining_bill_per_limit</th>
      <th>utilization_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
      <td>395.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>25.823359</td>
      <td>1806.329114</td>
      <td>3.412340</td>
      <td>0.014330</td>
      <td>0.035575</td>
      <td>0.329988</td>
      <td>0.345148</td>
      <td>8.456062</td>
      <td>0.034319</td>
      <td>0.119096</td>
    </tr>
    <tr>
      <th>std</th>
      <td>42.212225</td>
      <td>2251.223118</td>
      <td>17.863162</td>
      <td>0.044898</td>
      <td>0.113555</td>
      <td>0.366172</td>
      <td>0.431215</td>
      <td>20.216930</td>
      <td>0.103596</td>
      <td>0.212897</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.254800</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002025</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.483900</td>
      <td>700.000000</td>
      <td>0.000000</td>
      <td>0.000018</td>
      <td>0.000000</td>
      <td>0.250000</td>
      <td>0.099900</td>
      <td>1.087800</td>
      <td>0.000656</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>18.026100</td>
      <td>2100.000000</td>
      <td>0.062950</td>
      <td>0.004950</td>
      <td>0.009695</td>
      <td>0.500000</td>
      <td>0.649000</td>
      <td>4.590750</td>
      <td>0.011450</td>
      <td>0.129000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>201.025900</td>
      <td>18000.000000</td>
      <td>291.352900</td>
      <td>0.426000</td>
      <td>1.160000</td>
      <td>1.920000</td>
      <td>1.940000</td>
      <td>100.513000</td>
      <td>0.670000</td>
      <td>1.110000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# the rest
(
    combine
    .query("default_flag == 1 and outstanding >= {}".format(combine2.outstanding.quantile(q=0.25)))
    [['outstanding', 'credit_limit', 'total_retail_usage', 'total_3mo_usage_per_limit', 'total_6mo_usage_per_limit', 'payment_ratio', 
      'payment_ratio_6month', 'remaining_bill_per_number_of_cards', 'remaining_bill_per_limit', 'utilization_6month']]
    .describe()
)
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
      <th>outstanding</th>
      <th>credit_limit</th>
      <th>total_retail_usage</th>
      <th>total_3mo_usage_per_limit</th>
      <th>total_6mo_usage_per_limit</th>
      <th>payment_ratio</th>
      <th>payment_ratio_6month</th>
      <th>remaining_bill_per_number_of_cards</th>
      <th>remaining_bill_per_limit</th>
      <th>utilization_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>948.000000</td>
      <td>948.000000</td>
      <td>948.000000</td>
      <td>948.000000</td>
      <td>948.000000</td>
      <td>948.000000</td>
      <td>948.000000</td>
      <td>948.000000</td>
      <td>948.000000</td>
      <td>948.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2390.671699</td>
      <td>2443.037975</td>
      <td>90.716620</td>
      <td>0.162083</td>
      <td>0.250389</td>
      <td>0.309453</td>
      <td>0.452363</td>
      <td>695.432874</td>
      <td>0.912413</td>
      <td>0.852413</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4403.038995</td>
      <td>4481.382825</td>
      <td>483.714433</td>
      <td>0.184896</td>
      <td>0.265557</td>
      <td>0.302241</td>
      <td>0.369753</td>
      <td>950.706427</td>
      <td>0.360888</td>
      <td>0.309681</td>
    </tr>
    <tr>
      <th>min</th>
      <td>203.759200</td>
      <td>300.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>502.258375</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>0.038600</td>
      <td>0.070475</td>
      <td>0.101000</td>
      <td>0.139000</td>
      <td>182.442225</td>
      <td>0.772750</td>
      <td>0.664000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>893.138800</td>
      <td>1000.000000</td>
      <td>0.000000</td>
      <td>0.101000</td>
      <td>0.157500</td>
      <td>0.196000</td>
      <td>0.320000</td>
      <td>349.584900</td>
      <td>1.028917</td>
      <td>0.897225</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2410.627950</td>
      <td>2825.000000</td>
      <td>4.800000</td>
      <td>0.232000</td>
      <td>0.354000</td>
      <td>0.402500</td>
      <td>0.702250</td>
      <td>819.160842</td>
      <td>1.130000</td>
      <td>1.040000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79805.857400</td>
      <td>100000.000000</td>
      <td>8746.865800</td>
      <td>1.450000</td>
      <td>2.250000</td>
      <td>1.990000</td>
      <td>2.150000</td>
      <td>11100.000000</td>
      <td>2.066251</td>
      <td>1.850000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# using iqr to spot outliers
def upper_limit(df, column):
    iqr = df['{}'.format(column)].quantile(q=0.75) - df['{}'.format(column)].quantile(q=0.25)
    upper_limit = df['{}'.format(column)].quantile(q=0.75) + 1.5*iqr
    return upper_limit
```


```python
_ = plt.figure(figsize=(25,15))
ax1 = plt.subplot(441)
total_cash_usage_upper = upper_limit(combine2, 'total_cash_usage')
plot_cont_var(combine2.query("total_cash_usage <= {}".format(total_cash_usage_upper)), 'total_cash_usage', boxplot=True, log=False, ax=ax1)

ax2 = plt.subplot(442)
total_retail_usage_upper = upper_limit(combine2, 'total_retail_usage')
plot_cont_var(combine2.query("total_retail_usage <= {}".format(total_retail_usage_upper)), 'total_retail_usage', boxplot=True, log=False, ax=ax2)

ax3 = plt.subplot(443)
total_usage_per_limit_upper = upper_limit(combine2, 'total_usage_per_limit')
plot_cont_var(combine2.query("total_usage_per_limit <= {}".format(total_usage_per_limit_upper)), 'total_usage_per_limit', boxplot=True, log=False, ax=ax3)

ax4 = plt.subplot(444)
total_3mo_usage_per_limit_upper = upper_limit(combine2, 'total_3mo_usage_per_limit')
plot_cont_var(combine2.query("total_3mo_usage_per_limit <= {}".format(total_3mo_usage_per_limit_upper)), 'total_3mo_usage_per_limit', boxplot=True, log=False, ax=ax4)

ax5 = plt.subplot(445)
total_6mo_usage_per_limit_upper = upper_limit(combine2, 'total_6mo_usage_per_limit')
plot_cont_var(combine2.query("total_6mo_usage_per_limit <= {}".format(total_6mo_usage_per_limit_upper)), 'total_6mo_usage_per_limit', boxplot=True, log=False, ax=ax5)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
```


![png](assignment2-3-4_files/assignment2-3-4_91_0.png)


Instead of using symlog scale, I have filtered out skewed data points to plot out the distribution of these usage features. The reason why I did not use symlog was because some of these features have too many zero values and the charts will become unreadable as well (similar to the first iteration).

The distribution of total_cash_usage is as such because the majority of total_cash_usage values (~95%) are zero.

We can see that defaulters tend to have lower usage and the first sign of financial trouble can be observed 3 months ago (total_3mo_usage_per_limit). GoFin could use this information to **target potential borrowers** and assist them in better financial planning to ensure that they do not default.

It is heartening to see that a majority of the users lowered their spending when they know they have financial issues. GoFin should **take note of the group of users who still spend as per normal despite having financial problems**. From the table below, we can see that the credit limit of those who are not reducing spend is higher than those who had reduced. This shows that their income level is probably higher as well. If GoFin still decides to loan out money to this group, GoFin should **charge them a higher interest**. Not only will it **dissuade them from spending above their current limits**, GoFin would also be able to **maximise profit as this group are more probable in making the high interest repayment**, with their assumed higher income.


```python
# defaulters who are not lowering spend
(
    combine2
    .query("default_flag == 1 and total_retail_usage > {}".format(combine2.total_retail_usage.median()))
    [['total_retail_usage', 'total_3mo_usage_per_limit', 'outstanding', 'credit_limit', 'credit_limit_per_card']]
    .describe()
)
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
      <th>total_retail_usage</th>
      <th>total_3mo_usage_per_limit</th>
      <th>outstanding</th>
      <th>credit_limit</th>
      <th>credit_limit_per_card</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
      <td>197.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>437.670124</td>
      <td>0.201633</td>
      <td>3328.763954</td>
      <td>3670.050761</td>
      <td>1082.765188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>989.133176</td>
      <td>0.201352</td>
      <td>7668.487610</td>
      <td>8335.531672</td>
      <td>1648.500297</td>
    </tr>
    <tr>
      <th>min</th>
      <td>20.860000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>49.086700</td>
      <td>0.066800</td>
      <td>558.516500</td>
      <td>700.000000</td>
      <td>250.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>111.831900</td>
      <td>0.142000</td>
      <td>1141.326400</td>
      <td>1500.000000</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>300.000000</td>
      <td>0.260000</td>
      <td>3513.588200</td>
      <td>4400.000000</td>
      <td>1233.333333</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8746.865800</td>
      <td>1.010000</td>
      <td>79805.857400</td>
      <td>100000.000000</td>
      <td>14285.714286</td>
    </tr>
  </tbody>
</table>
</div>




```python
# defaulters who are lowering spend
(
    combine2
    .query("default_flag == 1 and total_retail_usage < {}".format(combine2.total_retail_usage.median()))
    [['total_retail_usage', 'total_3mo_usage_per_limit', 'outstanding', 'credit_limit', 'credit_limit_per_card']]
    .describe()
)
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
      <th>total_retail_usage</th>
      <th>total_3mo_usage_per_limit</th>
      <th>outstanding</th>
      <th>credit_limit</th>
      <th>credit_limit_per_card</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1146.000000</td>
      <td>1146.000000</td>
      <td>1146.000000</td>
      <td>1146.000000</td>
      <td>1146.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.982736</td>
      <td>0.104357</td>
      <td>1414.302355</td>
      <td>2012.652705</td>
      <td>767.540410</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.926212</td>
      <td>0.161107</td>
      <td>2606.213205</td>
      <td>2481.048964</td>
      <td>892.279885</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.069000</td>
      <td>300.000000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>16.614200</td>
      <td>500.000000</td>
      <td>250.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.040954</td>
      <td>482.931050</td>
      <td>900.000000</td>
      <td>400.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>0.145750</td>
      <td>1357.141200</td>
      <td>2300.000000</td>
      <td>966.666667</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20.000000</td>
      <td>1.450000</td>
      <td>18827.608000</td>
      <td>18000.000000</td>
      <td>7700.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = plt.figure(figsize=(25,15))
start = 441
for idx, col in enumerate(['years_since_card_issuing', 'payment_ratio', 'payment_ratio_6month', 'utilization_6month']):
    _ = plt.subplot(start + idx)
    _ = sns.boxplot(x='default_flag', y='{}'.format(col), data=combine2)
    _ = plt.title("Boxplot of {}".format(col))

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
```


![png](assignment2-3-4_files/assignment2-3-4_95_0.png)


It is expected to see that the payment ratios of defaulters are lower than non-defaulters. The difference in payment ratio between defaulters and non-defaulters is more stark when a more recent metric (payment_ratio) is used. Defaulters also tend to have higher utilization rate.

GoFin could use payment ratios as a simple metric to gauge potential defaulters. For example, if the user would incur an interest of X% when taking a loan from GoFin, GoFin could refer to his/her latest payment_ratio to check whether is the person able to repay the interest rate. If the payment_ratio is below X%, it shows that the user would probably default.

Defaulters tend to have:
- higher outstanding amount 
- higher utilization rate 
- lower usage amount
- lower payment ratio 

Did Branch D and G managed to catch all these?


```python
# creating new column called branch_d_g where entries that are belongs to branch d or g will be 1, the rest 0
combine3= (
    combine2
    .pipe(lambda x: x.assign(branch_d_g=np.where(x.branch_code=='d', 1,
                                                np.where(x.branch_code=='g', 1, 0))))
)
```


```python
# branch D and G
combine3.query("branch_d_g == 1 and default_flag == 1")[['outstanding', 'utilization_6month', 'total_retail_usage', 'total_usage_per_limit', 'total_3mo_usage_per_limit', 'payment_ratio', 'payment_ratio_6month']].describe()
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
      <th>outstanding</th>
      <th>utilization_6month</th>
      <th>total_retail_usage</th>
      <th>total_usage_per_limit</th>
      <th>total_3mo_usage_per_limit</th>
      <th>payment_ratio</th>
      <th>payment_ratio_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>42.000000</td>
      <td>42.000000</td>
      <td>42.000000</td>
      <td>42.000000</td>
      <td>42.000000</td>
      <td>42.000000</td>
      <td>42.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1368.410893</td>
      <td>0.749592</td>
      <td>81.073640</td>
      <td>0.038587</td>
      <td>0.160424</td>
      <td>0.355102</td>
      <td>0.444440</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1898.741874</td>
      <td>0.435787</td>
      <td>293.351158</td>
      <td>0.096711</td>
      <td>0.199159</td>
      <td>0.286905</td>
      <td>0.342695</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.632300</td>
      <td>0.000499</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>385.614725</td>
      <td>0.560750</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.018925</td>
      <td>0.149750</td>
      <td>0.122025</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>612.887450</td>
      <td>0.849000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.107000</td>
      <td>0.272600</td>
      <td>0.346500</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1837.664175</td>
      <td>1.077500</td>
      <td>5.500000</td>
      <td>0.004064</td>
      <td>0.217000</td>
      <td>0.489500</td>
      <td>0.700000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9209.977600</td>
      <td>1.510000</td>
      <td>1420.018600</td>
      <td>0.478139</td>
      <td>0.946784</td>
      <td>1.030000</td>
      <td>1.000100</td>
    </tr>
  </tbody>
</table>
</div>




```python
# other branches
combine3.query("branch_d_g == 0 and default_flag == 1")[['outstanding', 'utilization_6month', 'total_retail_usage', 'total_usage_per_limit', 'total_3mo_usage_per_limit', 'payment_ratio', 'payment_ratio_6month']].describe()
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
      <th>outstanding</th>
      <th>utilization_6month</th>
      <th>total_retail_usage</th>
      <th>total_usage_per_limit</th>
      <th>total_3mo_usage_per_limit</th>
      <th>payment_ratio</th>
      <th>payment_ratio_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1301.000000</td>
      <td>1301.000000</td>
      <td>1301.000000</td>
      <td>1301.000000</td>
      <td>1301.000000</td>
      <td>1301.000000</td>
      <td>1301.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1705.675434</td>
      <td>0.633088</td>
      <td>64.521243</td>
      <td>0.031973</td>
      <td>0.117277</td>
      <td>0.314214</td>
      <td>0.420067</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3899.392187</td>
      <td>0.438773</td>
      <td>411.647962</td>
      <td>0.113184</td>
      <td>0.169964</td>
      <td>0.323456</td>
      <td>0.393309</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>40.518200</td>
      <td>0.163000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000322</td>
      <td>0.091500</td>
      <td>0.106000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>533.971900</td>
      <td>0.732000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.055300</td>
      <td>0.200000</td>
      <td>0.286000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1711.118500</td>
      <td>0.988000</td>
      <td>2.000000</td>
      <td>0.004000</td>
      <td>0.162000</td>
      <td>0.461000</td>
      <td>0.671000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>79805.857400</td>
      <td>1.850000</td>
      <td>8746.865800</td>
      <td>1.270260</td>
      <td>1.450000</td>
      <td>1.990000</td>
      <td>2.150000</td>
    </tr>
  </tbody>
</table>
</div>




```python
_ = plt.figure(figsize=(25,15))
ax1 = plt.subplot(441)
plot_cont_var(combine3.query("default_flag == 1"), 'outstanding', boxplot=True, log=True, ax=ax1, x='branch_d_g')

ax2 = plt.subplot(442)
_ = sns.boxplot(x='branch_d_g', y='utilization_6month', data=combine3.query("default_flag == 1"))
_ = plt.title("Boxplot of utilization_6month")

ax3 = plt.subplot(443)
total_retail_usage_upper = upper_limit(combine2.query("default_flag == 1"), 'total_retail_usage')
plot_cont_var(combine3.query("default_flag == 1 and total_retail_usage <= {}".format(total_retail_usage_upper)), 'total_retail_usage', boxplot=True, log=False, ax=ax3,  x='branch_d_g')

ax4 = plt.subplot(444)
total_usage_per_limit_upper = upper_limit(combine2.query("default_flag == 1"), 'total_usage_per_limit')
plot_cont_var(combine3.query("default_flag == 1 and total_usage_per_limit <= {}".format(total_usage_per_limit_upper)), 'total_usage_per_limit', boxplot=True, log=False, ax=ax4,  x='branch_d_g')

ax5 = plt.subplot(445)
total_3mo_usage_per_limit_upper = upper_limit(combine2, 'total_3mo_usage_per_limit')
plot_cont_var(combine3.query("default_flag == 1 and total_3mo_usage_per_limit <= {}".format(total_3mo_usage_per_limit_upper)), 'total_3mo_usage_per_limit', boxplot=True, log=False, ax=ax5,  x='branch_d_g')

ax6 = plt.subplot(446)
_ = sns.boxplot(x='branch_d_g', y='payment_ratio', data=combine3.query("default_flag == 1"))
_ = plt.title("Boxplot of payment_ratio")


ax7 = plt.subplot(447)
_ = sns.boxplot(x='branch_d_g', y='payment_ratio_6month', data=combine3.query("default_flag == 1"))
_ = plt.title("Boxplot of payment_ratio_6month")


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)

```


![png](assignment2-3-4_files/assignment2-3-4_101_0.png)


We can see that users who took loans from Branch D and G have relatively higher total_3mo_usage_per_limit and payment_ratio as compared to other branches. This resulted in them having lower default rates as compared to other branches.

# Model

## Summary
- Two models output were not great. Accuracy did not beat baseline accuracy of 90.3%
- However, the recall score of the first model is good
- When it comes to model evaluation, GoFin should be more concern with recall score as it tells us how many actual defaulters are selected
- GoFin needs to balance the cost of have too many False Positives vs False Negatives
- Payment ratio and usage level are the most important features to predict defaulters
- Branch D and G did a great job in noticing them
- Could have tried boosting models


```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, make_scorer

from sklearn.externals import joblib

from imblearn.over_sampling import SMOTE
```

    /Users/timong/anaconda/envs/py36/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



```python
combine2.head(2)
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
      <th>user_id</th>
      <th>branch_code</th>
      <th>default_flag</th>
      <th>number_of_cards</th>
      <th>outstanding</th>
      <th>credit_limit</th>
      <th>total_cash_usage</th>
      <th>total_retail_usage</th>
      <th>total_usage_per_limit</th>
      <th>total_3mo_usage_per_limit</th>
      <th>total_6mo_usage_per_limit</th>
      <th>payment_ratio</th>
      <th>payment_ratio_6month</th>
      <th>overlimit_percentage</th>
      <th>years_since_card_issuing</th>
      <th>remaining_bill_per_number_of_cards</th>
      <th>remaining_bill_per_limit</th>
      <th>utilization_6month</th>
      <th>credit_limit_per_card</th>
      <th>having_2_or_3_cards</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>i</td>
      <td>0</td>
      <td>2</td>
      <td>3.6158</td>
      <td>700.0</td>
      <td>0.0</td>
      <td>0.0094</td>
      <td>0.000013</td>
      <td>0.011719</td>
      <td>0.01781</td>
      <td>1.0219</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>15.416667</td>
      <td>1.31615</td>
      <td>0.00376</td>
      <td>0.021949</td>
      <td>350.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>0</td>
      <td>2</td>
      <td>26.8691</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>0.1012</td>
      <td>0.000101</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.0000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.750000</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.000300</td>
      <td>500.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
y = combine2.default_flag.values
```


```python
# dropping total_cash_usage because majority of the values are 0
# creating new feature called is_overlimit, where 1 = overlimit_percentage > 0 and 0 = overlimit_percentage = 0
X = (
    combine2
    .pipe(lambda x: x.assign(is_overlimit=np.where(x.overlimit_percentage>0, 1, 0)))
    .drop(['user_id', 'branch_code', 'default_flag', 'number_of_cards', 'total_cash_usage', 'overlimit_percentage', 'credit_limit'], axis=1)
)
```


```python
X.head()
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
      <th>outstanding</th>
      <th>total_retail_usage</th>
      <th>total_usage_per_limit</th>
      <th>total_3mo_usage_per_limit</th>
      <th>total_6mo_usage_per_limit</th>
      <th>payment_ratio</th>
      <th>payment_ratio_6month</th>
      <th>years_since_card_issuing</th>
      <th>remaining_bill_per_number_of_cards</th>
      <th>remaining_bill_per_limit</th>
      <th>utilization_6month</th>
      <th>credit_limit_per_card</th>
      <th>having_2_or_3_cards</th>
      <th>is_overlimit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.6158</td>
      <td>0.0094</td>
      <td>0.000013</td>
      <td>0.011719</td>
      <td>0.017810</td>
      <td>1.0219</td>
      <td>1.0000</td>
      <td>15.416667</td>
      <td>1.31615</td>
      <td>0.003760</td>
      <td>0.021949</td>
      <td>350.000000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>26.8691</td>
      <td>0.1012</td>
      <td>0.000101</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.750000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000300</td>
      <td>500.000000</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>676.9149</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.040518</td>
      <td>0.047703</td>
      <td>1.0000</td>
      <td>1.0091</td>
      <td>10.750000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.267853</td>
      <td>933.333333</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>349.6732</td>
      <td>253.6660</td>
      <td>0.120793</td>
      <td>0.055971</td>
      <td>0.016851</td>
      <td>1.0000</td>
      <td>0.2264</td>
      <td>19.750000</td>
      <td>14.53335</td>
      <td>0.027683</td>
      <td>0.346635</td>
      <td>525.000000</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>940.2085</td>
      <td>266.6558</td>
      <td>0.266656</td>
      <td>0.323027</td>
      <td>0.131162</td>
      <td>0.9599</td>
      <td>0.9984</td>
      <td>1.666667</td>
      <td>297.59325</td>
      <td>0.595186</td>
      <td>0.336571</td>
      <td>500.000000</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# checking for multi-collinearity amongst X matrix
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif
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
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.290368</td>
      <td>outstanding</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.575063</td>
      <td>total_retail_usage</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.348370</td>
      <td>total_usage_per_limit</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.389666</td>
      <td>total_3mo_usage_per_limit</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.617509</td>
      <td>total_6mo_usage_per_limit</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.447502</td>
      <td>payment_ratio</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.971756</td>
      <td>payment_ratio_6month</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2.853440</td>
      <td>years_since_card_issuing</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.214386</td>
      <td>remaining_bill_per_number_of_cards</td>
    </tr>
    <tr>
      <th>9</th>
      <td>6.408319</td>
      <td>remaining_bill_per_limit</td>
    </tr>
    <tr>
      <th>10</th>
      <td>6.296889</td>
      <td>utilization_6month</td>
    </tr>
    <tr>
      <th>11</th>
      <td>3.379923</td>
      <td>credit_limit_per_card</td>
    </tr>
    <tr>
      <th>12</th>
      <td>5.237289</td>
      <td>having_2_or_3_cards</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2.446488</td>
      <td>is_overlimit</td>
    </tr>
  </tbody>
</table>
</div>




```python
# getting training set (77%) and testing set (33%)
# stratified so that the split of the imbalance dataset between train and test set will be the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 3, stratify=y)
```


```python
# scale X matrix with StandardScaler
ss = StandardScaler()
Xss_train = ss.fit_transform(X_train)

Xss_test = ss.transform(X_test)

# SMOTE the training set as the data set is skewed towards having more non_defaulters
sm = SMOTE(random_state = 1, ratio = 'minority')
Xss_sm_train, y_sm_train = sm.fit_sample(Xss_train, y_train)
```

    /Users/timong/anaconda/envs/py36/lib/python3.6/site-packages/sklearn/preprocessing/data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    /Users/timong/anaconda/envs/py36/lib/python3.6/site-packages/sklearn/base.py:464: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)
    /Users/timong/anaconda/envs/py36/lib/python3.6/site-packages/ipykernel_launcher.py:5: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      """



```python
# using RandomForestClassifer as an estimator
rclf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)

# train 30% of the training set
rclf.fit(Xss_sm_train, y_sm_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=5, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=None,
                oob_score=False, random_state=1, verbose=0, warm_start=False)




```python
gs_params ={
    'criterion': ['gini'],
    'max_depth': [None,1,5,10],
    'max_features': ['auto',3,7],
    'n_estimators':[200, 500, 1000],
    'random_state':[1]
}

rclf_gs = GridSearchCV(RandomForestClassifier(), gs_params, n_jobs = -1, verbose = 1, cv = 3, scoring='recall')
```


```python
rclf_gs.fit(Xss_sm_train, y_sm_train)
```

    Fitting 3 folds for each of 36 candidates, totalling 108 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  3.7min
    [Parallel(n_jobs=-1)]: Done 108 out of 108 | elapsed:  8.1min finished





    GridSearchCV(cv=3, error_score='raise-deprecating',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators='warn', n_jobs=None,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False),
           fit_params=None, iid='warn', n_jobs=-1,
           param_grid={'criterion': ['gini'], 'max_depth': [None, 1, 5, 10], 'max_features': ['auto', 3, 7], 'n_estimators': [200, 500, 1000], 'random_state': [1]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring='recall', verbose=1)




```python
rclf_gs.best_params_
```




    {'criterion': 'gini',
     'max_depth': None,
     'max_features': 'auto',
     'n_estimators': 500,
     'random_state': 1}




```python
# using gs best parameters
rclf_gs_best = rclf_gs.best_estimator_
```


```python
# function to evaulate performance of model
def sup_perf(model, X_test, y_test):
    model_yhat = model.predict(X_test)
    model_score = model.score(X_test, y_test)
    model_f1 = f1_score(y_test, model_yhat, average = 'binary')
    model_precision = precision_score(y_test, model_yhat, average = 'binary')
    model_recall = recall_score(y_test, model_yhat, average = 'binary')
    model_auc = roc_auc_score(y_test, model_yhat, average = 'macro')
    
    return (model_score, model_precision, model_recall, model_f1, model_auc)
```


```python
score_rclf, precision_rclf, recall_rclf, f1_rclf, auc_rclf = sup_perf(rclf, Xss_test, y_test)
```


```python
score_rclf_gs_best, precision_rclf_gs_best, recall_rclf_gs_best, f1_rclf_gs_best, auc_rclf_gs_best = sup_perf(rclf_gs_best, Xss_test, y_test)
```


```python
rclf_yhat = rclf.predict(Xss_test)
print (classification_report(y_test, rclf_yhat, labels = [0,1], target_names=['non_default','default']))
```

                  precision    recall  f1-score   support
    
     non_default       0.97      0.67      0.80      4547
         default       0.20      0.82      0.32       443
    
       micro avg       0.69      0.69      0.69      4990
       macro avg       0.59      0.75      0.56      4990
    weighted avg       0.91      0.69      0.75      4990
    



```python
# confusion matrix based normal random forest
conmat = pd.DataFrame(confusion_matrix(y_test, rclf_yhat, labels = [0,1]),
                      index = ['actual_non_default','actual_default'],
                      columns = ['predict_non_default','predict_default'])
conmat
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
      <th>predict_non_default</th>
      <th>predict_default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>actual_non_default</th>
      <td>3064</td>
      <td>1483</td>
    </tr>
    <tr>
      <th>actual_default</th>
      <td>80</td>
      <td>363</td>
    </tr>
  </tbody>
</table>
</div>




```python
rclf_gs_best_yhat = rclf_gs_best.predict(Xss_test)
print (classification_report(y_test, rclf_gs_best_yhat, labels = [0,1], target_names=['non_default','default']))
```

                  precision    recall  f1-score   support
    
     non_default       0.94      0.91      0.93      4547
         default       0.32      0.42      0.36       443
    
       micro avg       0.87      0.87      0.87      4990
       macro avg       0.63      0.67      0.64      4990
    weighted avg       0.89      0.87      0.88      4990
    



```python
# confusion matrix based on random forest with grid search
conmat = pd.DataFrame(confusion_matrix(y_test, rclf_gs_best_yhat, labels = [0,1]),
                      index = ['actual_non_defau;t','actual_default'],
                      columns = ['predict_non_default','predict_default'])
conmat
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
      <th>predict_non_default</th>
      <th>predict_default</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>actual_non_defau;t</th>
      <td>4141</td>
      <td>406</td>
    </tr>
    <tr>
      <th>actual_default</th>
      <td>255</td>
      <td>188</td>
    </tr>
  </tbody>
</table>
</div>




```python
model_comparison = pd.DataFrame({'rclf':[score_rclf, precision_rclf, recall_rclf, f1_rclf, auc_rclf],\
                                     'rclf (grid_search)':[score_rclf_gs_best, precision_rclf_gs_best, recall_rclf_gs_best, f1_rclf_gs_best, auc_rclf_gs_best]},\
                                    index = ['accuracy score','precision score (fraud)','recall score (fraud)','f1 score (fraud)','auc score (unweighted mean)'])
model_comparison
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
      <th>rclf</th>
      <th>rclf (grid_search)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>accuracy score</th>
      <td>0.686774</td>
      <td>0.867535</td>
    </tr>
    <tr>
      <th>precision score (fraud)</th>
      <td>0.196641</td>
      <td>0.316498</td>
    </tr>
    <tr>
      <th>recall score (fraud)</th>
      <td>0.819413</td>
      <td>0.424379</td>
    </tr>
    <tr>
      <th>f1 score (fraud)</th>
      <td>0.317169</td>
      <td>0.362584</td>
    </tr>
    <tr>
      <th>auc score (unweighted mean)</th>
      <td>0.746632</td>
      <td>0.667545</td>
    </tr>
  </tbody>
</table>
</div>




```python
# observing feature importance of normal random forest model
feature_impt = pd.DataFrame({'importance': rclf.feature_importances_, 'features':X.columns})
feature_impt.sort_values('importance', ascending = False).reset_index(drop=True)
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
      <th>features</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>payment_ratio</td>
      <td>0.223744</td>
    </tr>
    <tr>
      <th>1</th>
      <td>total_retail_usage</td>
      <td>0.166121</td>
    </tr>
    <tr>
      <th>2</th>
      <td>total_usage_per_limit</td>
      <td>0.146446</td>
    </tr>
    <tr>
      <th>3</th>
      <td>remaining_bill_per_limit</td>
      <td>0.084147</td>
    </tr>
    <tr>
      <th>4</th>
      <td>payment_ratio_6month</td>
      <td>0.066992</td>
    </tr>
    <tr>
      <th>5</th>
      <td>remaining_bill_per_number_of_cards</td>
      <td>0.061405</td>
    </tr>
    <tr>
      <th>6</th>
      <td>total_3mo_usage_per_limit</td>
      <td>0.060965</td>
    </tr>
    <tr>
      <th>7</th>
      <td>is_overlimit</td>
      <td>0.060608</td>
    </tr>
    <tr>
      <th>8</th>
      <td>outstanding</td>
      <td>0.056963</td>
    </tr>
    <tr>
      <th>9</th>
      <td>utilization_6month</td>
      <td>0.026462</td>
    </tr>
    <tr>
      <th>10</th>
      <td>credit_limit_per_card</td>
      <td>0.024061</td>
    </tr>
    <tr>
      <th>11</th>
      <td>total_6mo_usage_per_limit</td>
      <td>0.014218</td>
    </tr>
    <tr>
      <th>12</th>
      <td>years_since_card_issuing</td>
      <td>0.006244</td>
    </tr>
    <tr>
      <th>13</th>
      <td>having_2_or_3_cards</td>
      <td>0.001624</td>
    </tr>
  </tbody>
</table>
</div>


