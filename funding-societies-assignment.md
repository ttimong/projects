

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline
pd.set_option('display.max_columns', 100)
```


```python
df = pd.read_csv('./loans_fs.csv', index_col=0)
```

    /Users/timong/anaconda/envs/py36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2728: DtypeWarning: Columns (19) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


At Funding Societies, we use various data sources to assess the suitability of a borrower on our platform. We predict the willingness and ability of a borrower to repay us. This is a dataset of personal loans and the variables utilised to make lending decisions. 

Your task would be to summarise the data and perform some initial analysis on it with a focus on the factors that drive lending decisions.

Feel free to use any software to process, analyse, model and present your findings. The idea here is to give you a realistic experience of the work we do. Please send across all workings including your code and any working files. 

What we want to look at is your thought process in approaching the problem and so, please explain how you are cleaning and dissecting the data and why you are using any particular methods. All the best!


loan_amnt: amount of loan applied for by the borrower

funded_amnt: total amount committed to that loan at that point of time

funded_amnt_inv: total amount committed by investors for that loan at that point in time

empy_title: the job title supplied by the Borrower when applying for the loan

verification_status: indicates if income was verified by LC, not verified, or if the income source was verified

issue_d: the month which the loan was funded

pymnt_plan: indicates if a payment plan has been put in place for the loan

dti: a ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.


```python
df.head(3)
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
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>term</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>pymnt_plan</th>
      <th>url</th>
      <th>desc</th>
      <th>purpose</th>
      <th>title</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1296599</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>4975.0</td>
      <td>36 months</td>
      <td>10.65</td>
      <td>162.87</td>
      <td>B</td>
      <td>B2</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>24000.0</td>
      <td>Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://www.lendingclub.com/browse/loanDetail....</td>
      <td>Borrower added on 12/22/11 &gt; I need to upgra...</td>
      <td>credit_card</td>
      <td>Computer</td>
      <td>860xx</td>
      <td>AZ</td>
      <td>27.65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1314167</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>2500.0</td>
      <td>60 months</td>
      <td>15.27</td>
      <td>59.83</td>
      <td>C</td>
      <td>C4</td>
      <td>Ryder</td>
      <td>&lt; 1 year</td>
      <td>RENT</td>
      <td>30000.0</td>
      <td>Source Verified</td>
      <td>Dec-2011</td>
      <td>Charged Off</td>
      <td>n</td>
      <td>https://www.lendingclub.com/browse/loanDetail....</td>
      <td>Borrower added on 12/22/11 &gt; I plan to use t...</td>
      <td>car</td>
      <td>bike</td>
      <td>309xx</td>
      <td>GA</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1313524</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>2400.0</td>
      <td>36 months</td>
      <td>15.96</td>
      <td>84.33</td>
      <td>C</td>
      <td>C5</td>
      <td>NaN</td>
      <td>10+ years</td>
      <td>RENT</td>
      <td>12252.0</td>
      <td>Not Verified</td>
      <td>Dec-2011</td>
      <td>Fully Paid</td>
      <td>n</td>
      <td>https://www.lendingclub.com/browse/loanDetail....</td>
      <td>NaN</td>
      <td>small_business</td>
      <td>real estate business</td>
      <td>606xx</td>
      <td>IL</td>
      <td>8.72</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.term.unique()
```




    array([' 36 months', ' 60 months', '600 months'], dtype=object)




```python
df.pymnt_plan.unique()
```




    array(['n', 'y'], dtype=object)




```python
df['term2'] = [int(i.split()[0]) for i in df.term]
df = (
    df
    .pipe(lambda x: x.assign(payment_plan = np.where(x.pymnt_plan=='n',1,0)))
    .drop(['term', 'pymnt_plan'], axis=1)
    .rename(columns={"term2": "term"})
)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 887379 entries, 0 to 887378
    Data columns (total 24 columns):
    member_id              887379 non-null int64
    loan_amnt              887379 non-null float64
    funded_amnt            887379 non-null float64
    funded_amnt_inv        887379 non-null float64
    int_rate               887379 non-null float64
    installment            887379 non-null float64
    grade                  887379 non-null object
    sub_grade              887379 non-null object
    emp_title              835917 non-null object
    emp_length             842554 non-null object
    home_ownership         887379 non-null object
    annual_inc             887375 non-null float64
    verification_status    887379 non-null object
    issue_d                887379 non-null object
    loan_status            887379 non-null object
    url                    887379 non-null object
    desc                   126028 non-null object
    purpose                887379 non-null object
    title                  887227 non-null object
    zip_code               887379 non-null object
    addr_state             887379 non-null object
    dti                    887379 non-null float64
    term                   887379 non-null int64
    payment_plan           887379 non-null int64
    dtypes: float64(7), int64(3), object(14)
    memory usage: 169.3+ MB



```python
df.loan_status.unique()
```




    array(['Fully Paid', 'Charged Off', 'Current', 'Default',
           'Late (31-120 days)', 'In Grace Period', 'Late (16-30 days)',
           'Does not meet the credit policy. Status:Fully Paid',
           'Does not meet the credit policy. Status:Charged Off'], dtype=object)




```python
df.loan_status.value_counts()
```




    Current                                                597271
    Fully Paid                                             206166
    Charged Off                                             44912
    Late (31-120 days)                                      11499
    In Grace Period                                          6213
    Late (16-30 days)                                        2337
    Does not meet the credit policy. Status:Fully Paid       1976
    Default                                                  1213
    Does not meet the credit policy. Status:Charged Off       757
    Name: loan_status, dtype: int64




```python
fully_paid = df.query("loan_status=='Fully Paid'")
effy = df.query("loan_status =='Does not meet the credit policy. Status:Fully Paid'")
```


```python
fully_paid.describe()
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
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.061660e+05</td>
      <td>206166.000000</td>
      <td>2.061660e+05</td>
      <td>206166.000000</td>
      <td>206166.000000</td>
      <td>206166.000000</td>
      <td>2.061660e+05</td>
      <td>2.061660e+05</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.386835e+07</td>
      <td>14161.680879</td>
      <td>1.480704e+04</td>
      <td>13219.147069</td>
      <td>13.331721</td>
      <td>413.139794</td>
      <td>7.696922e+04</td>
      <td>-3.125179e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.628014e+07</td>
      <td>15452.568592</td>
      <td>6.690401e+04</td>
      <td>8053.417853</td>
      <td>4.682235</td>
      <td>244.179712</td>
      <td>1.013761e+05</td>
      <td>8.355176e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.069900e+04</td>
      <td>500.000000</td>
      <td>-2.000000e+05</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.690000</td>
      <td>-5.000000e+04</td>
      <td>-2.500000e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.684134e+06</td>
      <td>7200.000000</td>
      <td>7.000000e+03</td>
      <td>7000.000000</td>
      <td>10.160000</td>
      <td>234.377500</td>
      <td>4.500000e+04</td>
      <td>1.035000e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.449496e+06</td>
      <td>12000.000000</td>
      <td>1.200000e+04</td>
      <td>11700.471902</td>
      <td>13.110000</td>
      <td>360.080000</td>
      <td>6.400000e+04</td>
      <td>1.571000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.922561e+07</td>
      <td>18000.000000</td>
      <td>1.800000e+04</td>
      <td>18000.000000</td>
      <td>16.000000</td>
      <td>540.545000</td>
      <td>9.000000e+04</td>
      <td>2.145000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.350742e+07</td>
      <td>250000.000000</td>
      <td>2.500000e+06</td>
      <td>35000.000000</td>
      <td>28.990000</td>
      <td>1409.990000</td>
      <td>7.141778e+06</td>
      <td>5.714000e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
effy.describe()
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
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1976.000000</td>
      <td>1976.000000</td>
      <td>1.976000e+03</td>
      <td>1976.000000</td>
      <td>1976.000000</td>
      <td>1976.000000</td>
      <td>1.972000e+03</td>
      <td>1.976000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>467930.401316</td>
      <td>9894.825405</td>
      <td>1.106607e+04</td>
      <td>6423.460524</td>
      <td>13.422743</td>
      <td>287.551225</td>
      <td>7.433065e+04</td>
      <td>-3.391823e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>213173.949263</td>
      <td>15614.561631</td>
      <td>7.951404e+04</td>
      <td>5944.471361</td>
      <td>3.660910</td>
      <td>204.293623</td>
      <td>1.051943e+05</td>
      <td>8.489865e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>70473.000000</td>
      <td>500.000000</td>
      <td>-5.000000e+04</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.670000</td>
      <td>-5.000000e+04</td>
      <td>-2.500000e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>288368.750000</td>
      <td>4075.000000</td>
      <td>4.000000e+03</td>
      <td>1975.000000</td>
      <td>11.860000</td>
      <td>135.990000</td>
      <td>3.800000e+04</td>
      <td>8.217500e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>459222.500000</td>
      <td>7200.000000</td>
      <td>7.000000e+03</td>
      <td>4800.000000</td>
      <td>13.850000</td>
      <td>230.495000</td>
      <td>5.700000e+04</td>
      <td>1.445000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>651029.250000</td>
      <td>12000.000000</td>
      <td>1.200000e+04</td>
      <td>9403.167645</td>
      <td>15.450000</td>
      <td>390.722500</td>
      <td>8.500000e+04</td>
      <td>2.000000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>822719.000000</td>
      <td>250000.000000</td>
      <td>2.500000e+06</td>
      <td>25000.000000</td>
      <td>27.700000</td>
      <td>940.140000</td>
      <td>2.500000e+06</td>
      <td>2.995000e+01</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
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
      <th>member_id</th>
      <th>loan_amnt</th>
      <th>funded_amnt</th>
      <th>funded_amnt_inv</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.807380e+05</td>
      <td>880738.000000</td>
      <td>8.807380e+05</td>
      <td>880738.000000</td>
      <td>880738.000000</td>
      <td>880738.000000</td>
      <td>8.807340e+05</td>
      <td>8.807380e+05</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.499909e+07</td>
      <td>15554.446527</td>
      <td>1.614215e+04</td>
      <td>14701.383153</td>
      <td>13.243193</td>
      <td>436.684319</td>
      <td>7.759617e+04</td>
      <td>-2.859317e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.411448e+07</td>
      <td>15515.969260</td>
      <td>6.519445e+04</td>
      <td>8440.832553</td>
      <td>4.772118</td>
      <td>244.152557</td>
      <td>1.022653e+05</td>
      <td>7.973133e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.047300e+04</td>
      <td>500.000000</td>
      <td>-2.000000e+05</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.670000</td>
      <td>-5.000000e+04</td>
      <td>-2.500000e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.086997e+07</td>
      <td>8000.000000</td>
      <td>8.000000e+03</td>
      <td>8000.000000</td>
      <td>9.990000</td>
      <td>260.730000</td>
      <td>4.500000e+04</td>
      <td>1.184000e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.708669e+07</td>
      <td>13000.000000</td>
      <td>1.300000e+04</td>
      <td>13000.000000</td>
      <td>12.990000</td>
      <td>382.550000</td>
      <td>6.468000e+04</td>
      <td>1.761000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.847161e+07</td>
      <td>20000.000000</td>
      <td>2.000000e+04</td>
      <td>20000.000000</td>
      <td>16.290000</td>
      <td>572.427500</td>
      <td>9.000000e+04</td>
      <td>2.392000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.354484e+07</td>
      <td>250000.000000</td>
      <td>2.500000e+06</td>
      <td>35000.000000</td>
      <td>28.990000</td>
      <td>1445.460000</td>
      <td>9.500000e+06</td>
      <td>9.999000e+03</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df.query("loan_amnt>=0 and loan_status!='Issued' ")
```


```python
_ = plt.figure(figsize=(20,6))
_ = plt.subplot(121)
_ = sns.distplot(df.loan_amnt)
_ = plt.title('Frequency distribution')

_ = plt.subplot(122)
_ = sns.boxplot(x='loan_amnt', data=df, orient='v')
_ = plt.title('Boxplot')
```


![png](funding-societies-assignment_files/funding-societies-assignment_16_0.png)



```python

```
