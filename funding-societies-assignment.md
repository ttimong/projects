

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from scipy import stats

%matplotlib inline
pd.set_option('display.max_columns', 100)
```


```python
df = pd.read_csv('./loans_fs.csv', index_col=0)
```

    /Users/timong/anaconda/envs/py36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2728: DtypeWarning: Columns (19) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)


1. Predit willingess and ability of a borrower to repay us

2. Summarise the data and perform intial analysis on the focus that drive lending decisions (WHO SHOULD WE LEND MONEY TO?)

dti: a ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.

term: 36 months, 60 months, 600 months

payment_plan: y, n

grade: A, B, C, D, E, F, G

sub_grade: 1 to 5

emp_length: < 1 year to 10+ years increment by 1 year

home_ownership: rent, own, mortgage, other, none, any

verification_status: verified, source verified, not verified

loan_status: fully paid, does not meet credit policy. fully paid, current, in grace period, late (16-30 days), late (31-120 days), default, charged off, does not meet credit policy. charged off

purpose: credit card, car, small business, wedding, debt consolidation, home improvement, major purchase, medical, moving, vacation, house, renewable energy, educational, other


```python
df.head(1)
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
  </tbody>
</table>
</div>




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
    term                   887379 non-null object
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
    pymnt_plan             887379 non-null object
    url                    887379 non-null object
    desc                   126028 non-null object
    purpose                887379 non-null object
    title                  887227 non-null object
    zip_code               887379 non-null object
    addr_state             887379 non-null object
    dti                    887379 non-null float64
    dtypes: float64(7), int64(1), object(16)
    memory usage: 169.3+ MB



```python
# converting terms in int
df['term2'] = [int(i.split()[0]) for i in df.term]

# changing payment_plan to 1 (y) and 0 (n) and filling na with 999
df = (
    df
    .pipe(lambda x: x.assign(payment_plan = np.where(x.pymnt_plan=='y',1,0)))
    .drop(['term', 'pymnt_plan'], axis=1)
    .rename(columns={"term2": "term"})
    .fillna(999)
)
```


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
      <th>term</th>
      <th>payment_plan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.873790e+05</td>
      <td>887379.000000</td>
      <td>8.873790e+05</td>
      <td>887379.000000</td>
      <td>887379.000000</td>
      <td>887379.000000</td>
      <td>8.873790e+05</td>
      <td>8.873790e+05</td>
      <td>887379.000000</td>
      <td>887379.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.500182e+07</td>
      <td>15177.429402</td>
      <td>1.890163e+04</td>
      <td>14702.464383</td>
      <td>13.243805</td>
      <td>436.717127</td>
      <td>8.237772e+04</td>
      <td>-8.562732e+03</td>
      <td>45.641459</td>
      <td>0.000011</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.411335e+07</td>
      <td>16084.862871</td>
      <td>1.115294e+05</td>
      <td>8442.106732</td>
      <td>4.771725</td>
      <td>244.186593</td>
      <td>1.488760e+05</td>
      <td>1.375036e+05</td>
      <td>38.337477</td>
      <td>0.003357</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.047300e+04</td>
      <td>-50000.000000</td>
      <td>-2.000000e+05</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.670000</td>
      <td>-5.000000e+04</td>
      <td>-2.500000e+06</td>
      <td>36.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.087713e+07</td>
      <td>8000.000000</td>
      <td>8.000000e+03</td>
      <td>8000.000000</td>
      <td>9.990000</td>
      <td>260.705000</td>
      <td>4.500000e+04</td>
      <td>1.169000e+01</td>
      <td>36.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.709528e+07</td>
      <td>13000.000000</td>
      <td>1.290000e+04</td>
      <td>13000.000000</td>
      <td>12.990000</td>
      <td>382.550000</td>
      <td>6.400000e+04</td>
      <td>1.752000e+01</td>
      <td>36.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.847135e+07</td>
      <td>20000.000000</td>
      <td>2.000000e+04</td>
      <td>20000.000000</td>
      <td>16.290000</td>
      <td>572.600000</td>
      <td>9.000000e+04</td>
      <td>2.386000e+01</td>
      <td>60.000000</td>
      <td>0.000000</td>
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
      <td>600.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# assumptions
# funded_amnt, loan_amnt, annual_inc, dti cannot be less than 0. multiply them by -1
df_edit = (
    df
    .pipe(lambda x: x.assign(funded_amnt_edit = np.where(x.funded_amnt<0, x.funded_amnt*-1, x.funded_amnt)))
    .pipe(lambda x: x.assign(loan_amnt_edit = np.where(x.loan_amnt<0, x.loan_amnt*-1, x.loan_amnt)))
    .pipe(lambda x: x.assign(annual_inc_edit = np.where(x.annual_inc<0, x.annual_inc*-1, x.annual_inc)))
    .pipe(lambda x: x.assign(dti_edit = np.where(x.dti<0, x.dti*-1, x.dti)))
    .drop(["funded_amnt", "loan_amnt", "annual_inc", "dti"], axis=1)
    .rename(columns={"funded_amnt_edit": "funded_amnt", "loan_amnt_edit": "loan_amnt", "annual_inc_edit": "annual_inc", "dti_edit": "dti"})
    [[
        'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 
        'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
        'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
        'loan_status', 'url', 'desc', 'purpose', 'title', 'zip_code',
        'addr_state', 'dti', 'term', 'payment_plan']]
)
```


```python
# assumptions 
# funded_amnt or funded_amnt_inv cannot be more than loan_amnt
# funded_amnt_inv cannot be more than funded_amnt

# checking out how many rows will be removed
count = df_edit.query("funded_amnt_inv > loan_amnt or funded_amnt > loan_amnt or funded_amnt_inv > funded_amnt").shape[0]
print("erroneous rows: " + str(count))

print("number of rows in raw data: " + str(df_edit.shape[0]))
```

    erroneous rows: 4991
    number of rows in raw data: 887379



```python
# since assumed erroneous rows is only approx ~0.5% of raw data, I will remove them
df_edit = df_edit.query("funded_amnt_inv <= loan_amnt and funded_amnt <= loan_amnt and funded_amnt_inv <= funded_amnt")
```


```python
df_edit.describe()
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
      <th>term</th>
      <th>payment_plan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8.823880e+05</td>
      <td>882388.000000</td>
      <td>882388.000000</td>
      <td>882388.000000</td>
      <td>882388.000000</td>
      <td>882388.000000</td>
      <td>8.823880e+05</td>
      <td>8.823880e+05</td>
      <td>882388.000000</td>
      <td>882388.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.499902e+07</td>
      <td>15446.178014</td>
      <td>14955.470666</td>
      <td>14693.755111</td>
      <td>13.242777</td>
      <td>436.472471</td>
      <td>7.898817e+04</td>
      <td>4.299142e+03</td>
      <td>45.643554</td>
      <td>0.000011</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.411495e+07</td>
      <td>14319.637392</td>
      <td>9769.882116</td>
      <td>8436.780490</td>
      <td>4.771752</td>
      <td>244.024472</td>
      <td>1.150225e+05</td>
      <td>9.723190e+04</td>
      <td>38.372779</td>
      <td>0.003366</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.047300e+04</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.670000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>36.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.086971e+07</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>9.990000</td>
      <td>260.550000</td>
      <td>4.500000e+04</td>
      <td>1.195000e+01</td>
      <td>36.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.708695e+07</td>
      <td>13000.000000</td>
      <td>13000.000000</td>
      <td>13000.000000</td>
      <td>12.990000</td>
      <td>382.550000</td>
      <td>6.450000e+04</td>
      <td>1.772000e+01</td>
      <td>36.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.847158e+07</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>16.290000</td>
      <td>571.870000</td>
      <td>9.000000e+04</td>
      <td>2.407000e+01</td>
      <td>60.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.354484e+07</td>
      <td>250000.000000</td>
      <td>200000.000000</td>
      <td>35000.000000</td>
      <td>28.990000</td>
      <td>1445.460000</td>
      <td>9.500000e+06</td>
      <td>2.500000e+06</td>
      <td>600.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# seems there are some erroneous entry in dti
# will use IQR to remove outliers
upper_limit = np.percentile(df_edit.dti, 75) + (stats.iqr(df_edit.dti) * 1.5)
lower_limit = np.percentile(df_edit.dti, 25) - (stats.iqr(df_edit.dti) * 1.5)

df_edit2 = df_edit.query("dti >= {lower} and dti <= {upper}".format(lower=lower_limit, upper=upper_limit))
```


```python
_ = plt.figure(figsize=(20,5))
_ = plt.subplot(131)
_ = sns.distplot(df_edit.loan_amnt)
_ = plt.title('frequency distribution of loan amount')

_ = plt.subplot(132)
_ = sns.distplot(df_edit.funded_amnt)
_ = plt.title('frequency distribution of funded amount')

_ = plt.subplot(133)
_ = sns.distplot(df_edit.funded_amnt_inv)
_ = plt.title('frequency distribution of funded amount by investors')

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
```


![png](funding-societies-assignment_files/funding-societies-assignment_14_0.png)



```python

```
