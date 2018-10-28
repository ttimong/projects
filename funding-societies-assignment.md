

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

# visualisation tools
import matplotlib.pyplot as plt 
import seaborn as sns
from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
offline.init_notebook_mode()

# modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, make_scorer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, GridSearchCV

%matplotlib inline
pd.set_option('display.max_columns', 100)
```

    IOPub data rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_data_rate_limit`.



```python
df = pd.read_csv('./loans_fs.csv', index_col=0)
```

    /Users/timong/anaconda/envs/py36/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2728: DtypeWarning:
    
    Columns (19) have mixed types. Specify dtype option on import or set low_memory=False.
    


# Objective

The goal of this project is to determine which features would be useful in predicting borrowers who will repay their loan in time

# Key Takeways

1. Most important features to look at in predicting good borrowers are **sub-grade** and **annual income**
2. It was quite difficult to determine the important features with just the use of EDA. As such, I carried on to use machine learning models to further evaluate the importance of each feature

# Data Cleaning


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
      <td>8.873790e+05</td>
      <td>887379.000000</td>
      <td>8.873790e+05</td>
      <td>887379.000000</td>
      <td>887379.000000</td>
      <td>887379.000000</td>
      <td>8.873750e+05</td>
      <td>8.873790e+05</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.500182e+07</td>
      <td>15177.429402</td>
      <td>1.890163e+04</td>
      <td>14702.464383</td>
      <td>13.243805</td>
      <td>436.717127</td>
      <td>8.237809e+04</td>
      <td>-8.562732e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.411335e+07</td>
      <td>16084.862871</td>
      <td>1.115294e+05</td>
      <td>8442.106732</td>
      <td>4.771725</td>
      <td>244.186593</td>
      <td>1.488763e+05</td>
      <td>1.375036e+05</td>
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



If we were to refer to the data description above, we can see that there are entries where funded amount, loan amount, annual income and debt-to-ratio are negative amount

**Assumption 1: **It is impossible for a funded amount, loan amount, annual income and debt-to-ratio to be negative. I will assume that is a data entry issue and will multipy affected entry by -1


```python
# multiplying negative funded_amnt, loan_amnt, annual_inc, dti by -1
df_clean = (
    df
    .pipe(lambda x: x.assign(funded_amnt = np.where(x.funded_amnt<0, x.funded_amnt*-1, x.funded_amnt)))
    .pipe(lambda x: x.assign(loan_amnt = np.where(x.loan_amnt<0, x.loan_amnt*-1, x.loan_amnt)))
    .pipe(lambda x: x.assign(annual_inc = np.where(x.annual_inc<0, x.annual_inc*-1, x.annual_inc)))
    .pipe(lambda x: x.assign(dti = np.where(x.dti<0, x.dti*-1, x.dti)))
    [[
        'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate', 
        'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
        'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
        'loan_status', 'url', 'desc', 'purpose', 'title', 'zip_code',
        'addr_state', 'dti', 'term', 'pymnt_plan']]
)
```


```python
df_clean.dti.describe()
```




    count    8.873790e+05
    mean     8.598638e+03
    std      1.375014e+05
    min      0.000000e+00
    25%      1.198000e+01
    50%      1.778000e+01
    75%      2.420000e+01
    max      2.500000e+06
    Name: dti, dtype: float64



The standard deviation of dti is very large. This could mean that there are outliers in the data. 

**Assumption 2:** Debt-to-income ratio cannot be more or less than IQR +- 1.5 of itself


```python
df_clean.query("funded_amnt_inv > loan_amnt or funded_amnt > loan_amnt or funded_amnt_inv > funded_amnt").shape
```




    (4991, 24)



There exists entries where: 

1. Funded amount by investors is more than loan amount
2. Total funded amount is more than loan amount
3. Funded amount by investors is more than total funded amount

**Assumption 3:** Total funded amount is capped at loan amount request and funded amount by investor is capped at total funded amount


```python
# calculating upper and lower limit of dti
# dti cannot be less than 0 hence, min(0, value)
upper_limit_dti = np.percentile(df_clean.dti, 75) + (stats.iqr(df_clean.dti) * 1.5)
lower_limit_dti = min(0, np.percentile(df_clean.dti, 25) - (stats.iqr(df_clean.dti) * 1.5))

# checking out how many rows will be removed due to erroneous data
erroneous_row_count = (
    df_clean
    .query("funded_amnt_inv > loan_amnt or funded_amnt > loan_amnt or funded_amnt_inv > funded_amnt or dti < {lower} or dti >= {upper}".format(lower=lower_limit_dti, upper=upper_limit_dti))
    .shape[0]
    )

print("erroneous rows: " + str(erroneous_row_count))
print("number of rows in raw data: " + str(df_clean.shape[0]))
```

    erroneous rows: 10073
    number of rows in raw data: 887379



```python
# since assumed erroneous rows is only approx ~1.1% of raw data, I will remove them
df_clean = df_clean.query("funded_amnt_inv <= loan_amnt and funded_amnt <= loan_amnt and funded_amnt_inv <= funded_amnt and dti >= {lower} and dti <= {upper}".format(lower=lower_limit_dti, upper=upper_limit_dti))
```


```python
df_clean.describe()
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
      <td>8.773060e+05</td>
      <td>877306.000000</td>
      <td>877306.000000</td>
      <td>877306.000000</td>
      <td>877306.000000</td>
      <td>877306.000000</td>
      <td>8.773020e+05</td>
      <td>877306.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.499747e+07</td>
      <td>14753.444864</td>
      <td>14740.069514</td>
      <td>14700.564455</td>
      <td>13.242085</td>
      <td>436.652712</td>
      <td>7.503542e+04</td>
      <td>18.129166</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.411417e+07</td>
      <td>8433.803422</td>
      <td>8428.333243</td>
      <td>8440.571437</td>
      <td>4.771488</td>
      <td>244.142474</td>
      <td>6.485735e+04</td>
      <td>8.299734</td>
    </tr>
    <tr>
      <th>min</th>
      <td>7.047300e+04</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>15.670000</td>
      <td>1.896000e+03</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.086953e+07</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>8000.000000</td>
      <td>9.990000</td>
      <td>260.730000</td>
      <td>4.500000e+04</td>
      <td>11.910000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.708661e+07</td>
      <td>13000.000000</td>
      <td>13000.000000</td>
      <td>13000.000000</td>
      <td>12.990000</td>
      <td>382.550000</td>
      <td>6.500000e+04</td>
      <td>17.650000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.847134e+07</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>20000.000000</td>
      <td>16.290000</td>
      <td>572.300000</td>
      <td>9.000000e+04</td>
      <td>23.950000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.354484e+07</td>
      <td>35000.000000</td>
      <td>35000.000000</td>
      <td>35000.000000</td>
      <td>28.990000</td>
      <td>1445.460000</td>
      <td>9.500000e+06</td>
      <td>42.370000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 877306 entries, 0 to 887378
    Data columns (total 24 columns):
    member_id              877306 non-null int64
    loan_amnt              877306 non-null float64
    funded_amnt            877306 non-null float64
    funded_amnt_inv        877306 non-null float64
    int_rate               877306 non-null float64
    installment            877306 non-null float64
    grade                  877306 non-null object
    sub_grade              877306 non-null object
    emp_title              826424 non-null object
    emp_length             832990 non-null object
    home_ownership         877306 non-null object
    annual_inc             877302 non-null float64
    verification_status    877306 non-null object
    issue_d                877306 non-null object
    loan_status            877306 non-null object
    url                    877306 non-null object
    desc                   124657 non-null object
    purpose                877306 non-null object
    title                  877157 non-null object
    zip_code               877306 non-null object
    addr_state             877306 non-null object
    dti                    877306 non-null float64
    term                   877306 non-null object
    pymnt_plan             877306 non-null object
    dtypes: float64(7), int64(1), object(16)
    memory usage: 167.3+ MB



```python
df_clean.query("annual_inc != annual_inc")
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
      <th>grade</th>
      <th>sub_grade</th>
      <th>emp_title</th>
      <th>emp_length</th>
      <th>home_ownership</th>
      <th>annual_inc</th>
      <th>verification_status</th>
      <th>issue_d</th>
      <th>loan_status</th>
      <th>url</th>
      <th>desc</th>
      <th>purpose</th>
      <th>title</th>
      <th>zip_code</th>
      <th>addr_state</th>
      <th>dti</th>
      <th>term</th>
      <th>pymnt_plan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42449</th>
      <td>79912</td>
      <td>5000.0</td>
      <td>5000.0</td>
      <td>3025.0</td>
      <td>7.43</td>
      <td>155.38</td>
      <td>A</td>
      <td>A2</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>NONE</td>
      <td>NaN</td>
      <td>Not Verified</td>
      <td>Aug-2007</td>
      <td>Does not meet the credit policy. Status:Fully ...</td>
      <td>https://www.lendingclub.com/browse/loanDetail....</td>
      <td>I will be relocating to Boston next month and ...</td>
      <td>other</td>
      <td>Moving expenses</td>
      <td>100xx</td>
      <td>NY</td>
      <td>1.0</td>
      <td>36 months</td>
      <td>n</td>
    </tr>
    <tr>
      <th>42450</th>
      <td>79906</td>
      <td>7000.0</td>
      <td>7000.0</td>
      <td>3450.0</td>
      <td>7.75</td>
      <td>218.55</td>
      <td>A</td>
      <td>A3</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>NONE</td>
      <td>NaN</td>
      <td>Not Verified</td>
      <td>Aug-2007</td>
      <td>Does not meet the credit policy. Status:Fully ...</td>
      <td>https://www.lendingclub.com/browse/loanDetail....</td>
      <td>I am borrowing $7,000 for tuition and other ex...</td>
      <td>other</td>
      <td>Education</td>
      <td>100xx</td>
      <td>NY</td>
      <td>1.0</td>
      <td>36 months</td>
      <td>n</td>
    </tr>
    <tr>
      <th>42480</th>
      <td>79878</td>
      <td>6700.0</td>
      <td>6700.0</td>
      <td>6700.0</td>
      <td>7.75</td>
      <td>209.18</td>
      <td>A</td>
      <td>A3</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>NONE</td>
      <td>NaN</td>
      <td>Not Verified</td>
      <td>Jul-2007</td>
      <td>Does not meet the credit policy. Status:Fully ...</td>
      <td>https://www.lendingclub.com/browse/loanDetail....</td>
      <td>I am moving to Florida and would like to borro...</td>
      <td>other</td>
      <td>Moving expenses and security deposit</td>
      <td>100xx</td>
      <td>NY</td>
      <td>1.0</td>
      <td>36 months</td>
      <td>n</td>
    </tr>
    <tr>
      <th>42533</th>
      <td>70735</td>
      <td>6500.0</td>
      <td>6500.0</td>
      <td>0.0</td>
      <td>8.38</td>
      <td>204.84</td>
      <td>A</td>
      <td>A5</td>
      <td>NaN</td>
      <td>&lt; 1 year</td>
      <td>NONE</td>
      <td>NaN</td>
      <td>Not Verified</td>
      <td>Jun-2007</td>
      <td>Does not meet the credit policy. Status:Fully ...</td>
      <td>https://www.lendingclub.com/browse/loanDetail....</td>
      <td>Hi,   I'm buying  a used car. Anybody on faceb...</td>
      <td>other</td>
      <td>Buying a car</td>
      <td>100xx</td>
      <td>NY</td>
      <td>4.0</td>
      <td>36 months</td>
      <td>n</td>
    </tr>
  </tbody>
</table>
</div>



There are missing values in annual_inc and they are all categorised under grade A.

I will impute the null values with the median income of loans that are classified under grade A.


```python
median = df_clean.query("grade == 'A'")[['annual_inc']].median()
print (median)
```

    annual_inc    75000.0
    dtype: float64



```python
df_clean['annual_inc'].fillna(75000.0, inplace=True)

# replacing other null values as 'blank'
df_clean.fillna('blank', inplace=True)
```

# Data Transformation

Since we are interested in analysing borrowers who are willing to make good on their loans, I will remove entries where loan_status = 'Issued' because it does not inform us whether the borrower will make good on his/her loan.

I will classified all loan_status that are "Fully Paid" or "Current" as borrowers who are willing to make payment, i.e. good loans and the rest as borrowers who are unable/unwilling to make payment, i.e. bad loans borrowers. 


```python
# removing loan status "Issued" because it cannot be determine whether borrowers willing to make payment
# obtaining year where loan was issued
# replace blank space with _ for each cell
# good loans = 1 and bad loans = 0 under a new column 'target'
df_transform = (
    df_clean
    .query("loan_status != 'Issued'")
    .pipe(lambda x: x.assign(year = pd.to_datetime(x.issue_d).dt.year))
    .pipe(lambda x: x.assign(verification_status = x.verification_status.str.replace(' ','_')))
    .pipe(lambda x: x.assign(term = x.term.str.replace(' ','_'))).pipe(lambda x: x.assign(
            target = np.where(x.loan_status == 'Fully Paid',1,
                              np.where(x.loan_status == 'Current',1,
                                       np.where(x.loan_status == 'Does not meet the credit policy. Status:Fully Paid',1,0)))))
    )
```


```python
# plotting a correlation heatmap between continuous features
cont_feature_df = df_transform[['loan_amnt', 'int_rate', 'funded_amnt', 'funded_amnt_inv', 'installment', 'annual_inc', 'dti']]

_ = plt.subplots(figsize = (12,7))

corr = cont_feature_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(600, 20, as_cmap=True)

_ = sns.heatmap(corr, mask=mask, annot=True, cmap =cmap)
```


![png](funding-societies-assignment_files/funding-societies-assignment_23_0.png)


Since loan amount is perfectly correlated to funded amount, funded amount by investors and highly correlated to installment, I would be removing funded_amnt, funded_amnt_inv and installment from my analysis as they are adequately represened by loan_amnt


```python
# plotting frequency distribution graphs of continuous features, group by good loan and bad loans
_ = plt.figure(figsize=(20,10))
ax1 = plt.subplot(221)
_ = sns.distplot(df_transform.query("target == 1").loan_amnt, ax=ax1, label='good loans')
_ = sns.distplot(df_transform.query("target == 0").loan_amnt, ax=ax1, label='bad loans')
_ = plt.title("Frequency distribution of loan amount")
_ = ax1.legend()

ax2 = plt.subplot(222)
_ = sns.distplot(df_transform.query("target == 1").int_rate, ax=ax2, label='good loans')
_ = sns.distplot(df_transform.query("target == 0").int_rate, ax=ax2, label='bad loans')
_ = plt.title("Frequency distribution of interest rate")
_ = ax2.legend()

ax3 = plt.subplot(223)
_ = sns.distplot(np.log(df_transform.query("target == 1").annual_inc), ax=ax3, label='good loans')
_ = sns.distplot(np.log(df_transform.query("target == 0").annual_inc), ax=ax3, label='bad loans')
_ = plt.title("Frequency distribution of annual income")
_ = ax3.legend()

ax4 = plt.subplot(224)
_ = sns.distplot(df_transform.query("target == 1").dti, ax=ax4, label='good loans')
_ = sns.distplot(df_transform.query("target == 0").dti, ax=ax4, label='bad loans')
_ = plt.title("Frequency distribution of dti")
_ = ax4.legend()

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
```


![png](funding-societies-assignment_files/funding-societies-assignment_25_0.png)


From the frequency distribution charts above, we can see that **interest rate**, **annual income** and **debt-to-income ratio**  have some form of relationship with loan status


```python
cont_feature_df = df_transform[['loan_amnt', 'int_rate', 'annual_inc', 'dti']]

# checking for multi-collinearity amongst continuous features
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(cont_feature_df.values, i) for i in range(cont_feature_df.shape[1])]
vif["features"] = cont_feature_df.columns
vif[['features','VIF Factor']]
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
      <th>VIF Factor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>loan_amnt</td>
      <td>4.447659</td>
    </tr>
    <tr>
      <th>1</th>
      <td>int_rate</td>
      <td>5.526292</td>
    </tr>
    <tr>
      <th>2</th>
      <td>annual_inc</td>
      <td>2.485083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dti</td>
      <td>4.526477</td>
    </tr>
  </tbody>
</table>
</div>



We can see that the VIF factor of the selected continuous features is less than 10, meaning to say that the multi-collinearity relationship amongst them is low


```python
def perf_loans_cat_features(df, column):
    """
    creating dataframe that display the proportion of good/bad loans of each category
    """
    dictionary = {}
    for i in df[column].unique():
        dictionary[i] = round(len(df.query("{}=='{}' and target==1".format(column,i)))/len(df.query("{}=='{}'".format(column,i))) * 100,1)
    column_df = (
        pd.DataFrame(list(dictionary.items()), columns=[column, 'pctg_good_loans'])
        .pipe(lambda x: x.assign(total=100))
        .pipe(lambda x: x.assign(pctg_bad_loans=x.total-x.pctg_good_loans))
        .sort_values(column)
        .reset_index()
        [[column, 'pctg_good_loans', 'pctg_bad_loans', 'total']]
    )
    
    return column_df
```


```python
grades_df = perf_loans_cat_features(df_transform, 'grade')
sub_grades_df = perf_loans_cat_features(df_transform, 'sub_grade')
emp_length_df = perf_loans_cat_features(df_transform, 'emp_length')
home_ownership_df = perf_loans_cat_features(df_transform, 'home_ownership')
verification_status_df = perf_loans_cat_features(df_transform, 'verification_status')
term_df = perf_loans_cat_features(df_transform, 'term')
payment_plan_df = perf_loans_cat_features(df_transform, 'pymnt_plan')
purpose_df = perf_loans_cat_features(df_transform, 'purpose')
state_df = perf_loans_cat_features(df_transform, 'addr_state')
year_df = perf_loans_cat_features(df_transform, 'year')
```


```python
# example of the output of perf_loans_cat_features function
grades_df
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
      <th>grade</th>
      <th>pctg_good_loans</th>
      <th>pctg_bad_loans</th>
      <th>total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>97.5</td>
      <td>2.5</td>
      <td>100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>B</td>
      <td>94.7</td>
      <td>5.3</td>
      <td>100</td>
    </tr>
    <tr>
      <th>2</th>
      <td>C</td>
      <td>92.2</td>
      <td>7.8</td>
      <td>100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>D</td>
      <td>88.5</td>
      <td>11.5</td>
      <td>100</td>
    </tr>
    <tr>
      <th>4</th>
      <td>E</td>
      <td>86.1</td>
      <td>13.9</td>
      <td>100</td>
    </tr>
    <tr>
      <th>5</th>
      <td>F</td>
      <td>80.8</td>
      <td>19.2</td>
      <td>100</td>
    </tr>
    <tr>
      <th>6</th>
      <td>G</td>
      <td>76.7</td>
      <td>23.3</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>




```python
cat_feature_df = [grades_df, sub_grades_df, emp_length_df, home_ownership_df, verification_status_df, term_df, payment_plan_df, purpose_df, year_df]
cat_feature_list = ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'term', 'pymnt_plan', 'purpose', 'year']
```


```python
# plotting a percentage stacked bar chart of all the categorical features to visualise the proportion of bad loans and good loans
_ = plt.figure(figsize=(20,10))
start = 331
for idx, (df, col )in enumerate(zip(cat_feature_df, cat_feature_list)):
    _ = plt.subplot(start + idx)
    overall_plot = sns.barplot(x = df[col], y = df.total, color = "red", label='bad loans')
    bottom_plot = sns.barplot(x = df[col], y = df.pctg_good_loans, color = "green", label='good loans')
    _ = plt.title('Breakdown of loan status by ' + col)
    _ = plt.legend(loc=4, ncol = 1)
    _ = plt.ylabel('percentage')
    _ = plt.xticks(rotation=75, fontsize='10')
    
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=1, wspace=0.35)
```


![png](funding-societies-assignment_files/funding-societies-assignment_33_0.png)


From the percentage stacked bar charts above, we can see that **all of the categorical features** other than **employment length** and **verification status** have a certain relation to the status of the loan.

Although we can see that the percentage of bad loans is decreasing over the years, I will not be conducting a time series analysis.
The reason why the percentage of bad loans are decreasing over the years is probably due to the fact that the global economy is steadily recovering from the 07/08 financial crisis. From this, we can see that the state of global economy has a part to play in determining whether a loan will be defaulted. If time permits, I might include leading economic indicator(s) in my analysis. 


```python
# plotting heatmap of bad loan rate in each state
state_df2 = state_df
for col in state_df2.columns:
    state_df2[col] = state_df2[col].astype(str)
    
scl = [[0.0, 'rgb(202, 202, 202)'],[0.2, 'rgb(253, 205, 200)'],[0.4, 'rgb(252, 169, 161)'],\
            [0.6, 'rgb(247, 121, 108  )'],[0.8, 'rgb(232, 70, 54)'],[1.0, 'rgb(212, 31, 13)']]

state_df2['text'] = state_df2['addr_state'] + '<br>' +\
    'Percentage of bad loans: '+state_df2['pctg_bad_loans']

data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = state_df2['addr_state'],
        z = state_df2['pctg_bad_loans'].astype(float),
        locationmode = 'USA-states',
        text = state_df2['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            )
        ),
        colorbar = dict(
            title = "Percentage"
        )
    ) ]

layout = dict(
        title = 'Percentage of bad loans in each state<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)',
        ),
    )

fig = dict( data=data, layout=layout )

iplot(fig, filename='d3-cloropleth-map')
```


<div id="3c0569b7-bdce-4b09-ad89-4db6dba50da7" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("3c0569b7-bdce-4b09-ad89-4db6dba50da7", [{"autocolorscale": false, "colorbar": {"title": "Percentage"}, "colorscale": [[0.0, "rgb(202, 202, 202)"], [0.2, "rgb(253, 205, 200)"], [0.4, "rgb(252, 169, 161)"], [0.6, "rgb(247, 121, 108  )"], [0.8, "rgb(232, 70, 54)"], [1.0, "rgb(212, 31, 13)"]], "locationmode": "USA-states", "locations": ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"], "marker": {"line": {"color": "rgb(255,255,255)", "width": 2}}, "text": ["AK<br>Percentage of bad loans: 6.9", "AL<br>Percentage of bad loans: 9.0", "AR<br>Percentage of bad loans: 7.8", "AZ<br>Percentage of bad loans: 7.8", "CA<br>Percentage of bad loans: 8.2", "CO<br>Percentage of bad loans: 6.4", "CT<br>Percentage of bad loans: 6.9", "DC<br>Percentage of bad loans: 4.8", "DE<br>Percentage of bad loans: 7.6", "FL<br>Percentage of bad loans: 8.5", "GA<br>Percentage of bad loans: 7.2", "HI<br>Percentage of bad loans: 9.5", "IA<br>Percentage of bad loans: 23.1", "ID<br>Percentage of bad loans: 9.1", "IL<br>Percentage of bad loans: 6.4", "IN<br>Percentage of bad loans: 7.1", "KS<br>Percentage of bad loans: 6.3", "KY<br>Percentage of bad loans: 7.1", "LA<br>Percentage of bad loans: 8.2", "MA<br>Percentage of bad loans: 7.4", "MD<br>Percentage of bad loans: 8.1", "ME<br>Percentage of bad loans: 0.2", "MI<br>Percentage of bad loans: 7.5", "MN<br>Percentage of bad loans: 7.5", "MO<br>Percentage of bad loans: 7.7", "MS<br>Percentage of bad loans: 5.3", "MT<br>Percentage of bad loans: 6.6", "NC<br>Percentage of bad loans: 8.1", "ND<br>Percentage of bad loans: 0.9", "NE<br>Percentage of bad loans: 2.1", "NH<br>Percentage of bad loans: 5.7", "NJ<br>Percentage of bad loans: 8.1", "NM<br>Percentage of bad loans: 8.2", "NV<br>Percentage of bad loans: 9.5", "NY<br>Percentage of bad loans: 8.6", "OH<br>Percentage of bad loans: 7.3", "OK<br>Percentage of bad loans: 8.3", "OR<br>Percentage of bad loans: 7.2", "PA<br>Percentage of bad loans: 7.7", "RI<br>Percentage of bad loans: 7.6", "SC<br>Percentage of bad loans: 6.4", "SD<br>Percentage of bad loans: 7.8", "TN<br>Percentage of bad loans: 7.2", "TX<br>Percentage of bad loans: 6.8", "UT<br>Percentage of bad loans: 8.2", "VA<br>Percentage of bad loans: 8.3", "VT<br>Percentage of bad loans: 5.5", "WA<br>Percentage of bad loans: 7.3", "WI<br>Percentage of bad loans: 6.8", "WV<br>Percentage of bad loans: 6.1", "WY<br>Percentage of bad loans: 5.9"], "z": [6.9, 9.0, 7.8, 7.8, 8.2, 6.4, 6.9, 4.8, 7.6, 8.5, 7.2, 9.5, 23.1, 9.1, 6.4, 7.1, 6.3, 7.1, 8.2, 7.4, 8.1, 0.2, 7.5, 7.5, 7.7, 5.3, 6.6, 8.1, 0.9, 2.1, 5.7, 8.1, 8.2, 9.5, 8.6, 7.3, 8.3, 7.2, 7.7, 7.6, 6.4, 7.8, 7.2, 6.8, 8.2, 8.3, 5.5, 7.3, 6.8, 6.1, 5.9], "type": "choropleth", "uid": "55c6d286-dabd-11e8-950c-8c859016eb19"}], {"geo": {"lakecolor": "rgb(255, 255, 255)", "projection": {"type": "albers usa"}, "scope": "usa", "showlakes": true}, "title": "Percentage of bad loans in each state<br>(Hover for breakdown)"}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


From the US heatmap above, Iowa has the highest proportion of bad loans while Maine, North Dakota and Nebraska have the lowest proportion of bad loans

# Feature Importance

Although the EDA above gave us a good sensing on what features are important in determining whether a loan will be paid back/paid on time, we do not know how each of them affect one another. As such, I would be conducting a feature importance study to understand which are the most important features in predicting whether a loan would be repaid back on time.


```python
print(df_transform.columns)
```

    Index(['member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'int_rate',
           'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
           'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
           'loan_status', 'url', 'desc', 'purpose', 'title', 'zip_code',
           'addr_state', 'dti', 'term', 'pymnt_plan', 'year', 'target'],
          dtype='object')



```python
# dropping irrevelant columns
df_model_1 = df_transform.drop([
        "member_id", "funded_amnt", "funded_amnt_inv", "installment", "emp_title", "issue_d", "year", \
        "loan_status", "url", "desc","title", "zip_code"], axis=1)
```


```python
cat_feature_list = [i for i in df_model_1.columns if i not in cont_feature_df.columns]
```


```python
print(cat_feature_list)
```

    ['grade', 'sub_grade', 'emp_length', 'home_ownership', 'verification_status', 'purpose', 'addr_state', 'term', 'pymnt_plan', 'target']



```python
# applying one-hot encoding for categorical variables
df_model_1 = (
    pd.get_dummies(df_model_1, columns=[i for i in cat_feature_list if i != 'target'], \
                   prefix=[i for i in cat_feature_list if i != 'target'], drop_first=True)
)
```


```python
df_model_1.head(1)
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
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>target</th>
      <th>grade_B</th>
      <th>grade_C</th>
      <th>grade_D</th>
      <th>grade_E</th>
      <th>grade_F</th>
      <th>grade_G</th>
      <th>sub_grade_A2</th>
      <th>sub_grade_A3</th>
      <th>sub_grade_A4</th>
      <th>sub_grade_A5</th>
      <th>sub_grade_B1</th>
      <th>sub_grade_B2</th>
      <th>sub_grade_B3</th>
      <th>sub_grade_B4</th>
      <th>sub_grade_B5</th>
      <th>sub_grade_C1</th>
      <th>sub_grade_C2</th>
      <th>sub_grade_C3</th>
      <th>sub_grade_C4</th>
      <th>sub_grade_C5</th>
      <th>sub_grade_D1</th>
      <th>sub_grade_D2</th>
      <th>sub_grade_D3</th>
      <th>sub_grade_D4</th>
      <th>sub_grade_D5</th>
      <th>sub_grade_E1</th>
      <th>sub_grade_E2</th>
      <th>sub_grade_E3</th>
      <th>sub_grade_E4</th>
      <th>sub_grade_E5</th>
      <th>sub_grade_F1</th>
      <th>sub_grade_F2</th>
      <th>sub_grade_F3</th>
      <th>sub_grade_F4</th>
      <th>sub_grade_F5</th>
      <th>sub_grade_G1</th>
      <th>sub_grade_G2</th>
      <th>sub_grade_G3</th>
      <th>sub_grade_G4</th>
      <th>sub_grade_G5</th>
      <th>emp_length_10+ years</th>
      <th>emp_length_2 years</th>
      <th>emp_length_3 years</th>
      <th>emp_length_4 years</th>
      <th>emp_length_5 years</th>
      <th>...</th>
      <th>addr_state_CA</th>
      <th>addr_state_CO</th>
      <th>addr_state_CT</th>
      <th>addr_state_DC</th>
      <th>addr_state_DE</th>
      <th>addr_state_FL</th>
      <th>addr_state_GA</th>
      <th>addr_state_HI</th>
      <th>addr_state_IA</th>
      <th>addr_state_ID</th>
      <th>addr_state_IL</th>
      <th>addr_state_IN</th>
      <th>addr_state_KS</th>
      <th>addr_state_KY</th>
      <th>addr_state_LA</th>
      <th>addr_state_MA</th>
      <th>addr_state_MD</th>
      <th>addr_state_ME</th>
      <th>addr_state_MI</th>
      <th>addr_state_MN</th>
      <th>addr_state_MO</th>
      <th>addr_state_MS</th>
      <th>addr_state_MT</th>
      <th>addr_state_NC</th>
      <th>addr_state_ND</th>
      <th>addr_state_NE</th>
      <th>addr_state_NH</th>
      <th>addr_state_NJ</th>
      <th>addr_state_NM</th>
      <th>addr_state_NV</th>
      <th>addr_state_NY</th>
      <th>addr_state_OH</th>
      <th>addr_state_OK</th>
      <th>addr_state_OR</th>
      <th>addr_state_PA</th>
      <th>addr_state_RI</th>
      <th>addr_state_SC</th>
      <th>addr_state_SD</th>
      <th>addr_state_TN</th>
      <th>addr_state_TX</th>
      <th>addr_state_UT</th>
      <th>addr_state_VA</th>
      <th>addr_state_VT</th>
      <th>addr_state_WA</th>
      <th>addr_state_WI</th>
      <th>addr_state_WV</th>
      <th>addr_state_WY</th>
      <th>term__36_months</th>
      <th>term__60_months</th>
      <th>pymnt_plan_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5000.0</td>
      <td>10.65</td>
      <td>24000.0</td>
      <td>27.65</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 129 columns</p>
</div>




```python
df_model_1.shape
```




    (868956, 129)




```python
print('number of good loans: ' + str(df_model_1.query("target==1").shape[0]))
print('number of bad loans: ' + str(df_model_1.query("target==0").shape[0]))
```

    number of good loans: 802270
    number of bad loans: 66686


We can see that the data is skewed. I will under-sample the the data of good loans to obtain a data set where good and bad loans are proportionate, i.e. baseline accuracy = 0.5


```python
def under_sampling_good_loans(df):
    good_df = df.query("target==1")
    bad_df = df.query("target==0")
    good_sample_df = good_df.sample(n=bad_df.shape[0], random_state=13)
    final_df = pd.concat([bad_df, good_sample_df], axis=0, ignore_index=True)
    
    return final_df
```


```python
df_model_1 = under_sampling_good_loans(df_model_1)
```


```python
x = df_model_1[[i for i in df_model_1.columns if i != 'target']]
y = df_model_1['target']
```


```python
# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)
```


```python
# using min-max scaler so that my data will be within 1 and 0
mm = MinMaxScaler()
x_mm_train = mm.fit_transform(x_train)
x_mm_test = mm.transform(x_test)
```


```python
# using gridsearch to optimise my selected model
gs_params ={
    'max_depth': np.linspace(1,129,5,dtype=int),
    'n_estimators':np.linspace(10,100,5,dtype=int),
    'random_state':[13]
}

# choosing random forest as my base model
rclf_gs = GridSearchCV(RandomForestClassifier(), gs_params, n_jobs = -1, verbose = 1, cv = 3)
```


```python
rclf_gs.fit(x_mm_train, y_train)
```

    Fitting 3 folds for each of 25 candidates, totalling 75 fits


    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done  75 out of  75 | elapsed:  3.5min finished





    GridSearchCV(cv=3, error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'max_depth': array([  1,  33,  65,  97, 129]), 'n_estimators': array([ 10,  32,  55,  77, 100]), 'random_state': [13]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=1)




```python
rclf_gs.best_estimator_
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=33, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=13, verbose=0, warm_start=False)




```python
rfclf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=33, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
            oob_score=False, random_state=13, verbose=1, warm_start=False)
rfclf.fit(x_mm_train, y_train)
```

    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    3.6s
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:    8.6s finished





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=33, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=-1,
                oob_score=False, random_state=13, verbose=1, warm_start=False)




```python
def model_perf(model, X_test, y_test):
    """ 
    Evaluating performance of supervised models
    Returns
    -----------
    Model score, precision score (minority class), recall score (minority class),
    f1 score (minority class) and auc score (unweighted average)
    """
    model_yhat = model.predict(X_test)
    model_score = model.score(X_test, y_test)
    model_f1 = f1_score(y_test, model_yhat)
    model_precision = precision_score(y_test, model_yhat)
    model_recall = recall_score(y_test, model_yhat)
    model_auc = roc_auc_score(y_test, model_yhat)
    
    return (model_score, model_precision, model_recall, model_f1, model_auc)
```


```python
model_score, model_precision, model_recall, model_f1, model_auc = model_perf(rfclf, x_mm_test, y_test)
```

    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    0.3s
    [Parallel(n_jobs=4)]: Done 100 out of 100 | elapsed:    0.6s finished
    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    0.2s
    [Parallel(n_jobs=4)]: Done 100 out of 100 | elapsed:    0.5s finished



```python
model_perf_df = pd.DataFrame({'rf_all_features':[model_score, model_precision, model_recall, model_f1, model_auc]},
                      index = ['model score','precision score','recall score','f1 score', 'auc'])
model_perf_df
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
      <th>rf_all_features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>model score</th>
      <td>0.667950</td>
    </tr>
    <tr>
      <th>precision score</th>
      <td>0.679116</td>
    </tr>
    <tr>
      <th>recall score</th>
      <td>0.627907</td>
    </tr>
    <tr>
      <th>f1 score</th>
      <td>0.652508</td>
    </tr>
    <tr>
      <th>auc</th>
      <td>0.667671</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature_impt_df = (
    pd.DataFrame(rfclf.feature_importances_, index=x.columns, columns=['importance'])
    .reset_index()
    .rename(columns={"index":"features"})
    .sort_values('importance', ascending=False)
    .pipe(lambda x: x.assign(cumulative_sum=x.importance.cumsum()))
    .reset_index(drop=True)
    )
feature_impt_df  
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
      <th>cumulative_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>int_rate</td>
      <td>0.168165</td>
      <td>0.168165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dti</td>
      <td>0.109677</td>
      <td>0.277842</td>
    </tr>
    <tr>
      <th>2</th>
      <td>annual_inc</td>
      <td>0.106555</td>
      <td>0.384397</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loan_amnt</td>
      <td>0.095408</td>
      <td>0.479805</td>
    </tr>
    <tr>
      <th>4</th>
      <td>emp_length_10+ years</td>
      <td>0.014383</td>
      <td>0.494188</td>
    </tr>
    <tr>
      <th>5</th>
      <td>verification_status_Verified</td>
      <td>0.014361</td>
      <td>0.508549</td>
    </tr>
    <tr>
      <th>6</th>
      <td>purpose_debt_consolidation</td>
      <td>0.014104</td>
      <td>0.522652</td>
    </tr>
    <tr>
      <th>7</th>
      <td>verification_status_Source_Verified</td>
      <td>0.013772</td>
      <td>0.536425</td>
    </tr>
    <tr>
      <th>8</th>
      <td>addr_state_CA</td>
      <td>0.012769</td>
      <td>0.549194</td>
    </tr>
    <tr>
      <th>9</th>
      <td>purpose_credit_card</td>
      <td>0.011340</td>
      <td>0.560534</td>
    </tr>
    <tr>
      <th>10</th>
      <td>home_ownership_RENT</td>
      <td>0.010517</td>
      <td>0.571050</td>
    </tr>
    <tr>
      <th>11</th>
      <td>home_ownership_MORTGAGE</td>
      <td>0.010400</td>
      <td>0.581450</td>
    </tr>
    <tr>
      <th>12</th>
      <td>addr_state_NY</td>
      <td>0.010284</td>
      <td>0.591734</td>
    </tr>
    <tr>
      <th>13</th>
      <td>emp_length_2 years</td>
      <td>0.009588</td>
      <td>0.601322</td>
    </tr>
    <tr>
      <th>14</th>
      <td>addr_state_TX</td>
      <td>0.009459</td>
      <td>0.610781</td>
    </tr>
    <tr>
      <th>15</th>
      <td>addr_state_FL</td>
      <td>0.009444</td>
      <td>0.620225</td>
    </tr>
    <tr>
      <th>16</th>
      <td>emp_length_&lt; 1 year</td>
      <td>0.009208</td>
      <td>0.629433</td>
    </tr>
    <tr>
      <th>17</th>
      <td>emp_length_3 years</td>
      <td>0.009084</td>
      <td>0.638517</td>
    </tr>
    <tr>
      <th>18</th>
      <td>term__36_months</td>
      <td>0.008494</td>
      <td>0.647011</td>
    </tr>
    <tr>
      <th>19</th>
      <td>emp_length_5 years</td>
      <td>0.008313</td>
      <td>0.655325</td>
    </tr>
    <tr>
      <th>20</th>
      <td>grade_F</td>
      <td>0.008155</td>
      <td>0.663480</td>
    </tr>
    <tr>
      <th>21</th>
      <td>emp_length_4 years</td>
      <td>0.008031</td>
      <td>0.671511</td>
    </tr>
    <tr>
      <th>22</th>
      <td>grade_E</td>
      <td>0.007999</td>
      <td>0.679510</td>
    </tr>
    <tr>
      <th>23</th>
      <td>grade_D</td>
      <td>0.007996</td>
      <td>0.687506</td>
    </tr>
    <tr>
      <th>24</th>
      <td>term__60_months</td>
      <td>0.007974</td>
      <td>0.695480</td>
    </tr>
    <tr>
      <th>25</th>
      <td>grade_B</td>
      <td>0.007899</td>
      <td>0.703379</td>
    </tr>
    <tr>
      <th>26</th>
      <td>emp_length_7 years</td>
      <td>0.007661</td>
      <td>0.711039</td>
    </tr>
    <tr>
      <th>27</th>
      <td>home_ownership_OWN</td>
      <td>0.007541</td>
      <td>0.718580</td>
    </tr>
    <tr>
      <th>28</th>
      <td>emp_length_6 years</td>
      <td>0.007405</td>
      <td>0.725985</td>
    </tr>
    <tr>
      <th>29</th>
      <td>emp_length_8 years</td>
      <td>0.007221</td>
      <td>0.733206</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>98</th>
      <td>addr_state_NH</td>
      <td>0.001505</td>
      <td>0.981651</td>
    </tr>
    <tr>
      <th>99</th>
      <td>sub_grade_F2</td>
      <td>0.001490</td>
      <td>0.983141</td>
    </tr>
    <tr>
      <th>100</th>
      <td>addr_state_RI</td>
      <td>0.001449</td>
      <td>0.984590</td>
    </tr>
    <tr>
      <th>101</th>
      <td>purpose_house</td>
      <td>0.001439</td>
      <td>0.986029</td>
    </tr>
    <tr>
      <th>102</th>
      <td>sub_grade_F3</td>
      <td>0.001279</td>
      <td>0.987307</td>
    </tr>
    <tr>
      <th>103</th>
      <td>purpose_wedding</td>
      <td>0.001229</td>
      <td>0.988537</td>
    </tr>
    <tr>
      <th>104</th>
      <td>addr_state_MS</td>
      <td>0.001217</td>
      <td>0.989754</td>
    </tr>
    <tr>
      <th>105</th>
      <td>addr_state_DE</td>
      <td>0.001105</td>
      <td>0.990858</td>
    </tr>
    <tr>
      <th>106</th>
      <td>sub_grade_F4</td>
      <td>0.001091</td>
      <td>0.991949</td>
    </tr>
    <tr>
      <th>107</th>
      <td>sub_grade_F5</td>
      <td>0.001067</td>
      <td>0.993016</td>
    </tr>
    <tr>
      <th>108</th>
      <td>addr_state_MT</td>
      <td>0.001062</td>
      <td>0.994078</td>
    </tr>
    <tr>
      <th>109</th>
      <td>addr_state_SD</td>
      <td>0.000790</td>
      <td>0.994868</td>
    </tr>
    <tr>
      <th>110</th>
      <td>addr_state_WY</td>
      <td>0.000786</td>
      <td>0.995654</td>
    </tr>
    <tr>
      <th>111</th>
      <td>addr_state_DC</td>
      <td>0.000761</td>
      <td>0.996415</td>
    </tr>
    <tr>
      <th>112</th>
      <td>addr_state_VT</td>
      <td>0.000555</td>
      <td>0.996970</td>
    </tr>
    <tr>
      <th>113</th>
      <td>sub_grade_G1</td>
      <td>0.000470</td>
      <td>0.997440</td>
    </tr>
    <tr>
      <th>114</th>
      <td>purpose_renewable_energy</td>
      <td>0.000430</td>
      <td>0.997870</td>
    </tr>
    <tr>
      <th>115</th>
      <td>sub_grade_G2</td>
      <td>0.000389</td>
      <td>0.998259</td>
    </tr>
    <tr>
      <th>116</th>
      <td>addr_state_NE</td>
      <td>0.000327</td>
      <td>0.998586</td>
    </tr>
    <tr>
      <th>117</th>
      <td>sub_grade_G3</td>
      <td>0.000294</td>
      <td>0.998880</td>
    </tr>
    <tr>
      <th>118</th>
      <td>purpose_educational</td>
      <td>0.000285</td>
      <td>0.999165</td>
    </tr>
    <tr>
      <th>119</th>
      <td>sub_grade_G5</td>
      <td>0.000203</td>
      <td>0.999367</td>
    </tr>
    <tr>
      <th>120</th>
      <td>sub_grade_G4</td>
      <td>0.000199</td>
      <td>0.999566</td>
    </tr>
    <tr>
      <th>121</th>
      <td>home_ownership_OTHER</td>
      <td>0.000185</td>
      <td>0.999752</td>
    </tr>
    <tr>
      <th>122</th>
      <td>addr_state_ND</td>
      <td>0.000109</td>
      <td>0.999860</td>
    </tr>
    <tr>
      <th>123</th>
      <td>addr_state_ME</td>
      <td>0.000100</td>
      <td>0.999960</td>
    </tr>
    <tr>
      <th>124</th>
      <td>home_ownership_NONE</td>
      <td>0.000022</td>
      <td>0.999982</td>
    </tr>
    <tr>
      <th>125</th>
      <td>pymnt_plan_y</td>
      <td>0.000014</td>
      <td>0.999996</td>
    </tr>
    <tr>
      <th>126</th>
      <td>addr_state_IA</td>
      <td>0.000002</td>
      <td>0.999998</td>
    </tr>
    <tr>
      <th>127</th>
      <td>addr_state_ID</td>
      <td>0.000002</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>128 rows × 3 columns</p>
</div>




```python
shortlisted_features_df = feature_impt_df.query("importance >= 1/129")
shortlisted_features_df
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
      <th>cumulative_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>int_rate</td>
      <td>0.168165</td>
      <td>0.168165</td>
    </tr>
    <tr>
      <th>1</th>
      <td>dti</td>
      <td>0.109677</td>
      <td>0.277842</td>
    </tr>
    <tr>
      <th>2</th>
      <td>annual_inc</td>
      <td>0.106555</td>
      <td>0.384397</td>
    </tr>
    <tr>
      <th>3</th>
      <td>loan_amnt</td>
      <td>0.095408</td>
      <td>0.479805</td>
    </tr>
    <tr>
      <th>4</th>
      <td>emp_length_10+ years</td>
      <td>0.014383</td>
      <td>0.494188</td>
    </tr>
    <tr>
      <th>5</th>
      <td>verification_status_Verified</td>
      <td>0.014361</td>
      <td>0.508549</td>
    </tr>
    <tr>
      <th>6</th>
      <td>purpose_debt_consolidation</td>
      <td>0.014104</td>
      <td>0.522652</td>
    </tr>
    <tr>
      <th>7</th>
      <td>verification_status_Source_Verified</td>
      <td>0.013772</td>
      <td>0.536425</td>
    </tr>
    <tr>
      <th>8</th>
      <td>addr_state_CA</td>
      <td>0.012769</td>
      <td>0.549194</td>
    </tr>
    <tr>
      <th>9</th>
      <td>purpose_credit_card</td>
      <td>0.011340</td>
      <td>0.560534</td>
    </tr>
    <tr>
      <th>10</th>
      <td>home_ownership_RENT</td>
      <td>0.010517</td>
      <td>0.571050</td>
    </tr>
    <tr>
      <th>11</th>
      <td>home_ownership_MORTGAGE</td>
      <td>0.010400</td>
      <td>0.581450</td>
    </tr>
    <tr>
      <th>12</th>
      <td>addr_state_NY</td>
      <td>0.010284</td>
      <td>0.591734</td>
    </tr>
    <tr>
      <th>13</th>
      <td>emp_length_2 years</td>
      <td>0.009588</td>
      <td>0.601322</td>
    </tr>
    <tr>
      <th>14</th>
      <td>addr_state_TX</td>
      <td>0.009459</td>
      <td>0.610781</td>
    </tr>
    <tr>
      <th>15</th>
      <td>addr_state_FL</td>
      <td>0.009444</td>
      <td>0.620225</td>
    </tr>
    <tr>
      <th>16</th>
      <td>emp_length_&lt; 1 year</td>
      <td>0.009208</td>
      <td>0.629433</td>
    </tr>
    <tr>
      <th>17</th>
      <td>emp_length_3 years</td>
      <td>0.009084</td>
      <td>0.638517</td>
    </tr>
    <tr>
      <th>18</th>
      <td>term__36_months</td>
      <td>0.008494</td>
      <td>0.647011</td>
    </tr>
    <tr>
      <th>19</th>
      <td>emp_length_5 years</td>
      <td>0.008313</td>
      <td>0.655325</td>
    </tr>
    <tr>
      <th>20</th>
      <td>grade_F</td>
      <td>0.008155</td>
      <td>0.663480</td>
    </tr>
    <tr>
      <th>21</th>
      <td>emp_length_4 years</td>
      <td>0.008031</td>
      <td>0.671511</td>
    </tr>
    <tr>
      <th>22</th>
      <td>grade_E</td>
      <td>0.007999</td>
      <td>0.679510</td>
    </tr>
    <tr>
      <th>23</th>
      <td>grade_D</td>
      <td>0.007996</td>
      <td>0.687506</td>
    </tr>
    <tr>
      <th>24</th>
      <td>term__60_months</td>
      <td>0.007974</td>
      <td>0.695480</td>
    </tr>
    <tr>
      <th>25</th>
      <td>grade_B</td>
      <td>0.007899</td>
      <td>0.703379</td>
    </tr>
  </tbody>
</table>
</div>



I have used random forest to determine feature importance.

Since there are 129 features, a feature would be given a 0.00775(1/129) importance score on average. As such, the above dataframe is a filter that shows us features that are above the average score and it is ranked by feature importance, where the first feature is the most important

We can see that 25 features are selected. The end result may not be very intiutive from a business point of view as there are too many one-hot encoded features.

I will attempt to reduce the number of one-hot encoded features while maintaining the model performance.

# Data Cleaning and Transformation Part 2


```python
# converting grade and sub_grade into continuous variable
grade_dict = {"A":1.0, "B":2.0, "C":3.0, "D":4.0, "E":5.0, "F":6.0, "G":7.0}
sub_grade_dict = {
    "A1":1.0, "A2":1.2, "A3":1.4, "A4":1.6, "A5":1.8,
    "B1":2.0, "B2":2.2, "B3":2.4, "B4":2.6, "B5":2.8,
    "C1":3.0, "C2":3.2, "C3":3.4, "C4":3.6, "C5":3.8,
    "D1":4.0, "D2":4.2, "D3":4.4, "D4":4.6, "D5":4.8,
    "E1":5.0, "E2":5.2, "E3":5.4, "E4":5.6, "E5":5.8,
    "F1":6.0, "F2":6.2, "F3":6.4, "F4":6.6, "F5":6.8,
    "G1":7.0, "G2":7.2, "G3":7.4, "G4":7.6, "G5":7.8
    }
emp_length_dict={
    "10+ years":10, "< 1 year":0, "1 year":1, "2 years":2, "3 years":3,
    "4 years":4, "5 years":5, "6 years":6, "7 years":7, "8 years":8, "9 years":9
    }
```


```python
df_transform_2 = df_transform
df_transform_2['grade'] = [grade_dict[i] for i in df_transform_2['grade']]
df_transform_2['sub_grade'] = [sub_grade_dict[i] for i in df_transform_2['sub_grade']]  
df_transform_2['emp_length'] = [emp_length_dict[i] if i != 'blank' else np.nan for i in df_transform_2['emp_length']]
```


```python
df_transform_2['emp_length'].describe()
```




    count    825262.000000
    mean          6.009078
    std           3.665247
    min           0.000000
    25%           3.000000
    50%           6.000000
    75%          10.000000
    max          10.000000
    Name: emp_length, dtype: float64




```python
# replacing null values in emp_length with its median value
df_transform_2['emp_length'].fillna(6, inplace=True)
```


```python
# plotting a heatmap of a correlation matrix of all the continuous features
cont_feature_df = df_transform_2[['loan_amnt', 'int_rate', 'funded_amnt', 'funded_amnt_inv', 'installment', 'annual_inc', 'dti', 'grade', 'sub_grade', 'emp_length']]

_ = plt.subplots(figsize = (12,7))

corr = cont_feature_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(600, 20, as_cmap=True)

_ = sns.heatmap(corr, mask=mask, annot=True, cmap =cmap)
```


![png](funding-societies-assignment_files/funding-societies-assignment_67_0.png)


Since interest rate, grade and sub_grade are highly correlated, we will be removing two of them. However, which two should we be removing?


```python
grade_explore = (
    df_transform_2
    [['int_rate', 'grade']]
    .groupby("grade")
    .agg({"int_rate": "mean"})
    .reset_index()
)
_ = sns.barplot(x='grade', y='int_rate', data=grade_explore, color='royalblue')
```


![png](funding-societies-assignment_files/funding-societies-assignment_69_0.png)


There appears to be some anomaly in the data as the average interest rate of grade F (6.0) is similar to the average interest rate of grade C (3.0)


```python
# plotting the frequency distribution of interest rate of each grade
_ = plt.figure(figsize=(20,15))
for idx, g in enumerate(sorted(df_transform_2.grade.unique())):
    start_subplot = 331
    _ = plt.subplot(start_subplot + idx, )
    _ = sns.distplot(df_transform_2.query("grade=={}".format(g)).int_rate)
    _ = plt.xticks(np.linspace(0,30,5))
    _ = plt.title('Frequency distribution of interest rate (grade ' + str(g) + ')')
```


![png](funding-societies-assignment_files/funding-societies-assignment_71_0.png)


From the graphs above, we can see that there are some errerroneous data in grade F (6.0). Half of grade F loans have interest rate less than 7.5% which is an anomaly. Due to an interest of time I will drop these rows as instead of conducting further analysis as I would still retain ~10k rows of grade F loans which will suffice during modelling stage


```python
index_to_drop = df_transform_2.query("grade==6 and int_rate <= 7.5").index.values
df_clean_2 = (
    df_transform_2.
    drop(index_to_drop, axis=0)
    .reset_index()
    )
```


```python
# re-plotting the bar chart showing the relationship between interest rate and grade
grade_explore = (
    df_clean_2
    [['int_rate', 'grade']]
    .groupby("grade")
    .agg({"int_rate": "mean"})
    .reset_index()
)
_ = sns.barplot(x='grade', y='int_rate', data=grade_explore, color='royalblue')
_ = plt.title('Average interest rate of each loan grade')
```


![png](funding-societies-assignment_files/funding-societies-assignment_74_0.png)



```python
# plotting bar chart showing the relationship between interest rate and sub_grade
sub_grade_explore = (
    df_clean_2
    [['int_rate', 'sub_grade']]
    .groupby("sub_grade")
    .agg({"int_rate": "mean"})
    .reset_index()
)
_ = plt.figure(figsize=(10,6))
_ = sns.barplot(x='sub_grade', y='int_rate', data=sub_grade_explore, color='royalblue')
_ = plt.title('Average interest rate of each sub_grade')
_ = plt.xticks(rotation=75, fontsize='10')
```


![png](funding-societies-assignment_files/funding-societies-assignment_75_0.png)


From the correlation heatmap plot, we can see that interest rate has a high correlation with grade and sub_grade. As such, we can remove 2 of the three features in our model.

From the grade bar chart, we can see that there is an obvious difference in interet rate between each grade. Whereas from the sub_grade bar chart we can see that interest rate can be hardly differeniated at the tail end of the sub_grade list (G1 to G5). As such I will retain sub_grade and remove grade and interest rate in my model.


```python
df_transform_2 = df_clean_2
df_transform_2.columns
```




    Index(['index', 'member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv',
           'int_rate', 'installment', 'grade', 'sub_grade', 'emp_title',
           'emp_length', 'home_ownership', 'annual_inc', 'verification_status',
           'issue_d', 'loan_status', 'url', 'desc', 'purpose', 'title', 'zip_code',
           'addr_state', 'dti', 'term', 'pymnt_plan', 'year', 'target'],
          dtype='object')




```python
df_model_2 = df_transform_2[['loan_amnt', 'sub_grade', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', \
                             'purpose', 'addr_state', 'dti', 'term', 'pymnt_plan', 'target']]

cat_feature_list = ['home_ownership', 'verification_status', 'purpose', 'addr_state', 'term', 'pymnt_plan']
cont_feature_list = ['loan_amnt', 'sub_grade', 'emp_length', 'annual_inc', 'dti']
```


```python
# applying one-hot encoding to my categorical features
df_model_2 = pd.get_dummies(df_model_2, columns=[i for i in cat_feature_list], prefix=[i for i in cat_feature_list], drop_first=True)
```


```python
df_model_2.columns
```




    Index(['loan_amnt', 'sub_grade', 'emp_length', 'annual_inc', 'dti', 'target',
           'home_ownership_MORTGAGE', 'home_ownership_NONE',
           'home_ownership_OTHER', 'home_ownership_OWN', 'home_ownership_RENT',
           'verification_status_Source_Verified', 'verification_status_Verified',
           'purpose_credit_card', 'purpose_debt_consolidation',
           'purpose_educational', 'purpose_home_improvement', 'purpose_house',
           'purpose_major_purchase', 'purpose_medical', 'purpose_moving',
           'purpose_other', 'purpose_renewable_energy', 'purpose_small_business',
           'purpose_vacation', 'purpose_wedding', 'addr_state_AL', 'addr_state_AR',
           'addr_state_AZ', 'addr_state_CA', 'addr_state_CO', 'addr_state_CT',
           'addr_state_DC', 'addr_state_DE', 'addr_state_FL', 'addr_state_GA',
           'addr_state_HI', 'addr_state_IA', 'addr_state_ID', 'addr_state_IL',
           'addr_state_IN', 'addr_state_KS', 'addr_state_KY', 'addr_state_LA',
           'addr_state_MA', 'addr_state_MD', 'addr_state_ME', 'addr_state_MI',
           'addr_state_MN', 'addr_state_MO', 'addr_state_MS', 'addr_state_MT',
           'addr_state_NC', 'addr_state_ND', 'addr_state_NE', 'addr_state_NH',
           'addr_state_NJ', 'addr_state_NM', 'addr_state_NV', 'addr_state_NY',
           'addr_state_OH', 'addr_state_OK', 'addr_state_OR', 'addr_state_PA',
           'addr_state_RI', 'addr_state_SC', 'addr_state_SD', 'addr_state_TN',
           'addr_state_TX', 'addr_state_UT', 'addr_state_VA', 'addr_state_VT',
           'addr_state_WA', 'addr_state_WI', 'addr_state_WV', 'addr_state_WY',
           'term__36_months', 'term__60_months', 'pymnt_plan_y'],
          dtype='object')




```python
df_model_2.shape
```




    (859145, 79)




```python
# calculating the proportion of 1s in my one-hot encoded categorical features
pctg_of_1s_in_cat_feature_dict = {}
one_hot_encoding_feature_list = [i for i in df_model_2.columns if i not in cont_feature_list and i != 'target']
for i in one_hot_encoding_feature_list:
    pctg_of_1s_in_cat_feature_dict[i] = df_model_2.query("{} == 1".format(i)).shape[0]/df_model_2.shape[0] * 100
```


```python
pctg_of_1s_in_cat_feature_df = (
    pd.DataFrame(list(pctg_of_1s_in_cat_feature_dict.items()), columns=['features', 'pctg_of_1s'])
    .sort_values('pctg_of_1s')
    )

_ = plt.figure(figsize=(20,15))
_ = sns.barplot(x='pctg_of_1s', y='features', data=pctg_of_1s_in_cat_feature_df, color='royalblue', orient="h")
_ = plt.title("Percentage of 1s in column")
```


![png](funding-societies-assignment_files/funding-societies-assignment_83_0.png)


The chart above shows us the percentage of 1's of all the one-hot encoded features.

If a feature has too many rows of the same value, the coefficient of that feature would not give us much information because it would be bias towards dominant value. 

As such, I will remove features that are 90% filled with the same value


```python
pctg_of_1s_in_cat_feature_df.query("pctg_of_1s >= 10 and pctg_of_1s <= 90").features.unique()
```




    array(['addr_state_CA', 'purpose_credit_card', 'term__60_months',
           'verification_status_Verified',
           'verification_status_Source_Verified', 'home_ownership_RENT',
           'home_ownership_MORTGAGE', 'purpose_debt_consolidation',
           'term__36_months'], dtype=object)




```python
cont_feature_list
```




    ['loan_amnt', 'sub_grade', 'emp_length', 'annual_inc', 'dti']




```python
df_model_2 = df_model_2[['loan_amnt', 'sub_grade', 'emp_length', 'annual_inc', 'dti',\
                        'addr_state_CA', 'purpose_credit_card', 'term__60_months','verification_status_Verified', \
                        'verification_status_Source_Verified', 'home_ownership_RENT','home_ownership_MORTGAGE', \
                        'purpose_debt_consolidation','term__36_months', 'target']]
```


```python
df_model_2.shape
```




    (859145, 15)




```python
df_model_2 = under_sampling_good_loans(df_model_2)

x = df_model_2[[i for i in df_model_2.columns if i != 'target']]
y = df_model_2['target']

# split data train 70 % and test 30 %
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)

# using min-max scaler so that my data will be within 1 and 0
x_mm_train = mm.fit_transform(x_train)
x_mm_test = mm.transform(x_test)
```


```python
gs_params ={
    'max_depth': np.linspace(1,15,3,dtype=int),
    'n_estimators':np.linspace(10,100,5,dtype=int),
    'random_state':[13]
}

rclf_gs = GridSearchCV(RandomForestClassifier(), gs_params, n_jobs = -1, verbose = 1, cv = 3)

rclf_gs.fit(x_mm_train, y_train)
```

    Fitting 3 folds for each of 15 candidates, totalling 45 fits


    [Parallel(n_jobs=-1)]: Done  45 out of  45 | elapsed:   49.4s finished





    GridSearchCV(cv=3, error_score='raise',
           estimator=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False),
           fit_params=None, iid=True, n_jobs=-1,
           param_grid={'max_depth': array([ 1,  8, 15]), 'n_estimators': array([ 10,  32,  55,  77, 100]), 'random_state': [13]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=1)




```python
rclf_gs.best_estimator_
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=8, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=77, n_jobs=1,
                oob_score=False, random_state=13, verbose=0, warm_start=False)




```python
rfclf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=8, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=77, n_jobs=-1,
            oob_score=False, random_state=13, verbose=1, warm_start=False)

rfclf.fit(x_mm_train, y_train)
```

    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:    0.8s
    [Parallel(n_jobs=-1)]: Done  77 out of  77 | elapsed:    1.5s finished





    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=8, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=77, n_jobs=-1,
                oob_score=False, random_state=13, verbose=1, warm_start=False)




```python
model_score_2, model_precision_2, model_recall_2, model_f1_2, model_auc_2 = model_perf(rfclf, x_mm_test, y_test)
model_perf_df = pd.DataFrame({'rf_all_features':[model_score, model_precision, model_recall, model_f1, model_auc],
                             'rf_reduced_features': [model_score_2, model_precision_2, model_recall_2, model_f1_2, model_auc_2]},
                      index = ['model score','precision score','recall score','f1 score', 'auc'])
model_perf_df
```

    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=4)]: Done  77 out of  77 | elapsed:    0.1s finished
    [Parallel(n_jobs=4)]: Done  42 tasks      | elapsed:    0.1s
    [Parallel(n_jobs=4)]: Done  77 out of  77 | elapsed:    0.1s finished





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
      <th>rf_all_features</th>
      <th>rf_reduced_features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>model score</th>
      <td>0.667950</td>
      <td>0.630858</td>
    </tr>
    <tr>
      <th>precision score</th>
      <td>0.679116</td>
      <td>0.636698</td>
    </tr>
    <tr>
      <th>recall score</th>
      <td>0.627907</td>
      <td>0.594222</td>
    </tr>
    <tr>
      <th>f1 score</th>
      <td>0.652508</td>
      <td>0.614727</td>
    </tr>
    <tr>
      <th>auc</th>
      <td>0.667671</td>
      <td>0.630538</td>
    </tr>
  </tbody>
</table>
</div>




```python
feature_impt_df = (
    pd.DataFrame(rfclf.feature_importances_, index=x.columns, columns=['importance'])
    .reset_index()
    .rename(columns={"index":"features"})
    .sort_values('importance', ascending=False)
    .pipe(lambda x: x.assign(cumulative_sum=x.importance.cumsum()))
    .reset_index(drop=True)
    )
feature_impt_df  
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
      <th>cumulative_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub_grade</td>
      <td>0.570812</td>
      <td>0.570812</td>
    </tr>
    <tr>
      <th>1</th>
      <td>annual_inc</td>
      <td>0.113106</td>
      <td>0.683918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>dti</td>
      <td>0.057790</td>
      <td>0.741709</td>
    </tr>
    <tr>
      <th>3</th>
      <td>verification_status_Verified</td>
      <td>0.052740</td>
      <td>0.794449</td>
    </tr>
    <tr>
      <th>4</th>
      <td>loan_amnt</td>
      <td>0.050128</td>
      <td>0.844577</td>
    </tr>
    <tr>
      <th>5</th>
      <td>purpose_credit_card</td>
      <td>0.033116</td>
      <td>0.877693</td>
    </tr>
    <tr>
      <th>6</th>
      <td>term__36_months</td>
      <td>0.024531</td>
      <td>0.902224</td>
    </tr>
    <tr>
      <th>7</th>
      <td>home_ownership_RENT</td>
      <td>0.022185</td>
      <td>0.924409</td>
    </tr>
    <tr>
      <th>8</th>
      <td>emp_length</td>
      <td>0.017985</td>
      <td>0.942394</td>
    </tr>
    <tr>
      <th>9</th>
      <td>verification_status_Source_Verified</td>
      <td>0.017878</td>
      <td>0.960272</td>
    </tr>
    <tr>
      <th>10</th>
      <td>term__60_months</td>
      <td>0.016771</td>
      <td>0.977043</td>
    </tr>
    <tr>
      <th>11</th>
      <td>home_ownership_MORTGAGE</td>
      <td>0.013642</td>
      <td>0.990684</td>
    </tr>
    <tr>
      <th>12</th>
      <td>purpose_debt_consolidation</td>
      <td>0.004687</td>
      <td>0.995371</td>
    </tr>
    <tr>
      <th>13</th>
      <td>addr_state_CA</td>
      <td>0.004629</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
shortlisted_features_df = feature_impt_df.query("importance >= 1/13")
shortlisted_features_df
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
      <th>cumulative_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sub_grade</td>
      <td>0.570812</td>
      <td>0.570812</td>
    </tr>
    <tr>
      <th>1</th>
      <td>annual_inc</td>
      <td>0.113106</td>
      <td>0.683918</td>
    </tr>
  </tbody>
</table>
</div>



# Conclusion

Although the revised random forest model gave a lower score on model performance as compared to the original model, the difference was not a lot, ~3%.

However, the result of the feature importance is much clearer and more intuitive to business users. The top features are **sub_grade** and **annual_inc**.

# Considerations

1. To engineer a new feature that gives a scoring on the sentiment of the 'description' and 'title' columns (sentiment analysis)
2. To include leading economic indicator(s) as addition feature(s)
3. Create a total of 3 folds to cross-validate my model
4. Use lasso regression to conduct feature selection
5. Use recursive feature elimination to conduct feature selection 
6. Use decision tree to plot out the how decisions are branched out
