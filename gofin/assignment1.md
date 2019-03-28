

```python
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

pd.set_option("display.max_columns", 100)
%matplotlib inline
```

    /Users/timong/anaconda/envs/py36/lib/python3.6/site-packages/psycopg2/__init__.py:144: UserWarning: The psycopg2 wheel package will be renamed from release 2.8; in order to keep installing from binary please use "pip install psycopg2-binary" instead. For details see: <http://initd.org/psycopg/docs/install.html#binary-install-from-pypi>.
      """)



```python
def pg_load_table(file_path, table_name, dbname, user):
    '''
    This function upload csv to a target table
    '''
    try:
        conn = psycopg2.connect(dbname=dbname, user=user)
        print("Connecting to Database")
        cur = conn.cursor()
        f = open(file_path, "r")
        # Truncate the table first
        cur.execute("Truncate {} Cascade;".format(table_name))
        print("Truncated {}".format(table_name))
        # Load table from the file with header
        cur.copy_expert("copy {} from STDIN CSV HEADER QUOTE '\"'".format(table_name), f)
        cur.execute("commit;")
        print("Loaded data into {}".format(table_name))
        conn.close()
        print("DB connection closed.")

    except Exception as e:
        print("Error: {}".format(str(e)))
        sys.exit(1)
```


```python
with open("./postgres.yaml", "r") as stream:
    try:
        data_loaded = yaml.load(stream)
    except yaml.YAMLERROR as exc:
        print(exc)
```


```python
dbname = data_loaded['dbname']
user = data_loaded['user']
pg_load_table('./user_transactions.csv', 'transactions', dbname, user)
pg_load_table('./user_label_branch_edit.csv', 'user_label_branch', dbname, user)
# decided to load user_base_part1 only as user_base_part2 is just a duplicate of the first (ref to below cell)
pg_load_table('./user_base_part1.csv', 'user_base', dbname, user)
```

    Connecting to Database
    Truncated transactions
    Loaded data into transactions
    DB connection closed.
    Connecting to Database
    Truncated user_label_branch
    Loaded data into user_label_branch
    DB connection closed.
    Connecting to Database
    Truncated user_base
    Loaded data into user_base
    DB connection closed.


outstanding: total outstanding amount of credit card usage
credit_limit: credit limit amount that can be used
total_cash_usage: last month total cash usage of customer
total_retail_usage: last month total retail usage of customer
bill: last month customer bill amount
remaining_bill: remaining bill that has not been paid in the last month (assuming that this refers to the remaining amount of > month -2 bills that was not paid in month -1)

number_of_cards: # cards owned by customer
branch_code: branch code of bank
default_flag: 1: default, 0: non_default

total_usage = assume total_cash_usage + total_retail_usage
payment_ratio: payment per bill ratio in the last month (payment made to billing amount in the last month)
payment_ratio_3month: payment per bill ratio in the last 3 month
payment_ratio_6month: payment per bill ratio in the last 6 month
overlimit_percentage: overlimit percentage


```python
user1 = pd.read_csv('./user_base_part1.csv')
user2 = pd.read_csv('./user_base_part2.csv')
transactions = pd.read_csv('./user_transactions.csv')
user_label_branch = pd.read_csv('./user_label_branch_edit.csv')
```


```python
user1.equals(user2)
# user_base_part1 dataframe is exactly the same as user_base_part2
```




    True



## SQL 1


```python
query_combine = """
select 
    label.user_id, label.number_of_cards, label.branch_code, label.default_flag,
    transactions.outstanding, transactions.credit_limit, transactions.bill, transactions.total_cash_usage, transactions.total_retail_usage, transactions.remaining_bill,
    base.payment_ratio, base.overlimit_percentage, base.payment_ratio_3month, base.payment_ratio_6month, base.deliquency_score, base.years_since_card_issuing, base.total_usage,
    base.remaining_bill_per_number_of_cards, base.remaining_bill_per_limit, base.total_usage_per_limit, base.total_3mo_usage_per_limit, base.total_6mo_usage_per_limit,
    base.utilization_3month, base.utilization_6month
from 
    user_label_branch as label
join
    transactions
on 
    label.user_id = transactions.user_id
join
    user_base as base
on 
    label.user_id = base.user_id
"""
combine = pd.read_sql(query_combine, con = psycopg2.connect(database = dbname, user = user))
```


```python
combine.head()
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
      <th>number_of_cards</th>
      <th>branch_code</th>
      <th>default_flag</th>
      <th>outstanding</th>
      <th>credit_limit</th>
      <th>bill</th>
      <th>total_cash_usage</th>
      <th>total_retail_usage</th>
      <th>remaining_bill</th>
      <th>payment_ratio</th>
      <th>overlimit_percentage</th>
      <th>payment_ratio_3month</th>
      <th>payment_ratio_6month</th>
      <th>deliquency_score</th>
      <th>years_since_card_issuing</th>
      <th>total_usage</th>
      <th>remaining_bill_per_number_of_cards</th>
      <th>remaining_bill_per_limit</th>
      <th>total_usage_per_limit</th>
      <th>total_3mo_usage_per_limit</th>
      <th>total_6mo_usage_per_limit</th>
      <th>utilization_3month</th>
      <th>utilization_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>I</td>
      <td>0</td>
      <td>36158.0</td>
      <td>7000000.0</td>
      <td>23437.0</td>
      <td>0.0</td>
      <td>94.0</td>
      <td>26323.0</td>
      <td>102.19</td>
      <td>0.0</td>
      <td>74.78</td>
      <td>100.00</td>
      <td>0.0</td>
      <td>15.416667</td>
      <td>94.0</td>
      <td>13161.5</td>
      <td>0.003760</td>
      <td>0.000013</td>
      <td>0.011719</td>
      <td>0.017810</td>
      <td>0.013228</td>
      <td>0.021949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>268691.0</td>
      <td>10000000.0</td>
      <td>254564.0</td>
      <td>0.0</td>
      <td>1012.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>0.750000</td>
      <td>1012.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000101</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004232</td>
      <td>0.000300</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>A</td>
      <td>0</td>
      <td>6769149.0</td>
      <td>28000000.0</td>
      <td>4159779.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>100.00</td>
      <td>0.0</td>
      <td>100.00</td>
      <td>100.91</td>
      <td>NaN</td>
      <td>10.750000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040518</td>
      <td>0.047703</td>
      <td>0.249389</td>
      <td>0.267853</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
      <td>G</td>
      <td>0</td>
      <td>3496732.0</td>
      <td>21000000.0</td>
      <td>111231.0</td>
      <td>0.0</td>
      <td>2536660.0</td>
      <td>581334.0</td>
      <td>100.00</td>
      <td>0.0</td>
      <td>25.01</td>
      <td>22.64</td>
      <td>0.0</td>
      <td>19.750000</td>
      <td>2536660.0</td>
      <td>145333.5</td>
      <td>0.027683</td>
      <td>0.120793</td>
      <td>0.055971</td>
      <td>0.016851</td>
      <td>0.101912</td>
      <td>0.346635</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
      <td>A</td>
      <td>0</td>
      <td>9402085.0</td>
      <td>10000000.0</td>
      <td>6099283.0</td>
      <td>0.0</td>
      <td>2666558.0</td>
      <td>5951865.0</td>
      <td>95.99</td>
      <td>0.0</td>
      <td>97.49</td>
      <td>99.84</td>
      <td>NaN</td>
      <td>1.666667</td>
      <td>2666558.0</td>
      <td>2975932.5</td>
      <td>0.595186</td>
      <td>0.266656</td>
      <td>0.323027</td>
      <td>0.131162</td>
      <td>0.707865</td>
      <td>0.336571</td>
    </tr>
  </tbody>
</table>
</div>



## SQL 2


```python
query_default_stats = """
with main as (
select 
    label.user_id, label.number_of_cards, label.branch_code, label.default_flag,
    transactions.outstanding, transactions.credit_limit, transactions.bill, transactions.total_cash_usage, transactions.total_retail_usage, transactions.remaining_bill,
    base.payment_ratio, base.overlimit_percentage, base.payment_ratio_3month, base.payment_ratio_6month, base.deliquency_score, base.years_since_card_issuing, base.total_usage,
    base.remaining_bill_per_number_of_cards, base.remaining_bill_per_limit, base.total_usage_per_limit, base.total_3mo_usage_per_limit, base.total_6mo_usage_per_limit,
    base.utilization_3month, base.utilization_6month
from 
    user_label_branch as label
join
    transactions
on 
    label.user_id = transactions.user_id
join
    user_base as base
on 
    label.user_id = base.user_id
)
select
    default_flag,
    branch_code,
    count(user_id) as user_count,
    avg(deliquency_score) as avg_deliquency_score,
    percentile_disc(0.5) within group (order by deliquency_score) as median_deliquency_score,
    avg(number_of_cards) as avg_no_cards,
    percentile_disc(0.5) within group (order by number_of_cards) as median_no_cards,
    avg(years_since_card_issuing) as avg_years_since_card_issuing,
    percentile_disc(0.5) within group (order by years_since_card_issuing) as median_years_since_card_issuing,
    avg(outstanding) as avg_outstanding,
    percentile_disc(0.5) within group (order by outstanding) as median_outstanding,
    avg(credit_limit) as avg_credit_limit,
    percentile_disc(0.5) within group (order by credit_limit) as median_credit_limit,
    avg(bill) as avg_bill,
    percentile_disc(0.5) within group (order by bill) as median_bill,
    avg(remaining_bill) as avg_remaining_bill,
    percentile_disc(0.5) within group (order by remaining_bill) as median_remaining_bill,
    avg(remaining_bill_per_number_of_cards) as avg_remaining_bill_per_card,
    percentile_disc(0.5) within group (order by remaining_bill_per_number_of_cards) as median_remaining_bill_per_number_of_cards,
    avg(remaining_bill_per_limit) as avg_remaining_bill_per_limit,
    percentile_disc(0.5) within group (order by remaining_bill_per_limit) as median_remaining_bill_per_limit,
    avg(total_cash_usage) as avg_total_cash_usage,
    percentile_disc(0.5) within group (order by total_cash_usage) as median_total_cash_usage,
    avg(total_retail_usage) as avg_total_retail_usage,
    percentile_disc(0.5) within group (order by total_retail_usage) as median_total_retail_usage,
    avg(total_usage) as avg_total_usage,
    percentile_disc(0.5) within group (order by total_usage) as median_total_usage,
    avg(total_usage_per_limit) as avg_total_usage_per_limit,
    percentile_disc(0.5) within group (order by total_usage_per_limit) as median_total_usage_per_limit,
    avg(total_3mo_usage_per_limit) as avg_total_3mo_usage_per_limit,
    percentile_disc(0.5) within group (order by total_3mo_usage_per_limit) as median_total_3mo_usage_per_limit,
    avg(total_6mo_usage_per_limit) as avg_total_6mo_usage_per_limit,
    percentile_disc(0.5) within group (order by total_6mo_usage_per_limit) as median_total_6mo_usage_per_limit,
    avg(payment_ratio) as avg_payment_ratio,
    percentile_disc(0.5) within group (order by payment_ratio) as median_payment_ratio,
    avg(payment_ratio_3month) as avg_payment_ratio_3mo,
    percentile_disc(0.5) within group (order by payment_ratio_3month) as median_payment_ratio_3mo,
    avg(payment_ratio_6month) as avg_payment_ratio_6mo,
    percentile_disc(0.5) within group (order by payment_ratio_6month) as median_payment_ratio_6mo,
    avg(overlimit_percentage) as avg_overlimit_percentage,
    percentile_disc(0.5) within group (order by overlimit_percentage) as median_overlimit_percentage,
    avg(utilization_3month) as avg_3mo_utilization,
    percentile_disc(0.5) within group (order by utilization_3month) as median_3mo_utilization,
    avg(utilization_6month) as avg_6mo_utilization,
    percentile_disc(0.5) within group (order by utilization_6month) as median_6mo_utilization
from main
group by 1, 2
order by 2, 1
"""
default_stats = pd.read_sql(query_default_stats, con = psycopg2.connect(database = dbname, user = user))
# can use min, max, percentile_disc(0.25) and percentile_disc(0.75) functions as well
```


```python
default_stats
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
      <th>branch_code</th>
      <th>user_count</th>
      <th>avg_deliquency_score</th>
      <th>median_deliquency_score</th>
      <th>avg_no_cards</th>
      <th>median_no_cards</th>
      <th>avg_years_since_card_issuing</th>
      <th>median_years_since_card_issuing</th>
      <th>avg_outstanding</th>
      <th>median_outstanding</th>
      <th>avg_credit_limit</th>
      <th>median_credit_limit</th>
      <th>avg_bill</th>
      <th>median_bill</th>
      <th>avg_remaining_bill</th>
      <th>median_remaining_bill</th>
      <th>avg_remaining_bill_per_card</th>
      <th>median_remaining_bill_per_number_of_cards</th>
      <th>avg_remaining_bill_per_limit</th>
      <th>median_remaining_bill_per_limit</th>
      <th>avg_total_cash_usage</th>
      <th>median_total_cash_usage</th>
      <th>avg_total_retail_usage</th>
      <th>median_total_retail_usage</th>
      <th>avg_total_usage</th>
      <th>median_total_usage</th>
      <th>avg_total_usage_per_limit</th>
      <th>median_total_usage_per_limit</th>
      <th>avg_total_3mo_usage_per_limit</th>
      <th>median_total_3mo_usage_per_limit</th>
      <th>avg_total_6mo_usage_per_limit</th>
      <th>median_total_6mo_usage_per_limit</th>
      <th>avg_payment_ratio</th>
      <th>median_payment_ratio</th>
      <th>avg_payment_ratio_3mo</th>
      <th>median_payment_ratio_3mo</th>
      <th>avg_payment_ratio_6mo</th>
      <th>median_payment_ratio_6mo</th>
      <th>avg_overlimit_percentage</th>
      <th>median_overlimit_percentage</th>
      <th>avg_3mo_utilization</th>
      <th>median_3mo_utilization</th>
      <th>avg_6mo_utilization</th>
      <th>median_6mo_utilization</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>A</td>
      <td>7926</td>
      <td>0.001020</td>
      <td>0.0</td>
      <td>2.556523</td>
      <td>2</td>
      <td>6.801886</td>
      <td>5.750000</td>
      <td>1.287613e+07</td>
      <td>5502098.0</td>
      <td>2.475858e+07</td>
      <td>11000000.0</td>
      <td>8.833918e+06</td>
      <td>3564668.0</td>
      <td>8.689746e+06</td>
      <td>2943538.0</td>
      <td>3.162557e+06</td>
      <td>1265266.000</td>
      <td>0.435955</td>
      <td>0.268000</td>
      <td>72852.422786</td>
      <td>0.0</td>
      <td>2.550122e+06</td>
      <td>386423.0</td>
      <td>2.623016e+06</td>
      <td>437000.0</td>
      <td>0.110354</td>
      <td>0.034400</td>
      <td>0.159605</td>
      <td>0.103000</td>
      <td>0.185342</td>
      <td>0.110000</td>
      <td>68.618142</td>
      <td>36.01</td>
      <td>57.181604</td>
      <td>54.30</td>
      <td>84.539334</td>
      <td>66.70</td>
      <td>2.645548</td>
      <td>0.00</td>
      <td>0.543604</td>
      <td>0.526000</td>
      <td>0.506983</td>
      <td>0.453000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>A</td>
      <td>767</td>
      <td>0.035202</td>
      <td>0.0</td>
      <td>2.646675</td>
      <td>2</td>
      <td>6.834343</td>
      <td>5.330000</td>
      <td>2.221352e+07</td>
      <td>6581960.0</td>
      <td>2.861799e+07</td>
      <td>13000000.0</td>
      <td>1.715552e+07</td>
      <td>5527047.0</td>
      <td>1.924337e+07</td>
      <td>5708855.0</td>
      <td>6.428818e+06</td>
      <td>2675706.000</td>
      <td>0.693199</td>
      <td>0.850000</td>
      <td>112908.829205</td>
      <td>0.0</td>
      <td>7.822129e+05</td>
      <td>0.0</td>
      <td>8.951217e+05</td>
      <td>0.0</td>
      <td>0.034665</td>
      <td>0.000000</td>
      <td>0.125564</td>
      <td>0.054800</td>
      <td>0.223138</td>
      <td>0.096200</td>
      <td>18.657862</td>
      <td>0.00</td>
      <td>15.137184</td>
      <td>20.20</td>
      <td>37.914003</td>
      <td>29.10</td>
      <td>8.125671</td>
      <td>0.00</td>
      <td>0.720949</td>
      <td>0.868000</td>
      <td>0.682328</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>B</td>
      <td>1216</td>
      <td>0.000822</td>
      <td>0.0</td>
      <td>2.438322</td>
      <td>2</td>
      <td>6.441246</td>
      <td>5.500000</td>
      <td>8.524250e+06</td>
      <td>3392930.0</td>
      <td>1.477508e+07</td>
      <td>6000000.0</td>
      <td>6.100867e+06</td>
      <td>2480236.0</td>
      <td>5.376491e+06</td>
      <td>2061332.0</td>
      <td>2.064334e+06</td>
      <td>943965.000</td>
      <td>0.459918</td>
      <td>0.329000</td>
      <td>138006.226974</td>
      <td>0.0</td>
      <td>1.878351e+06</td>
      <td>150000.0</td>
      <td>2.016488e+06</td>
      <td>192000.0</td>
      <td>0.129818</td>
      <td>0.024300</td>
      <td>0.184489</td>
      <td>0.122821</td>
      <td>0.219093</td>
      <td>0.121000</td>
      <td>56.624359</td>
      <td>25.10</td>
      <td>72.476727</td>
      <td>52.20</td>
      <td>78.262714</td>
      <td>66.70</td>
      <td>3.223873</td>
      <td>0.00</td>
      <td>0.562567</td>
      <td>0.575000</td>
      <td>0.530451</td>
      <td>0.511000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>B</td>
      <td>132</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.424242</td>
      <td>2</td>
      <td>6.464015</td>
      <td>5.000000</td>
      <td>9.706657e+06</td>
      <td>3586261.0</td>
      <td>1.521970e+07</td>
      <td>6000000.0</td>
      <td>7.930416e+06</td>
      <td>3338474.0</td>
      <td>7.824128e+06</td>
      <td>3232414.0</td>
      <td>2.864681e+06</td>
      <td>1477983.000</td>
      <td>0.604379</td>
      <td>0.747779</td>
      <td>129166.666667</td>
      <td>0.0</td>
      <td>4.758968e+05</td>
      <td>0.0</td>
      <td>6.050635e+05</td>
      <td>0.0</td>
      <td>0.030750</td>
      <td>0.000000</td>
      <td>0.127516</td>
      <td>0.057919</td>
      <td>0.170329</td>
      <td>0.086800</td>
      <td>16.732273</td>
      <td>0.00</td>
      <td>35.322955</td>
      <td>25.00</td>
      <td>49.792879</td>
      <td>33.30</td>
      <td>7.446591</td>
      <td>0.00</td>
      <td>0.623807</td>
      <td>0.748000</td>
      <td>0.573085</td>
      <td>0.537000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>C</td>
      <td>305</td>
      <td>0.006557</td>
      <td>0.0</td>
      <td>2.413115</td>
      <td>2</td>
      <td>6.445442</td>
      <td>5.670000</td>
      <td>7.265405e+06</td>
      <td>3632373.0</td>
      <td>1.496066e+07</td>
      <td>7000000.0</td>
      <td>5.674350e+06</td>
      <td>3186854.0</td>
      <td>5.316427e+06</td>
      <td>2589638.0</td>
      <td>2.162015e+06</td>
      <td>1178209.000</td>
      <td>0.468860</td>
      <td>0.308000</td>
      <td>34081.967213</td>
      <td>0.0</td>
      <td>1.169826e+06</td>
      <td>272379.0</td>
      <td>1.203908e+06</td>
      <td>307820.0</td>
      <td>0.113672</td>
      <td>0.032000</td>
      <td>0.181225</td>
      <td>0.130000</td>
      <td>0.217267</td>
      <td>0.151000</td>
      <td>64.686754</td>
      <td>31.00</td>
      <td>64.156951</td>
      <td>53.40</td>
      <td>68.493443</td>
      <td>66.70</td>
      <td>2.999803</td>
      <td>0.00</td>
      <td>0.567405</td>
      <td>0.599381</td>
      <td>0.530460</td>
      <td>0.515000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>C</td>
      <td>27</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.296296</td>
      <td>2</td>
      <td>5.812222</td>
      <td>5.250000</td>
      <td>8.422023e+06</td>
      <td>4027195.0</td>
      <td>1.385185e+07</td>
      <td>7000000.0</td>
      <td>7.070592e+06</td>
      <td>3487608.0</td>
      <td>6.811747e+06</td>
      <td>3682477.0</td>
      <td>2.682734e+06</td>
      <td>1841239.000</td>
      <td>0.689242</td>
      <td>1.040000</td>
      <td>35185.185185</td>
      <td>0.0</td>
      <td>8.090296e+05</td>
      <td>0.0</td>
      <td>8.442147e+05</td>
      <td>0.0</td>
      <td>0.056509</td>
      <td>0.000000</td>
      <td>0.143865</td>
      <td>0.081300</td>
      <td>0.229708</td>
      <td>0.146000</td>
      <td>17.541481</td>
      <td>0.00</td>
      <td>56.719630</td>
      <td>16.91</td>
      <td>86.610370</td>
      <td>46.20</td>
      <td>8.276296</td>
      <td>4.36</td>
      <td>0.725468</td>
      <td>0.927000</td>
      <td>0.651004</td>
      <td>0.819000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>D</td>
      <td>193</td>
      <td>0.005181</td>
      <td>0.0</td>
      <td>2.466321</td>
      <td>2</td>
      <td>6.830174</td>
      <td>5.670000</td>
      <td>8.158524e+06</td>
      <td>4645467.0</td>
      <td>1.459585e+07</td>
      <td>7000000.0</td>
      <td>5.954875e+06</td>
      <td>2408082.0</td>
      <td>6.327167e+06</td>
      <td>2739377.0</td>
      <td>2.335120e+06</td>
      <td>1280569.000</td>
      <td>0.466917</td>
      <td>0.406000</td>
      <td>37046.632124</td>
      <td>0.0</td>
      <td>9.026852e+05</td>
      <td>96000.0</td>
      <td>9.397319e+05</td>
      <td>100000.0</td>
      <td>0.098800</td>
      <td>0.009030</td>
      <td>0.172793</td>
      <td>0.098945</td>
      <td>0.237442</td>
      <td>0.138000</td>
      <td>68.188290</td>
      <td>28.80</td>
      <td>51.799171</td>
      <td>49.70</td>
      <td>54.455544</td>
      <td>60.80</td>
      <td>2.894611</td>
      <td>0.00</td>
      <td>0.577920</td>
      <td>0.640000</td>
      <td>0.572731</td>
      <td>0.588000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>D</td>
      <td>12</td>
      <td>3.000000</td>
      <td>3.0</td>
      <td>2.500000</td>
      <td>2</td>
      <td>6.553306</td>
      <td>4.666667</td>
      <td>5.998939e+06</td>
      <td>3929654.0</td>
      <td>1.458333e+07</td>
      <td>5000000.0</td>
      <td>5.686584e+06</td>
      <td>4179179.0</td>
      <td>5.003234e+06</td>
      <td>3661225.0</td>
      <td>1.699731e+06</td>
      <td>1830613.000</td>
      <td>0.586711</td>
      <td>0.732000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.500417e+05</td>
      <td>0.0</td>
      <td>2.500417e+05</td>
      <td>0.0</td>
      <td>0.011335</td>
      <td>0.000000</td>
      <td>0.061745</td>
      <td>0.005950</td>
      <td>0.084252</td>
      <td>0.043800</td>
      <td>28.376667</td>
      <td>10.10</td>
      <td>24.955833</td>
      <td>14.10</td>
      <td>38.505833</td>
      <td>12.20</td>
      <td>1.976667</td>
      <td>0.00</td>
      <td>0.629502</td>
      <td>0.801000</td>
      <td>0.676764</td>
      <td>0.846000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>E</td>
      <td>533</td>
      <td>0.013133</td>
      <td>0.0</td>
      <td>2.439024</td>
      <td>2</td>
      <td>6.139134</td>
      <td>5.250000</td>
      <td>8.720811e+06</td>
      <td>3891908.0</td>
      <td>1.536210e+07</td>
      <td>6000000.0</td>
      <td>6.559310e+06</td>
      <td>3192305.0</td>
      <td>6.404521e+06</td>
      <td>2906328.0</td>
      <td>2.477430e+06</td>
      <td>1376924.000</td>
      <td>0.540825</td>
      <td>0.473000</td>
      <td>73245.778612</td>
      <td>0.0</td>
      <td>1.468568e+06</td>
      <td>161487.0</td>
      <td>1.541782e+06</td>
      <td>212000.0</td>
      <td>0.120858</td>
      <td>0.025100</td>
      <td>0.186145</td>
      <td>0.119000</td>
      <td>0.224260</td>
      <td>0.140000</td>
      <td>45.786998</td>
      <td>19.50</td>
      <td>22.693771</td>
      <td>49.10</td>
      <td>-79.521614</td>
      <td>51.80</td>
      <td>5.268762</td>
      <td>0.00</td>
      <td>0.608025</td>
      <td>0.619000</td>
      <td>0.565686</td>
      <td>0.531000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>E</td>
      <td>114</td>
      <td>1.728070</td>
      <td>0.0</td>
      <td>2.385965</td>
      <td>2</td>
      <td>6.542649</td>
      <td>6.000000</td>
      <td>1.414213e+07</td>
      <td>2063070.0</td>
      <td>1.836842e+07</td>
      <td>6000000.0</td>
      <td>1.038201e+07</td>
      <td>2011535.0</td>
      <td>1.239350e+07</td>
      <td>1557436.0</td>
      <td>3.984641e+06</td>
      <td>778718.000</td>
      <td>0.478849</td>
      <td>0.287000</td>
      <td>61403.508772</td>
      <td>0.0</td>
      <td>2.804506e+05</td>
      <td>0.0</td>
      <td>3.418541e+05</td>
      <td>0.0</td>
      <td>0.024669</td>
      <td>0.000000</td>
      <td>0.094704</td>
      <td>0.007040</td>
      <td>0.208226</td>
      <td>0.038300</td>
      <td>9.316842</td>
      <td>0.00</td>
      <td>23.516491</td>
      <td>20.70</td>
      <td>48.705614</td>
      <td>30.70</td>
      <td>3.608864</td>
      <td>0.00</td>
      <td>0.531628</td>
      <td>0.404000</td>
      <td>0.521024</td>
      <td>0.576000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>F</td>
      <td>1551</td>
      <td>0.001938</td>
      <td>0.0</td>
      <td>2.470664</td>
      <td>2</td>
      <td>6.588155</td>
      <td>5.750000</td>
      <td>9.654000e+06</td>
      <td>3572509.0</td>
      <td>1.679400e+07</td>
      <td>7000000.0</td>
      <td>6.141305e+06</td>
      <td>2547440.0</td>
      <td>6.392245e+06</td>
      <td>1784757.0</td>
      <td>2.365171e+06</td>
      <td>828836.000</td>
      <td>0.438022</td>
      <td>0.273536</td>
      <td>69082.223082</td>
      <td>0.0</td>
      <td>1.730743e+06</td>
      <td>208000.0</td>
      <td>1.799825e+06</td>
      <td>248000.0</td>
      <td>0.128719</td>
      <td>0.024900</td>
      <td>0.187987</td>
      <td>0.127417</td>
      <td>0.217808</td>
      <td>0.133000</td>
      <td>114.249800</td>
      <td>42.80</td>
      <td>28.616312</td>
      <td>55.90</td>
      <td>109.774404</td>
      <td>66.70</td>
      <td>3.077789</td>
      <td>0.00</td>
      <td>0.563470</td>
      <td>0.558000</td>
      <td>0.524712</td>
      <td>0.491000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>F</td>
      <td>143</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.692308</td>
      <td>2</td>
      <td>6.700821</td>
      <td>5.580000</td>
      <td>2.222779e+07</td>
      <td>5380189.0</td>
      <td>2.322378e+07</td>
      <td>8000000.0</td>
      <td>1.516696e+07</td>
      <td>5127044.0</td>
      <td>2.025488e+07</td>
      <td>5197174.0</td>
      <td>6.375708e+06</td>
      <td>2411509.000</td>
      <td>0.769113</td>
      <td>0.971026</td>
      <td>151398.601399</td>
      <td>0.0</td>
      <td>1.529138e+06</td>
      <td>0.0</td>
      <td>1.680537e+06</td>
      <td>0.0</td>
      <td>0.030258</td>
      <td>0.000000</td>
      <td>0.137041</td>
      <td>0.095486</td>
      <td>0.229040</td>
      <td>0.114000</td>
      <td>-0.328881</td>
      <td>0.00</td>
      <td>19.497273</td>
      <td>21.70</td>
      <td>16.845874</td>
      <td>25.50</td>
      <td>8.008951</td>
      <td>2.89</td>
      <td>0.796538</td>
      <td>0.934000</td>
      <td>0.735313</td>
      <td>0.862000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>G</td>
      <td>536</td>
      <td>0.001866</td>
      <td>0.0</td>
      <td>2.373134</td>
      <td>2</td>
      <td>6.062058</td>
      <td>5.420000</td>
      <td>6.655654e+06</td>
      <td>3439701.0</td>
      <td>1.398787e+07</td>
      <td>6000000.0</td>
      <td>4.377028e+06</td>
      <td>2156970.0</td>
      <td>3.973315e+06</td>
      <td>1120823.0</td>
      <td>1.498267e+06</td>
      <td>487031.000</td>
      <td>0.395261</td>
      <td>0.129000</td>
      <td>30701.716418</td>
      <td>0.0</td>
      <td>1.690657e+06</td>
      <td>373920.0</td>
      <td>1.721359e+06</td>
      <td>384488.0</td>
      <td>0.141053</td>
      <td>0.054000</td>
      <td>0.203421</td>
      <td>0.153715</td>
      <td>0.223329</td>
      <td>0.155000</td>
      <td>67.577052</td>
      <td>99.00</td>
      <td>212.179907</td>
      <td>70.60</td>
      <td>198.343955</td>
      <td>66.70</td>
      <td>2.580802</td>
      <td>0.00</td>
      <td>0.522593</td>
      <td>0.485335</td>
      <td>0.495616</td>
      <td>0.421000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>G</td>
      <td>30</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.833333</td>
      <td>3</td>
      <td>6.603656</td>
      <td>5.670000</td>
      <td>1.675818e+07</td>
      <td>6522120.0</td>
      <td>1.806667e+07</td>
      <td>7000000.0</td>
      <td>1.454119e+07</td>
      <td>6305666.0</td>
      <td>1.526482e+07</td>
      <td>6361390.0</td>
      <td>5.234407e+06</td>
      <td>2976937.500</td>
      <td>0.842753</td>
      <td>0.909000</td>
      <td>186666.666667</td>
      <td>0.0</td>
      <td>1.035014e+06</td>
      <td>0.0</td>
      <td>1.221681e+06</td>
      <td>0.0</td>
      <td>0.049467</td>
      <td>0.000000</td>
      <td>0.199896</td>
      <td>0.114000</td>
      <td>0.304692</td>
      <td>0.127000</td>
      <td>23.623000</td>
      <td>9.64</td>
      <td>27.204667</td>
      <td>21.10</td>
      <td>46.819333</td>
      <td>36.00</td>
      <td>11.679667</td>
      <td>6.36</td>
      <td>0.863436</td>
      <td>0.948167</td>
      <td>0.778723</td>
      <td>0.822170</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>H</td>
      <td>334</td>
      <td>0.003003</td>
      <td>0.0</td>
      <td>2.392216</td>
      <td>2</td>
      <td>6.507295</td>
      <td>5.580000</td>
      <td>7.721575e+06</td>
      <td>3298004.0</td>
      <td>1.476198e+07</td>
      <td>6000000.0</td>
      <td>6.097793e+06</td>
      <td>2548091.0</td>
      <td>6.054229e+06</td>
      <td>1836264.0</td>
      <td>2.250804e+06</td>
      <td>856016.000</td>
      <td>0.456134</td>
      <td>0.287000</td>
      <td>45808.383234</td>
      <td>0.0</td>
      <td>1.383473e+06</td>
      <td>87340.0</td>
      <td>1.429281e+06</td>
      <td>105000.0</td>
      <td>0.106692</td>
      <td>0.008670</td>
      <td>0.178071</td>
      <td>0.121000</td>
      <td>0.221791</td>
      <td>0.134000</td>
      <td>47.506287</td>
      <td>24.30</td>
      <td>50.694371</td>
      <td>50.00</td>
      <td>211.128952</td>
      <td>66.70</td>
      <td>3.193503</td>
      <td>0.00</td>
      <td>0.576439</td>
      <td>0.592000</td>
      <td>0.534470</td>
      <td>0.500000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>H</td>
      <td>33</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.333333</td>
      <td>2</td>
      <td>6.007182</td>
      <td>5.580000</td>
      <td>6.600326e+06</td>
      <td>3980541.0</td>
      <td>1.281818e+07</td>
      <td>5000000.0</td>
      <td>5.190916e+06</td>
      <td>3346949.0</td>
      <td>5.480239e+06</td>
      <td>3300972.0</td>
      <td>2.349247e+06</td>
      <td>1650486.000</td>
      <td>0.587898</td>
      <td>0.655000</td>
      <td>30303.030303</td>
      <td>0.0</td>
      <td>1.661576e+05</td>
      <td>0.0</td>
      <td>1.964606e+05</td>
      <td>0.0</td>
      <td>0.017353</td>
      <td>0.000000</td>
      <td>0.120362</td>
      <td>0.078000</td>
      <td>0.225618</td>
      <td>0.133000</td>
      <td>13.295455</td>
      <td>0.00</td>
      <td>7.876364</td>
      <td>22.40</td>
      <td>38.751212</td>
      <td>29.10</td>
      <td>7.380909</td>
      <td>2.22</td>
      <td>0.641307</td>
      <td>0.902000</td>
      <td>0.637781</td>
      <td>0.790000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>I</td>
      <td>888</td>
      <td>0.003378</td>
      <td>0.0</td>
      <td>2.388514</td>
      <td>2</td>
      <td>6.188920</td>
      <td>5.000000</td>
      <td>8.232260e+06</td>
      <td>3872598.0</td>
      <td>1.301014e+07</td>
      <td>5000000.0</td>
      <td>5.472819e+06</td>
      <td>2788118.0</td>
      <td>5.950464e+06</td>
      <td>2909645.0</td>
      <td>2.317086e+06</td>
      <td>1294819.000</td>
      <td>0.536820</td>
      <td>0.536000</td>
      <td>38998.667793</td>
      <td>0.0</td>
      <td>1.457592e+06</td>
      <td>151028.0</td>
      <td>1.496590e+06</td>
      <td>182512.0</td>
      <td>0.122945</td>
      <td>0.020500</td>
      <td>0.191180</td>
      <td>0.123000</td>
      <td>0.231331</td>
      <td>0.127672</td>
      <td>49.628739</td>
      <td>25.60</td>
      <td>-9.406813</td>
      <td>50.00</td>
      <td>64.256667</td>
      <td>61.40</td>
      <td>3.834020</td>
      <td>0.00</td>
      <td>0.653151</td>
      <td>0.726000</td>
      <td>0.600227</td>
      <td>0.604000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>I</td>
      <td>102</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.470588</td>
      <td>2</td>
      <td>5.469304</td>
      <td>4.416667</td>
      <td>1.442864e+07</td>
      <td>5213854.0</td>
      <td>1.631373e+07</td>
      <td>7000000.0</td>
      <td>1.090356e+07</td>
      <td>3884657.0</td>
      <td>1.273640e+07</td>
      <td>4687506.0</td>
      <td>4.614816e+06</td>
      <td>2081872.000</td>
      <td>0.748662</td>
      <td>0.923000</td>
      <td>57843.137255</td>
      <td>0.0</td>
      <td>4.086566e+05</td>
      <td>0.0</td>
      <td>4.664998e+05</td>
      <td>0.0</td>
      <td>0.036940</td>
      <td>0.000000</td>
      <td>0.126805</td>
      <td>0.065400</td>
      <td>0.218363</td>
      <td>0.109000</td>
      <td>18.702255</td>
      <td>0.00</td>
      <td>29.433529</td>
      <td>18.50</td>
      <td>44.420784</td>
      <td>28.10</td>
      <td>6.802255</td>
      <td>0.54</td>
      <td>0.758098</td>
      <td>0.915000</td>
      <td>0.679303</td>
      <td>0.794751</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>J</td>
      <td>377</td>
      <td>0.007958</td>
      <td>0.0</td>
      <td>2.403183</td>
      <td>2</td>
      <td>7.675767</td>
      <td>6.670000</td>
      <td>7.245646e+06</td>
      <td>4128545.0</td>
      <td>1.382759e+07</td>
      <td>7000000.0</td>
      <td>5.132854e+06</td>
      <td>2975039.0</td>
      <td>4.664065e+06</td>
      <td>2306840.0</td>
      <td>1.879326e+06</td>
      <td>1000000.000</td>
      <td>0.453706</td>
      <td>0.330000</td>
      <td>47745.358090</td>
      <td>0.0</td>
      <td>1.331914e+06</td>
      <td>98066.0</td>
      <td>1.379659e+06</td>
      <td>112426.0</td>
      <td>0.107233</td>
      <td>0.013800</td>
      <td>0.182236</td>
      <td>0.112000</td>
      <td>0.224593</td>
      <td>0.134000</td>
      <td>49.541777</td>
      <td>32.10</td>
      <td>-51.512414</td>
      <td>53.10</td>
      <td>64.616870</td>
      <td>66.70</td>
      <td>3.021141</td>
      <td>0.00</td>
      <td>0.581383</td>
      <td>0.580000</td>
      <td>0.549552</td>
      <td>0.520000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>J</td>
      <td>32</td>
      <td>4.468750</td>
      <td>4.0</td>
      <td>2.281250</td>
      <td>2</td>
      <td>8.017187</td>
      <td>7.500000</td>
      <td>2.449283e+06</td>
      <td>667120.0</td>
      <td>9.406250e+06</td>
      <td>5000000.0</td>
      <td>1.815114e+06</td>
      <td>284964.0</td>
      <td>2.018201e+06</td>
      <td>232897.0</td>
      <td>9.342067e+05</td>
      <td>86788.000</td>
      <td>0.296930</td>
      <td>0.034688</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.635156e+05</td>
      <td>0.0</td>
      <td>3.041406e+05</td>
      <td>0.0</td>
      <td>0.057047</td>
      <td>0.000000</td>
      <td>0.062017</td>
      <td>0.000129</td>
      <td>0.270922</td>
      <td>0.009930</td>
      <td>16.563750</td>
      <td>0.00</td>
      <td>28.415000</td>
      <td>14.75</td>
      <td>71.181875</td>
      <td>14.80</td>
      <td>1.339063</td>
      <td>0.00</td>
      <td>0.370655</td>
      <td>0.146000</td>
      <td>0.501434</td>
      <td>0.159000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>K</td>
      <td>157</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.324841</td>
      <td>2</td>
      <td>4.828433</td>
      <td>4.170000</td>
      <td>8.499764e+06</td>
      <td>3487698.0</td>
      <td>1.431847e+07</td>
      <td>8000000.0</td>
      <td>4.734344e+06</td>
      <td>1762184.0</td>
      <td>4.937736e+06</td>
      <td>847066.0</td>
      <td>2.070711e+06</td>
      <td>423533.000</td>
      <td>0.390963</td>
      <td>0.144000</td>
      <td>46974.522293</td>
      <td>0.0</td>
      <td>2.510065e+06</td>
      <td>300000.0</td>
      <td>2.557039e+06</td>
      <td>303000.0</td>
      <td>0.156043</td>
      <td>0.045500</td>
      <td>0.183819</td>
      <td>0.132000</td>
      <td>0.206226</td>
      <td>0.135375</td>
      <td>63.753439</td>
      <td>99.80</td>
      <td>104.134140</td>
      <td>62.30</td>
      <td>123.014841</td>
      <td>68.20</td>
      <td>2.443376</td>
      <td>0.00</td>
      <td>0.517623</td>
      <td>0.517000</td>
      <td>0.464164</td>
      <td>0.380165</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>K</td>
      <td>17</td>
      <td>4.882353</td>
      <td>5.0</td>
      <td>2.411765</td>
      <td>2</td>
      <td>5.780412</td>
      <td>4.920000</td>
      <td>9.153972e+06</td>
      <td>4756363.0</td>
      <td>1.411765e+07</td>
      <td>5000000.0</td>
      <td>8.477999e+06</td>
      <td>4350295.0</td>
      <td>7.729623e+06</td>
      <td>3091805.0</td>
      <td>3.363808e+06</td>
      <td>1545903.000</td>
      <td>0.659798</td>
      <td>0.935000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>4.583505e+05</td>
      <td>0.0</td>
      <td>4.583505e+05</td>
      <td>0.0</td>
      <td>0.027350</td>
      <td>0.000000</td>
      <td>0.124619</td>
      <td>0.091400</td>
      <td>0.217316</td>
      <td>0.166000</td>
      <td>22.829412</td>
      <td>0.00</td>
      <td>33.118824</td>
      <td>18.60</td>
      <td>44.078824</td>
      <td>37.40</td>
      <td>3.527647</td>
      <td>0.00</td>
      <td>0.761455</td>
      <td>0.933000</td>
      <td>0.700330</td>
      <td>0.811000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0</td>
      <td>a</td>
      <td>12</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.416667</td>
      <td>2</td>
      <td>5.493056</td>
      <td>3.500000</td>
      <td>8.620119e+06</td>
      <td>4216107.0</td>
      <td>1.241667e+07</td>
      <td>9000000.0</td>
      <td>4.824404e+06</td>
      <td>1733129.0</td>
      <td>7.322296e+06</td>
      <td>2030088.0</td>
      <td>2.691134e+06</td>
      <td>781048.750</td>
      <td>0.438896</td>
      <td>0.205195</td>
      <td>8333.333333</td>
      <td>0.0</td>
      <td>2.591943e+06</td>
      <td>453615.0</td>
      <td>2.600276e+06</td>
      <td>453615.0</td>
      <td>0.152059</td>
      <td>0.114857</td>
      <td>0.170921</td>
      <td>0.154567</td>
      <td>0.123226</td>
      <td>0.080908</td>
      <td>78.003333</td>
      <td>100.00</td>
      <td>66.452500</td>
      <td>61.03</td>
      <td>64.937500</td>
      <td>66.67</td>
      <td>0.325000</td>
      <td>0.00</td>
      <td>0.607713</td>
      <td>0.624839</td>
      <td>0.506037</td>
      <td>0.387658</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0</td>
      <td>b</td>
      <td>3</td>
      <td>0.333333</td>
      <td>0.0</td>
      <td>2.000000</td>
      <td>2</td>
      <td>8.805556</td>
      <td>9.666667</td>
      <td>9.319113e+06</td>
      <td>10655619.0</td>
      <td>1.666667e+07</td>
      <td>10000000.0</td>
      <td>7.137456e+06</td>
      <td>5808204.0</td>
      <td>4.513808e+06</td>
      <td>2751121.0</td>
      <td>2.115963e+06</td>
      <td>2751121.000</td>
      <td>0.286173</td>
      <td>0.308294</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.519933e+06</td>
      <td>0.0</td>
      <td>1.519933e+06</td>
      <td>0.0</td>
      <td>0.151993</td>
      <td>0.000000</td>
      <td>0.119672</td>
      <td>0.098490</td>
      <td>0.058464</td>
      <td>0.040299</td>
      <td>39.440000</td>
      <td>18.32</td>
      <td>25.020000</td>
      <td>18.70</td>
      <td>18.956667</td>
      <td>6.96</td>
      <td>2.863333</td>
      <td>0.00</td>
      <td>0.707964</td>
      <td>0.700497</td>
      <td>0.545148</td>
      <td>0.617280</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0</td>
      <td>c</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>3.000000</td>
      <td>2</td>
      <td>9.333333</td>
      <td>5.333333</td>
      <td>4.960059e+06</td>
      <td>3453946.0</td>
      <td>2.400000e+07</td>
      <td>5000000.0</td>
      <td>2.668120e+06</td>
      <td>1876878.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>3.355000e+05</td>
      <td>0.0</td>
      <td>3.355000e+05</td>
      <td>0.0</td>
      <td>0.007802</td>
      <td>0.000000</td>
      <td>0.261349</td>
      <td>0.039518</td>
      <td>0.343397</td>
      <td>0.116953</td>
      <td>100.220000</td>
      <td>100.00</td>
      <td>83.185000</td>
      <td>75.00</td>
      <td>109.470000</td>
      <td>100.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.368442</td>
      <td>0.145236</td>
      <td>0.164845</td>
      <td>0.058240</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0</td>
      <td>e</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.000000</td>
      <td>2</td>
      <td>5.208333</td>
      <td>3.583333</td>
      <td>2.651156e+06</td>
      <td>2555490.0</td>
      <td>3.000000e+06</td>
      <td>3000000.0</td>
      <td>1.830246e+06</td>
      <td>987358.0</td>
      <td>1.358706e+06</td>
      <td>0.0</td>
      <td>6.793530e+05</td>
      <td>0.000</td>
      <td>0.452902</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>5.025350e+04</td>
      <td>9937.0</td>
      <td>5.025350e+04</td>
      <td>9937.0</td>
      <td>0.016751</td>
      <td>0.003312</td>
      <td>0.066211</td>
      <td>0.049333</td>
      <td>0.325778</td>
      <td>0.316000</td>
      <td>55.480000</td>
      <td>10.96</td>
      <td>568.000000</td>
      <td>10.19</td>
      <td>6.855000</td>
      <td>-53.96</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.723709</td>
      <td>0.543489</td>
      <td>0.519073</td>
      <td>0.514987</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0</td>
      <td>f</td>
      <td>2</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.500000</td>
      <td>2</td>
      <td>5.416667</td>
      <td>3.500000</td>
      <td>2.247750e+06</td>
      <td>222457.0</td>
      <td>4.750000e+07</td>
      <td>8000000.0</td>
      <td>1.988835e+05</td>
      <td>143819.0</td>
      <td>1.316705e+05</td>
      <td>0.0</td>
      <td>6.583525e+04</td>
      <td>0.000</td>
      <td>0.001513</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.120095e+06</td>
      <td>127001.0</td>
      <td>2.120095e+06</td>
      <td>127001.0</td>
      <td>0.257804</td>
      <td>0.001460</td>
      <td>0.107205</td>
      <td>0.056081</td>
      <td>0.070133</td>
      <td>0.027234</td>
      <td>100.115000</td>
      <td>100.00</td>
      <td>101.725000</td>
      <td>50.00</td>
      <td>66.665000</td>
      <td>33.33</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.103958</td>
      <td>0.067686</td>
      <td>0.166146</td>
      <td>0.063229</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>f</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>3.000000</td>
      <td>3</td>
      <td>5.416667</td>
      <td>5.416667</td>
      <td>3.409834e+06</td>
      <td>3409834.0</td>
      <td>3.000000e+06</td>
      <td>3000000.0</td>
      <td>3.424633e+06</td>
      <td>3424633.0</td>
      <td>3.342896e+06</td>
      <td>3342896.0</td>
      <td>1.114299e+06</td>
      <td>1114298.667</td>
      <td>1.114299</td>
      <td>1.114299</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.091508</td>
      <td>0.091508</td>
      <td>0.132900</td>
      <td>0.132900</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>18.410000</td>
      <td>18.41</td>
      <td>20.630000</td>
      <td>20.63</td>
      <td>13.620000</td>
      <td>13.62</td>
      <td>1.087057</td>
      <td>1.087057</td>
      <td>1.004675</td>
      <td>1.004675</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0</td>
      <td>i</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.000000</td>
      <td>2</td>
      <td>1.500000</td>
      <td>1.500000</td>
      <td>2.516577e+06</td>
      <td>2516577.0</td>
      <td>3.000000e+06</td>
      <td>3000000.0</td>
      <td>2.444000e+06</td>
      <td>2444000.0</td>
      <td>2.473317e+06</td>
      <td>2473317.0</td>
      <td>1.236658e+06</td>
      <td>1236658.500</td>
      <td>0.824439</td>
      <td>0.824439</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.951056</td>
      <td>0.951056</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>54.810000</td>
      <td>54.81</td>
      <td>100.000000</td>
      <td>100.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.521206</td>
      <td>0.521206</td>
      <td>1.009874</td>
      <td>1.009874</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>j</td>
      <td>1</td>
      <td>5.000000</td>
      <td>5.0</td>
      <td>2.000000</td>
      <td>2</td>
      <td>4.083333</td>
      <td>4.083333</td>
      <td>1.275770e+05</td>
      <td>127577.0</td>
      <td>7.000000e+06</td>
      <td>7000000.0</td>
      <td>1.217350e+05</td>
      <td>121735.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>38.710000</td>
      <td>38.71</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.016969</td>
      <td>0.016969</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0</td>
      <td>k</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.000000</td>
      <td>2</td>
      <td>4.500000</td>
      <td>4.500000</td>
      <td>1.783339e+06</td>
      <td>1783339.0</td>
      <td>3.000000e+06</td>
      <td>3000000.0</td>
      <td>1.550000e+06</td>
      <td>1550000.0</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.412000e+05</td>
      <td>241200.0</td>
      <td>2.412000e+05</td>
      <td>241200.0</td>
      <td>0.080400</td>
      <td>0.080400</td>
      <td>0.239882</td>
      <td>0.239882</td>
      <td>0.363026</td>
      <td>0.363026</td>
      <td>100.050000</td>
      <td>100.05</td>
      <td>100.000000</td>
      <td>100.00</td>
      <td>100.000000</td>
      <td>100.00</td>
      <td>0.000000</td>
      <td>0.00</td>
      <td>0.354937</td>
      <td>0.354937</td>
      <td>0.203476</td>
      <td>0.203476</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0</td>
      <td>None</td>
      <td>190</td>
      <td>0.015789</td>
      <td>0.0</td>
      <td>2.494737</td>
      <td>2</td>
      <td>5.826249</td>
      <td>5.330000</td>
      <td>1.137199e+07</td>
      <td>4499151.0</td>
      <td>2.584211e+07</td>
      <td>13000000.0</td>
      <td>6.331030e+06</td>
      <td>3012700.0</td>
      <td>6.565698e+06</td>
      <td>2106724.0</td>
      <td>2.742107e+06</td>
      <td>904268.000</td>
      <td>0.385517</td>
      <td>0.142000</td>
      <td>67894.736842</td>
      <td>0.0</td>
      <td>2.899940e+06</td>
      <td>75250.0</td>
      <td>2.967835e+06</td>
      <td>75250.0</td>
      <td>0.096722</td>
      <td>0.015825</td>
      <td>0.163448</td>
      <td>0.111000</td>
      <td>0.173740</td>
      <td>0.119000</td>
      <td>63.442000</td>
      <td>99.88</td>
      <td>61.134421</td>
      <td>74.30</td>
      <td>81.440737</td>
      <td>66.70</td>
      <td>1.569474</td>
      <td>0.00</td>
      <td>0.486936</td>
      <td>0.455000</td>
      <td>0.457809</td>
      <td>0.345814</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>None</td>
      <td>5</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.400000</td>
      <td>2</td>
      <td>4.150000</td>
      <td>5.080000</td>
      <td>1.353733e+07</td>
      <td>6936314.0</td>
      <td>2.060000e+07</td>
      <td>10000000.0</td>
      <td>1.194035e+07</td>
      <td>6777660.0</td>
      <td>1.333488e+07</td>
      <td>6826680.0</td>
      <td>5.295947e+06</td>
      <td>3413340.000</td>
      <td>0.812468</td>
      <td>0.975000</td>
      <td>780000.000000</td>
      <td>0.0</td>
      <td>3.563080e+04</td>
      <td>0.0</td>
      <td>8.156308e+05</td>
      <td>43000.0</td>
      <td>0.082934</td>
      <td>0.005370</td>
      <td>0.106486</td>
      <td>0.006070</td>
      <td>0.021151</td>
      <td>0.014556</td>
      <td>6.120000</td>
      <td>0.00</td>
      <td>18.138000</td>
      <td>18.10</td>
      <td>22.376000</td>
      <td>23.10</td>
      <td>11.946000</td>
      <td>0.00</td>
      <td>0.758039</td>
      <td>0.994000</td>
      <td>0.869850</td>
      <td>0.790000</td>
    </tr>
  </tbody>
</table>
</div>



## SQL 3


```python
# check branch_code variant
query_check_bc = """
select distinct
    branch_code
from user_label_branch
"""
check_branch_code = pd.read_sql(query_check_bc, con = psycopg2.connect(database = dbname, user = user))
```


```python
check_branch_code
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>i</td>
    </tr>
    <tr>
      <th>3</th>
      <td>j</td>
    </tr>
    <tr>
      <th>4</th>
      <td>H</td>
    </tr>
    <tr>
      <th>5</th>
      <td>e</td>
    </tr>
    <tr>
      <th>6</th>
      <td>J</td>
    </tr>
    <tr>
      <th>7</th>
      <td>C</td>
    </tr>
    <tr>
      <th>8</th>
      <td>D</td>
    </tr>
    <tr>
      <th>9</th>
      <td>I</td>
    </tr>
    <tr>
      <th>10</th>
      <td>f</td>
    </tr>
    <tr>
      <th>11</th>
      <td>G</td>
    </tr>
    <tr>
      <th>12</th>
      <td>B</td>
    </tr>
    <tr>
      <th>13</th>
      <td>F</td>
    </tr>
    <tr>
      <th>14</th>
      <td>c</td>
    </tr>
    <tr>
      <th>15</th>
      <td>K</td>
    </tr>
    <tr>
      <th>16</th>
      <td>k</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A</td>
    </tr>
    <tr>
      <th>18</th>
      <td>E</td>
    </tr>
    <tr>
      <th>19</th>
      <td>a</td>
    </tr>
  </tbody>
</table>
</div>



There are null values in branch code and there is a mix of upper and lower case. 


```python
print("number of null values in branch_code column: " + str(len(user_label_branch[user_label_branch.branch_code.isnull()])))
print("% of rows where branch_code is null: " + str(round(len(user_label_branch[user_label_branch.branch_code.isnull()])/user_label_branch.shape[0]*100, 1)) + '%')
```

    number of null values in branch_code column: 195
    % of rows where branch_code is null: 1.2%



```python
# change upper case branch code to lower case
user_label_branch = (
    user_label_branch
    .pipe(lambda x: x.assign(branch_code=x.branch_code.str.lower()))
)
```


```python
user_label_branch.branch_code.describe()
```




    count     15450
    unique       11
    top           a
    freq       8705
    Name: branch_code, dtype: object




```python
# replace null values with 'a' as it occurs the most often
user_label_branch.loc[user_label_branch[user_label_branch.branch_code.isnull()].index.values, 'branch_code'] = 'a'
```


```python
# reload user_label_branch into psql
user_label_branch.to_csv('./user_label_branch_edit.csv', index=False)
pg_load_table('./user_label_branch_edit.csv', 'user_label_branch', dbname, user)
```

    Connecting to Database
    Truncated user_label_branch
    Loaded data into user_label_branch
    DB connection closed.



```python
# use 'usage' as a proxy to number of transactions completed. assuming higher the usage, the more the number of transactions completed
# since 6 months usage data is available, I shall provide the top 5 users with the most number of 'transactions' from each 'branch code'
# to get total_usage_6months, I will multiply 'total_6mo_usage_per_limit' with 'credit_limit'
# check 'total_6mo_usage_per_limit' and 'credit_limit' for anomaly
combine[['total_6mo_usage_per_limit', 'credit_limit']].describe()
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
      <th>total_6mo_usage_per_limit</th>
      <th>credit_limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>15645.000000</td>
      <td>1.564500e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.202454</td>
      <td>2.082010e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.274421</td>
      <td>2.955419e+07</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.569000</td>
      <td>3.000000e+06</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.032600</td>
      <td>5.000000e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.117000</td>
      <td>9.000000e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.285000</td>
      <td>2.200000e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>8.110000</td>
      <td>1.000000e+09</td>
    </tr>
  </tbody>
</table>
</div>




```python
# there are negative values for total_6mo_usage_per_limit
print("number of entries where total_6mo_usage_per_limit is negative: " + str(len(combine.query("total_6mo_usage_per_limit < 0"))))
```

    number of entries where total_6mo_usage_per_limit is negative: 17



```python
combine.query("total_6mo_usage_per_limit < 0")[['total_usage_per_limit', 'total_6mo_usage_per_limit']]
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
      <th>total_6mo_usage_per_limit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>0.000000</td>
      <td>-0.081559</td>
    </tr>
    <tr>
      <th>208</th>
      <td>0.010286</td>
      <td>-0.001212</td>
    </tr>
    <tr>
      <th>1423</th>
      <td>0.000000</td>
      <td>-0.000152</td>
    </tr>
    <tr>
      <th>3371</th>
      <td>0.090400</td>
      <td>-0.012500</td>
    </tr>
    <tr>
      <th>4866</th>
      <td>0.005380</td>
      <td>-0.009210</td>
    </tr>
    <tr>
      <th>5226</th>
      <td>0.007330</td>
      <td>-0.000325</td>
    </tr>
    <tr>
      <th>7482</th>
      <td>0.063800</td>
      <td>-0.037700</td>
    </tr>
    <tr>
      <th>8943</th>
      <td>0.000000</td>
      <td>-0.025600</td>
    </tr>
    <tr>
      <th>9639</th>
      <td>0.000000</td>
      <td>-0.009000</td>
    </tr>
    <tr>
      <th>10122</th>
      <td>0.000000</td>
      <td>-0.137000</td>
    </tr>
    <tr>
      <th>10378</th>
      <td>0.017800</td>
      <td>-0.002090</td>
    </tr>
    <tr>
      <th>10919</th>
      <td>0.009020</td>
      <td>-0.000767</td>
    </tr>
    <tr>
      <th>11364</th>
      <td>0.337000</td>
      <td>-0.569000</td>
    </tr>
    <tr>
      <th>11396</th>
      <td>0.000000</td>
      <td>-0.160000</td>
    </tr>
    <tr>
      <th>14542</th>
      <td>0.012000</td>
      <td>-0.005530</td>
    </tr>
    <tr>
      <th>14696</th>
      <td>0.007700</td>
      <td>-0.000943</td>
    </tr>
    <tr>
      <th>15294</th>
      <td>0.104000</td>
      <td>-0.078200</td>
    </tr>
  </tbody>
</table>
</div>




```python
# assuming there cannot be negative values for 'total_6mo_usage_per_limit
# from the above dataframe, we can see that 'total_usage_per_limit' and 'total_6mo_usage_per_limit are similar, other than the negative sign
# to clean the data we shall take the absolute of existing value
user1 = (
    user1
    .pipe(lambda x: x.assign(total_6mo_usage_per_limit=abs(x.total_6mo_usage_per_limit)))
)
```


```python
# reload user_base into psql
user1.to_csv('./user_base_part1_edit.csv', index=False)
pg_load_table('./user_base_part1_edit.csv', 'user_base', dbname, user)
```

    Connecting to Database
    Truncated user_base
    Loaded data into user_base
    DB connection closed.



```python
query_top5 = """
with main as (
select
    base.user_id,
    base.total_6mo_usage_per_limit * t.credit_limit as total_usage_6month,
    label.branch_code
from user_base as base
join transactions as t
on base.user_id = t.user_id
join user_label_branch as label
on base.user_id = label.user_id
)
select
    branch_code,
    rank,
    user_id,
    total_usage_6month
from
    (select
        user_id,
        total_usage_6month,
        branch_code,
        rank() over (partition by branch_code order by total_usage_6month desc) as rank
    from main
    ) as tmp
where rank <= 5
order by 1, 2
"""
top5 = pd.read_sql(query_top5, con = psycopg2.connect(database = dbname, user = user))
```


```python
top5
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
      <th>rank</th>
      <th>user_id</th>
      <th>total_usage_6month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>1</td>
      <td>9597</td>
      <td>8.570000e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>a</td>
      <td>2</td>
      <td>13236</td>
      <td>4.980000e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>a</td>
      <td>3</td>
      <td>10550</td>
      <td>4.543000e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>a</td>
      <td>4</td>
      <td>11041</td>
      <td>4.541600e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>a</td>
      <td>5</td>
      <td>4056</td>
      <td>4.288000e+08</td>
    </tr>
    <tr>
      <th>5</th>
      <td>b</td>
      <td>1</td>
      <td>14061</td>
      <td>1.434000e+08</td>
    </tr>
    <tr>
      <th>6</th>
      <td>b</td>
      <td>2</td>
      <td>9096</td>
      <td>1.245000e+08</td>
    </tr>
    <tr>
      <th>7</th>
      <td>b</td>
      <td>3</td>
      <td>1442</td>
      <td>7.323522e+07</td>
    </tr>
    <tr>
      <th>8</th>
      <td>b</td>
      <td>4</td>
      <td>15208</td>
      <td>5.187000e+07</td>
    </tr>
    <tr>
      <th>9</th>
      <td>b</td>
      <td>5</td>
      <td>2813</td>
      <td>5.083000e+07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>c</td>
      <td>1</td>
      <td>11551</td>
      <td>2.378500e+07</td>
    </tr>
    <tr>
      <th>11</th>
      <td>c</td>
      <td>2</td>
      <td>13651</td>
      <td>2.320000e+07</td>
    </tr>
    <tr>
      <th>12</th>
      <td>c</td>
      <td>3</td>
      <td>15480</td>
      <td>1.922400e+07</td>
    </tr>
    <tr>
      <th>13</th>
      <td>c</td>
      <td>4</td>
      <td>10617</td>
      <td>1.913500e+07</td>
    </tr>
    <tr>
      <th>14</th>
      <td>c</td>
      <td>5</td>
      <td>12337</td>
      <td>1.568000e+07</td>
    </tr>
    <tr>
      <th>15</th>
      <td>d</td>
      <td>1</td>
      <td>7056</td>
      <td>5.185800e+07</td>
    </tr>
    <tr>
      <th>16</th>
      <td>d</td>
      <td>2</td>
      <td>12704</td>
      <td>2.376000e+07</td>
    </tr>
    <tr>
      <th>17</th>
      <td>d</td>
      <td>3</td>
      <td>3356</td>
      <td>2.014800e+07</td>
    </tr>
    <tr>
      <th>18</th>
      <td>d</td>
      <td>4</td>
      <td>233</td>
      <td>1.815032e+07</td>
    </tr>
    <tr>
      <th>19</th>
      <td>d</td>
      <td>5</td>
      <td>15102</td>
      <td>1.680000e+07</td>
    </tr>
    <tr>
      <th>20</th>
      <td>e</td>
      <td>1</td>
      <td>15640</td>
      <td>3.250500e+07</td>
    </tr>
    <tr>
      <th>21</th>
      <td>e</td>
      <td>1</td>
      <td>13547</td>
      <td>3.250500e+07</td>
    </tr>
    <tr>
      <th>22</th>
      <td>e</td>
      <td>3</td>
      <td>3663</td>
      <td>2.929500e+07</td>
    </tr>
    <tr>
      <th>23</th>
      <td>e</td>
      <td>3</td>
      <td>15599</td>
      <td>2.929500e+07</td>
    </tr>
    <tr>
      <th>24</th>
      <td>e</td>
      <td>5</td>
      <td>10638</td>
      <td>2.764800e+07</td>
    </tr>
    <tr>
      <th>25</th>
      <td>f</td>
      <td>1</td>
      <td>3591</td>
      <td>1.376000e+08</td>
    </tr>
    <tr>
      <th>26</th>
      <td>f</td>
      <td>2</td>
      <td>1182</td>
      <td>1.364048e+08</td>
    </tr>
    <tr>
      <th>27</th>
      <td>f</td>
      <td>3</td>
      <td>3411</td>
      <td>1.047550e+08</td>
    </tr>
    <tr>
      <th>28</th>
      <td>f</td>
      <td>4</td>
      <td>10127</td>
      <td>1.041900e+08</td>
    </tr>
    <tr>
      <th>29</th>
      <td>f</td>
      <td>5</td>
      <td>11066</td>
      <td>9.796000e+07</td>
    </tr>
    <tr>
      <th>30</th>
      <td>g</td>
      <td>1</td>
      <td>1408</td>
      <td>9.456487e+07</td>
    </tr>
    <tr>
      <th>31</th>
      <td>g</td>
      <td>2</td>
      <td>13331</td>
      <td>6.150000e+07</td>
    </tr>
    <tr>
      <th>32</th>
      <td>g</td>
      <td>3</td>
      <td>1157</td>
      <td>3.738000e+07</td>
    </tr>
    <tr>
      <th>33</th>
      <td>g</td>
      <td>4</td>
      <td>13674</td>
      <td>2.988900e+07</td>
    </tr>
    <tr>
      <th>34</th>
      <td>g</td>
      <td>5</td>
      <td>12760</td>
      <td>2.550000e+07</td>
    </tr>
    <tr>
      <th>35</th>
      <td>h</td>
      <td>1</td>
      <td>7050</td>
      <td>1.005000e+08</td>
    </tr>
    <tr>
      <th>36</th>
      <td>h</td>
      <td>2</td>
      <td>4717</td>
      <td>4.825000e+07</td>
    </tr>
    <tr>
      <th>37</th>
      <td>h</td>
      <td>3</td>
      <td>12515</td>
      <td>3.328600e+07</td>
    </tr>
    <tr>
      <th>38</th>
      <td>h</td>
      <td>4</td>
      <td>8077</td>
      <td>2.737800e+07</td>
    </tr>
    <tr>
      <th>39</th>
      <td>h</td>
      <td>5</td>
      <td>4290</td>
      <td>2.215500e+07</td>
    </tr>
    <tr>
      <th>40</th>
      <td>i</td>
      <td>1</td>
      <td>3677</td>
      <td>7.617600e+07</td>
    </tr>
    <tr>
      <th>41</th>
      <td>i</td>
      <td>2</td>
      <td>11605</td>
      <td>6.616000e+07</td>
    </tr>
    <tr>
      <th>42</th>
      <td>i</td>
      <td>3</td>
      <td>8469</td>
      <td>6.409200e+07</td>
    </tr>
    <tr>
      <th>43</th>
      <td>i</td>
      <td>4</td>
      <td>3346</td>
      <td>5.508000e+07</td>
    </tr>
    <tr>
      <th>44</th>
      <td>i</td>
      <td>5</td>
      <td>10366</td>
      <td>3.701100e+07</td>
    </tr>
    <tr>
      <th>45</th>
      <td>j</td>
      <td>1</td>
      <td>15385</td>
      <td>2.925000e+07</td>
    </tr>
    <tr>
      <th>46</th>
      <td>j</td>
      <td>2</td>
      <td>12301</td>
      <td>2.863500e+07</td>
    </tr>
    <tr>
      <th>47</th>
      <td>j</td>
      <td>3</td>
      <td>10858</td>
      <td>2.789800e+07</td>
    </tr>
    <tr>
      <th>48</th>
      <td>j</td>
      <td>4</td>
      <td>10420</td>
      <td>2.280000e+07</td>
    </tr>
    <tr>
      <th>49</th>
      <td>j</td>
      <td>5</td>
      <td>15016</td>
      <td>1.796000e+07</td>
    </tr>
    <tr>
      <th>50</th>
      <td>k</td>
      <td>1</td>
      <td>12092</td>
      <td>5.071600e+07</td>
    </tr>
    <tr>
      <th>51</th>
      <td>k</td>
      <td>2</td>
      <td>10610</td>
      <td>3.401000e+07</td>
    </tr>
    <tr>
      <th>52</th>
      <td>k</td>
      <td>3</td>
      <td>14907</td>
      <td>1.640000e+07</td>
    </tr>
    <tr>
      <th>53</th>
      <td>k</td>
      <td>4</td>
      <td>13903</td>
      <td>1.284000e+07</td>
    </tr>
    <tr>
      <th>54</th>
      <td>k</td>
      <td>5</td>
      <td>271</td>
      <td>1.216082e+07</td>
    </tr>
  </tbody>
</table>
</div>



## SQL 4


```python
query_decile = """
with main as (
select
    t.user_id,
    label.default_flag,
    t.outstanding,
    ntile(10) over (order by t.outstanding asc) AS decile
FROM transactions as t
join user_label_branch as label
on t.user_id = label.user_id
),
intermediate as (
select 
    decile,
    count(user_id) as user_count,
    sum(default_flag) as default_count,
    cast(sum(default_flag) as float)/cast(count(user_id) as float) as default_rate,
    min(outstanding) as min_outstanding,
    avg(outstanding) as avg_outstanding,
    max(outstanding) as max_outstanding
from main
group by 1
)
select
    decile,
    user_count,
    default_count,
    default_rate,
    sum(default_rate) over (order by decile) as cumulative_default_rate,
    min_outstanding,
    avg_outstanding,
    max_outstanding
from intermediate
order by 1
"""
decile = pd.read_sql(query_decile, con = psycopg2.connect(database = dbname, user = user))
```


```python
decile
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
      <th>decile</th>
      <th>user_count</th>
      <th>default_count</th>
      <th>default_rate</th>
      <th>cumulative_default_rate</th>
      <th>min_outstanding</th>
      <th>avg_outstanding</th>
      <th>max_outstanding</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1565</td>
      <td>344</td>
      <td>0.219808</td>
      <td>0.219808</td>
      <td>0.0</td>
      <td>1.321829e+05</td>
      <td>334128.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1565</td>
      <td>53</td>
      <td>0.033866</td>
      <td>0.253674</td>
      <td>334750.0</td>
      <td>7.923163e+05</td>
      <td>1334327.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1565</td>
      <td>48</td>
      <td>0.030671</td>
      <td>0.284345</td>
      <td>1334725.0</td>
      <td>1.985958e+06</td>
      <td>2587708.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1565</td>
      <td>97</td>
      <td>0.061981</td>
      <td>0.346326</td>
      <td>2587715.0</td>
      <td>3.058284e+06</td>
      <td>3463385.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1565</td>
      <td>101</td>
      <td>0.064537</td>
      <td>0.410863</td>
      <td>3463542.0</td>
      <td>4.042975e+06</td>
      <td>4721563.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>1564</td>
      <td>138</td>
      <td>0.088235</td>
      <td>0.499098</td>
      <td>4722525.0</td>
      <td>5.346846e+06</td>
      <td>6028571.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>1564</td>
      <td>113</td>
      <td>0.072251</td>
      <td>0.571349</td>
      <td>6029687.0</td>
      <td>7.188113e+06</td>
      <td>8494395.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>1564</td>
      <td>132</td>
      <td>0.084399</td>
      <td>0.655748</td>
      <td>8496119.0</td>
      <td>1.099222e+07</td>
      <td>14730155.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1564</td>
      <td>153</td>
      <td>0.097826</td>
      <td>0.753574</td>
      <td>14730155.0</td>
      <td>2.000410e+07</td>
      <td>27377448.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1564</td>
      <td>237</td>
      <td>0.151535</td>
      <td>0.905108</td>
      <td>27377448.0</td>
      <td>6.254995e+07</td>
      <td>798058574.0</td>
    </tr>
  </tbody>
</table>
</div>


