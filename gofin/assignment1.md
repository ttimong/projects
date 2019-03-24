

```python
import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
%matplotlib inline
```


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
dbname = 'gofin'
user = 'timong'
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
      <td>a</td>
      <td>8128</td>
      <td>0.001367</td>
      <td>0.0</td>
      <td>2.554872</td>
      <td>2</td>
      <td>6.777147</td>
      <td>5.670000</td>
      <td>1.283469e+07</td>
      <td>5483803.0</td>
      <td>2.476569e+07</td>
      <td>11000000.0</td>
      <td>8.769491e+06</td>
      <td>3528391.0</td>
      <td>8.638075e+06</td>
      <td>2891734.0</td>
      <td>3.152033e+06</td>
      <td>1238865.0</td>
      <td>0.434780</td>
      <td>0.266000</td>
      <td>72641.277436</td>
      <td>0.0</td>
      <td>2.558361e+06</td>
      <td>378409.0</td>
      <td>2.631043e+06</td>
      <td>427000.0</td>
      <td>0.110096</td>
      <td>0.034000</td>
      <td>0.159711</td>
      <td>0.103136</td>
      <td>0.185219</td>
      <td>0.110000</td>
      <td>68.511000</td>
      <td>36.60</td>
      <td>57.287692</td>
      <td>54.50</td>
      <td>84.437961</td>
      <td>66.7</td>
      <td>2.616967</td>
      <td>0.00</td>
      <td>0.542374</td>
      <td>0.523742</td>
      <td>0.505832</td>
      <td>0.450000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>a</td>
      <td>772</td>
      <td>0.034974</td>
      <td>0.0</td>
      <td>2.645078</td>
      <td>2</td>
      <td>6.816958</td>
      <td>5.330000</td>
      <td>2.215733e+07</td>
      <td>6581960.0</td>
      <td>2.856606e+07</td>
      <td>13000000.0</td>
      <td>1.712175e+07</td>
      <td>5527047.0</td>
      <td>1.920511e+07</td>
      <td>5708855.0</td>
      <td>6.421481e+06</td>
      <td>2675706.0</td>
      <td>0.693971</td>
      <td>0.850000</td>
      <td>117229.367876</td>
      <td>0.0</td>
      <td>7.773775e+05</td>
      <td>0.0</td>
      <td>8.946068e+05</td>
      <td>0.0</td>
      <td>0.034978</td>
      <td>0.000000</td>
      <td>0.125440</td>
      <td>0.054800</td>
      <td>0.221830</td>
      <td>0.095200</td>
      <td>18.576658</td>
      <td>0.00</td>
      <td>15.156619</td>
      <td>20.10</td>
      <td>37.813368</td>
      <td>28.9</td>
      <td>8.150415</td>
      <td>0.00</td>
      <td>0.721190</td>
      <td>0.868000</td>
      <td>0.683543</td>
      <td>0.750000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>b</td>
      <td>1219</td>
      <td>0.001641</td>
      <td>0.0</td>
      <td>2.437244</td>
      <td>2</td>
      <td>6.447065</td>
      <td>5.580000</td>
      <td>8.526206e+06</td>
      <td>3410092.0</td>
      <td>1.477974e+07</td>
      <td>6000000.0</td>
      <td>6.103418e+06</td>
      <td>2487707.0</td>
      <td>5.374368e+06</td>
      <td>2076427.0</td>
      <td>2.064461e+06</td>
      <td>944461.0</td>
      <td>0.459491</td>
      <td>0.329000</td>
      <td>137666.589007</td>
      <td>0.0</td>
      <td>1.877469e+06</td>
      <td>150000.0</td>
      <td>2.015266e+06</td>
      <td>192000.0</td>
      <td>0.129873</td>
      <td>0.024300</td>
      <td>0.184329</td>
      <td>0.122821</td>
      <td>0.218698</td>
      <td>0.121000</td>
      <td>56.582067</td>
      <td>25.10</td>
      <td>72.359934</td>
      <td>52.20</td>
      <td>78.116760</td>
      <td>66.7</td>
      <td>3.222986</td>
      <td>0.00</td>
      <td>0.562925</td>
      <td>0.576000</td>
      <td>0.530487</td>
      <td>0.512561</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>b</td>
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
      <td>1477983.0</td>
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
      <td>33.3</td>
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
      <td>c</td>
      <td>307</td>
      <td>0.006515</td>
      <td>0.0</td>
      <td>2.416938</td>
      <td>2</td>
      <td>6.464255</td>
      <td>5.670000</td>
      <td>7.250387e+06</td>
      <td>3632373.0</td>
      <td>1.501954e+07</td>
      <td>7000000.0</td>
      <td>5.654766e+06</td>
      <td>3186854.0</td>
      <td>5.281793e+06</td>
      <td>2587976.0</td>
      <td>2.147930e+06</td>
      <td>1172769.0</td>
      <td>0.465805</td>
      <td>0.294000</td>
      <td>33859.934853</td>
      <td>0.0</td>
      <td>1.164391e+06</td>
      <td>272379.0</td>
      <td>1.198251e+06</td>
      <td>307820.0</td>
      <td>0.112982</td>
      <td>0.031667</td>
      <td>0.181747</td>
      <td>0.130000</td>
      <td>0.218102</td>
      <td>0.151000</td>
      <td>64.918241</td>
      <td>31.80</td>
      <td>64.280912</td>
      <td>53.60</td>
      <td>68.760391</td>
      <td>66.7</td>
      <td>2.980261</td>
      <td>0.00</td>
      <td>0.566108</td>
      <td>0.599000</td>
      <td>0.528078</td>
      <td>0.511440</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>c</td>
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
      <td>1841239.0</td>
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
      <td>46.2</td>
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
      <td>d</td>
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
      <td>1280569.0</td>
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
      <td>60.8</td>
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
      <td>d</td>
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
      <td>1830613.0</td>
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
      <td>12.2</td>
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
      <td>e</td>
      <td>535</td>
      <td>0.013084</td>
      <td>0.0</td>
      <td>2.437383</td>
      <td>2</td>
      <td>6.135655</td>
      <td>5.250000</td>
      <td>8.698120e+06</td>
      <td>3858611.0</td>
      <td>1.531589e+07</td>
      <td>6000000.0</td>
      <td>6.541631e+06</td>
      <td>3160704.0</td>
      <td>6.385659e+06</td>
      <td>2888880.0</td>
      <td>2.470708e+06</td>
      <td>1375386.0</td>
      <td>0.540497</td>
      <td>0.473000</td>
      <td>72971.962617</td>
      <td>0.0</td>
      <td>1.463266e+06</td>
      <td>157200.0</td>
      <td>1.536206e+06</td>
      <td>212000.0</td>
      <td>0.120469</td>
      <td>0.025100</td>
      <td>0.185696</td>
      <td>0.119000</td>
      <td>0.224648</td>
      <td>0.141000</td>
      <td>45.823234</td>
      <td>19.50</td>
      <td>24.732299</td>
      <td>49.10</td>
      <td>-79.198710</td>
      <td>51.8</td>
      <td>5.249065</td>
      <td>0.00</td>
      <td>0.608457</td>
      <td>0.619000</td>
      <td>0.565512</td>
      <td>0.531000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>e</td>
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
      <td>778718.0</td>
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
      <td>30.7</td>
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
      <td>f</td>
      <td>1553</td>
      <td>0.001935</td>
      <td>0.0</td>
      <td>2.470702</td>
      <td>2</td>
      <td>6.586646</td>
      <td>5.750000</td>
      <td>9.644462e+06</td>
      <td>3572509.0</td>
      <td>1.683355e+07</td>
      <td>7000000.0</td>
      <td>6.133652e+06</td>
      <td>2543805.0</td>
      <td>6.384182e+06</td>
      <td>1770815.0</td>
      <td>2.362210e+06</td>
      <td>825824.0</td>
      <td>0.437460</td>
      <td>0.267000</td>
      <td>68993.256922</td>
      <td>0.0</td>
      <td>1.731244e+06</td>
      <td>208000.0</td>
      <td>1.800237e+06</td>
      <td>248000.0</td>
      <td>0.128885</td>
      <td>0.024900</td>
      <td>0.187883</td>
      <td>0.127417</td>
      <td>0.217712</td>
      <td>0.133000</td>
      <td>114.231597</td>
      <td>43.40</td>
      <td>28.710464</td>
      <td>55.90</td>
      <td>109.718886</td>
      <td>66.7</td>
      <td>3.073825</td>
      <td>0.00</td>
      <td>0.562878</td>
      <td>0.557540</td>
      <td>0.524250</td>
      <td>0.489000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>f</td>
      <td>144</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.694444</td>
      <td>2</td>
      <td>6.691903</td>
      <td>5.580000</td>
      <td>2.209711e+07</td>
      <td>5339719.0</td>
      <td>2.308333e+07</td>
      <td>8000000.0</td>
      <td>1.508542e+07</td>
      <td>4845923.0</td>
      <td>2.013744e+07</td>
      <td>5154543.0</td>
      <td>6.339170e+06</td>
      <td>2377798.0</td>
      <td>0.771510</td>
      <td>0.971026</td>
      <td>150347.222222</td>
      <td>0.0</td>
      <td>1.518519e+06</td>
      <td>0.0</td>
      <td>1.668866e+06</td>
      <td>0.0</td>
      <td>0.030048</td>
      <td>0.000000</td>
      <td>0.136725</td>
      <td>0.093500</td>
      <td>0.228372</td>
      <td>0.114000</td>
      <td>-0.326597</td>
      <td>0.00</td>
      <td>19.489722</td>
      <td>21.60</td>
      <td>16.872153</td>
      <td>25.4</td>
      <td>8.047917</td>
      <td>2.89</td>
      <td>0.798556</td>
      <td>0.934000</td>
      <td>0.737184</td>
      <td>0.862000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>g</td>
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
      <td>487031.0</td>
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
      <td>0.223330</td>
      <td>0.155000</td>
      <td>67.577052</td>
      <td>99.00</td>
      <td>212.179907</td>
      <td>70.60</td>
      <td>198.343955</td>
      <td>66.7</td>
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
      <td>g</td>
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
      <td>2976937.5</td>
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
      <td>36.0</td>
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
      <td>h</td>
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
      <td>856016.0</td>
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
      <td>66.7</td>
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
      <td>h</td>
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
      <td>1650486.0</td>
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
      <td>29.1</td>
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
      <td>i</td>
      <td>889</td>
      <td>0.003375</td>
      <td>0.0</td>
      <td>2.388076</td>
      <td>2</td>
      <td>6.183646</td>
      <td>5.000000</td>
      <td>8.225831e+06</td>
      <td>3872598.0</td>
      <td>1.299888e+07</td>
      <td>5000000.0</td>
      <td>5.469412e+06</td>
      <td>2788118.0</td>
      <td>5.946553e+06</td>
      <td>2909645.0</td>
      <td>2.315871e+06</td>
      <td>1294819.0</td>
      <td>0.537143</td>
      <td>0.537000</td>
      <td>38954.799775</td>
      <td>0.0</td>
      <td>1.455952e+06</td>
      <td>151028.0</td>
      <td>1.494907e+06</td>
      <td>182512.0</td>
      <td>0.122806</td>
      <td>0.020500</td>
      <td>0.191190</td>
      <td>0.123000</td>
      <td>0.232317</td>
      <td>0.128000</td>
      <td>49.572913</td>
      <td>25.60</td>
      <td>-9.334578</td>
      <td>50.00</td>
      <td>64.296873</td>
      <td>62.2</td>
      <td>3.829708</td>
      <td>0.00</td>
      <td>0.653003</td>
      <td>0.726000</td>
      <td>0.600688</td>
      <td>0.605000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>i</td>
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
      <td>2081872.0</td>
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
      <td>28.1</td>
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
      <td>j</td>
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
      <td>1000000.0</td>
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
      <td>66.7</td>
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
      <td>j</td>
      <td>33</td>
      <td>4.484848</td>
      <td>4.0</td>
      <td>2.272727</td>
      <td>2</td>
      <td>7.897980</td>
      <td>7.500000</td>
      <td>2.378929e+06</td>
      <td>667120.0</td>
      <td>9.333333e+06</td>
      <td>5000000.0</td>
      <td>1.763800e+06</td>
      <td>284964.0</td>
      <td>1.957044e+06</td>
      <td>232897.0</td>
      <td>9.058974e+05</td>
      <td>86788.0</td>
      <td>0.287932</td>
      <td>0.034688</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.555303e+05</td>
      <td>0.0</td>
      <td>2.949242e+05</td>
      <td>0.0</td>
      <td>0.055318</td>
      <td>0.000000</td>
      <td>0.060137</td>
      <td>0.000129</td>
      <td>0.262712</td>
      <td>0.009930</td>
      <td>16.061818</td>
      <td>0.00</td>
      <td>27.553939</td>
      <td>14.75</td>
      <td>70.197879</td>
      <td>18.8</td>
      <td>1.298485</td>
      <td>0.00</td>
      <td>0.359937</td>
      <td>0.146000</td>
      <td>0.486240</td>
      <td>0.159000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0</td>
      <td>k</td>
      <td>158</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>2.322785</td>
      <td>2</td>
      <td>4.826354</td>
      <td>4.170000</td>
      <td>8.457255e+06</td>
      <td>3471577.0</td>
      <td>1.424684e+07</td>
      <td>7000000.0</td>
      <td>4.714190e+06</td>
      <td>1659766.0</td>
      <td>4.906485e+06</td>
      <td>827958.0</td>
      <td>2.057606e+06</td>
      <td>413979.0</td>
      <td>0.388488</td>
      <td>0.130000</td>
      <td>46677.215190</td>
      <td>0.0</td>
      <td>2.495705e+06</td>
      <td>272328.0</td>
      <td>2.542382e+06</td>
      <td>300000.0</td>
      <td>0.155565</td>
      <td>0.045500</td>
      <td>0.184173</td>
      <td>0.132000</td>
      <td>0.207218</td>
      <td>0.135375</td>
      <td>63.983165</td>
      <td>99.80</td>
      <td>104.107975</td>
      <td>62.30</td>
      <td>122.869177</td>
      <td>68.2</td>
      <td>2.427911</td>
      <td>0.00</td>
      <td>0.516594</td>
      <td>0.508000</td>
      <td>0.462514</td>
      <td>0.371000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>k</td>
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
      <td>1545903.0</td>
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
      <td>37.4</td>
      <td>3.527647</td>
      <td>0.00</td>
      <td>0.761455</td>
      <td>0.933000</td>
      <td>0.700330</td>
      <td>0.811000</td>
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
      <td>I</td>
    </tr>
    <tr>
      <th>9</th>
      <td>k</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A</td>
    </tr>
    <tr>
      <th>11</th>
      <td>a</td>
    </tr>
    <tr>
      <th>12</th>
      <td>D</td>
    </tr>
    <tr>
      <th>13</th>
      <td>f</td>
    </tr>
    <tr>
      <th>14</th>
      <td>G</td>
    </tr>
    <tr>
      <th>15</th>
      <td>B</td>
    </tr>
    <tr>
      <th>16</th>
      <td>F</td>
    </tr>
    <tr>
      <th>17</th>
      <td>c</td>
    </tr>
    <tr>
      <th>18</th>
      <td>K</td>
    </tr>
    <tr>
      <th>19</th>
      <td>E</td>
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
user_label_branch2.branch_code.describe()
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




```python

```
