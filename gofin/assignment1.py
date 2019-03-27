
# coding: utf-8


import pandas as pd
import numpy as np
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

pd.set_option("display.max_columns", 100)
get_ipython().magic(u'matplotlib inline')



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



with open("./postgres.yaml", "r") as stream:
    try:
        data_loaded = yaml.load(stream)
    except yaml.YAMLERROR as exc:
        print(exc)



dbname = data_loaded['dbname']
user = data_loaded['user']
pg_load_table('./user_transactions.csv', 'transactions', dbname, user)
pg_load_table('./user_label_branch_edit.csv', 'user_label_branch', dbname, user)
# decided to load user_base_part1 only as user_base_part2 is just a duplicate of the first (ref to below cell)
pg_load_table('./user_base_part1.csv', 'user_base', dbname, user)


# outstanding: total outstanding amount of credit card usage
# credit_limit: credit limit amount that can be used
# total_cash_usage: last month total cash usage of customer
# total_retail_usage: last month total retail usage of customer
# bill: last month customer bill amount
# remaining_bill: remaining bill that has not been paid in the last month (assuming that this refers to the remaining amount of > month -2 bills that was not paid in month -1)
# 
# number_of_cards: # cards owned by customer
# branch_code: branch code of bank
# default_flag: 1: default, 0: non_default
# 
# total_usage = assume total_cash_usage + total_retail_usage
# payment_ratio: payment per bill ratio in the last month (payment made to billing amount in the last month)
# payment_ratio_3month: payment per bill ratio in the last 3 month
# payment_ratio_6month: payment per bill ratio in the last 6 month
# overlimit_percentage: overlimit percentage


user1 = pd.read_csv('./user_base_part1.csv')
user2 = pd.read_csv('./user_base_part2.csv')
transactions = pd.read_csv('./user_transactions.csv')
user_label_branch = pd.read_csv('./user_label_branch_edit.csv')



user1.equals(user2)
# user_base_part1 dataframe is exactly the same as user_base_part2


# ## SQL 1


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



combine.head()


# ## SQL 2


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



default_stats


# ## SQL 3


# check branch_code variant
query_check_bc = """
select distinct
    branch_code
from user_label_branch
"""
check_branch_code = pd.read_sql(query_check_bc, con = psycopg2.connect(database = dbname, user = user))



check_branch_code


# There are null values in branch code and there is a mix of upper and lower case. 


print("number of null values in branch_code column: " + str(len(user_label_branch[user_label_branch.branch_code.isnull()])))
print("% of rows where branch_code is null: " + str(round(len(user_label_branch[user_label_branch.branch_code.isnull()])/user_label_branch.shape[0]*100, 1)) + '%')



# change upper case branch code to lower case
user_label_branch = (
    user_label_branch
    .pipe(lambda x: x.assign(branch_code=x.branch_code.str.lower()))
)



user_label_branch.branch_code.describe()



# replace null values with 'a' as it occurs the most often
user_label_branch.loc[user_label_branch[user_label_branch.branch_code.isnull()].index.values, 'branch_code'] = 'a'



# reload user_label_branch into psql
user_label_branch.to_csv('./user_label_branch_edit.csv', index=False)
pg_load_table('./user_label_branch_edit.csv', 'user_label_branch', dbname, user)



# use 'usage' as a proxy to number of transactions completed. assuming higher the usage, the more the number of transactions completed
# since 6 months usage data is available, I shall provide the top 5 users with the most number of 'transactions' from each 'branch code'
# to get total_usage_6months, I will multiply 'total_6mo_usage_per_limit' with 'credit_limit'
# check 'total_6mo_usage_per_limit' and 'credit_limit' for anomaly
combine[['total_6mo_usage_per_limit', 'credit_limit']].describe()



# there are negative values for total_6mo_usage_per_limit
print("number of entries where total_6mo_usage_per_limit is negative: " + str(len(combine.query("total_6mo_usage_per_limit < 0"))))



combine.query("total_6mo_usage_per_limit < 0")[['total_usage_per_limit', 'total_6mo_usage_per_limit']]



# assuming there cannot be negative values for 'total_6mo_usage_per_limit
# from the above dataframe, we can see that 'total_usage_per_limit' and 'total_6mo_usage_per_limit are similar, other than the negative sign
# to clean the data we shall take the absolute of existing value
user1 = (
    user1
    .pipe(lambda x: x.assign(total_6mo_usage_per_limit=abs(x.total_6mo_usage_per_limit)))
)



# reload user_base into psql
user1.to_csv('./user_base_part1_edit.csv', index=False)
pg_load_table('./user_base_part1_edit.csv', 'user_base', dbname, user)



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



top5


# ## SQL 4


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



decile

