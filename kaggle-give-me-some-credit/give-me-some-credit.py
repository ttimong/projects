
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option("display.max_columns", 100)




train = pd.read_csv('./cs-training.csv', index_col=0)




test = pd.read_csv('./cs-test.csv', index_col=0)




test.head(2)




train.head(2)




test.shape




train.shape




combine = pd.concat([train, test], axis=0)




combine.head(2)




combine.columns = ['target', 'unsecuredline_utilization', 'age', 'count_of_30-59_days_past_due_not_worse', 'debt_ratio', 'monthly_income', 'count_open_creditlines_and_loans', 'count_90_days_late',
                'count_real_estate_loans_or_lines', 'count_of_60-89_days_past_due_not_worse', 'count_dependents']




combine.info()




target_dist = (
    train
    .groupby("target")
    .agg({"age": "count"})
    .reset_index()
)




target_dist




missing_dpt = combine[combine.count_dependents.isnull()]
missing_dpt_target_dist = (
    missing_dpt
    .groupby("target")
    .agg({"age": "count"})
    .reset_index()
)




missing_dpt_target_dist




combine.count_dependents.describe()




# impute null values in count_dependents column with median value, i.e. 0
combine.count_dependents.fillna(0, inplace=True)




missing_income = combine[combine.monthly_income.isnull()]
missing_income_target_dist = (
    missing_income
    .groupby("target")
    .agg({"age": "count"})
    .reset_index()
)
missing_income_target_dist




combine.age.sort_values().unique()




combine.query("monthly_income == 0").debt_ratio.describe()




combine.query("monthly_income > 0").debt_ratio.describe()




combine.query("monthly_income > 0").debt_ratio.quantile(q=0.99)




combine.monthly_income.fillna(-1, inplace=True)




combine.loc[combine.query("monthly_income == -1 and debt_ratio > 3").index.values, 'monthly_income'] = 0




len(combine.query("monthly_income == -1"))




combine.query("monthly_income == -1 and (target == 1 or target == 0 )").age.unique()

