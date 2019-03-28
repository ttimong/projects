
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




combine = pd.concat([train, test], axis=0)




combine.head(2)




combine.columns = ['target', 'unsecured_line_utilization', 'age', 'count_of_30-59_days_past_due_not_worse', 'debt_ratio', 'monthly_income', 'count_open_creditlines_and_loans', 'count_90_days_late',
                'count_real_estate_loans', 'count_of_60-89_days_past_due_not_worse', 'count_dependents']




combine.info()




combine.describe()




combine.unsecured_line_utilization.quantile(q=0.99)




# highly likely that unsecured_line_utilization to be more than 1.1
# assume that any value above 1.1 is erroneous 
print("count of erroneous unsecured_line_utilization: " + str(len(combine.query("unsecured_line_utilization > 1.1"))))


# erroneous unsecuredline_utilization is about 1% of the entire data. I will replace these erroneous values with the median value of unsecured_line_utilization



# replacing errorneous values for unsecured_line_utilization
combine.loc[combine.query("unsecured_line_utilization >= 1.1").index.values, 'unsecured_line_utilization'] = combine.query("unsecured_line_utilization < 1.1").unsecured_line_utilization.median()




# replacing age = 0 with the mean age of data set
combine.loc[combine.query("age == 0").index.values, 'age'] = combine.query("age != 0").age.median()




combine.count_dependents.describe()




# impute null values in count_dependents column with median value, i.e. 0
combine.count_dependents.fillna(0, inplace=True)




combine['age_group'] = ['21- 40' if (i>=21 and i<41) else '41-60' if (i>=41 and i<61) else '61-80' if (i>=61 and i<81) else '>81' for i in combine.age]




combine.




_ = sns.boxplot(x='age_group', y='monthly_income', data=combine)




_ = sns.boxplot(x='age_group', y='monthly_income', data=combine.query("monthly_income < 200000"))




combine.query("monthly_income > 0").monthly_income.quantile(q=0.99)




combine = (
    combine
    .pipe(lambda x: x.assign(monthly_income=np.where(combine[combine.monthly_income > 25000]




len(combine.query("monthly_income >25000"))




len(combine)














combine.query("monthly_income == 83000")




_ = sns.distplot(combine[combine.monthly_income.notnull()].query("monthly_income< 25000").monthly_income)
_ = plt.xticks(rotation=75)




len(combine[combine.monthly_income.notnull()].query("monthly_income< 200000"))/len(combine[combine.monthly_income.notnull()])




combine[combine.monthly_income.notnull()].query("monthly_income > 25000").groupby("target").agg({"age":"count"}).reset_index()




_ = sns.distplot(combine.query("monthly_income >= 0 and monthly_income <={}".format(combine.query("monthly_income != 0").monthly_income.quantile(q=0.99))).monthly_income)




len(combine.query("monthly_income == 0"))




combine.monthly_income.quantile(q=0.99)




combine.query("monthly_income == 0").groupby("target").agg({"age": "count"}).reset_index()




len(combine.query("monthly_income == 0 and debt_ratio <3"))




combine.query("monthly_income > 0").debt_ratio.quantile(q=0.99)




len(combine[combine.monthly_income.isnull()].query("debt_ratio>3"))


# We can see that at 99 percentile, debt ratio is less than 3 when monthly income is more than 0



# filling null values in monthly_income column with -1
combine.monthly_income.fillna(-1, inplace=True)




# from the above observations, it is very likely that a person's monthly_income would be zero if his/her debt_ratio is above 3
combine.loc[combine.query("monthly_income == -1 and debt_ratio > 3").index.values, 'monthly_income'] = 0




# how many null values of monthly_income left
len(combine.query("monthly_income == -1"))




combine.age.sort_values().unique()




combine.query("age == 0")




# creating new column age_group so that we can better ascertain the missing monthly_income based on age
combine['age_group'] = ['21- 30' if (i>=21 and i<31) else '31-40' if (i>=31 and i<41) else '41-50' if (i>=41 and i<50) else '51-60' if (i>=51 and i<60) else '61-70' if (i>=61 and i<70)                       else '71-80' if (i>=71 and i<80) else '81-90' if (i>=81 and i<90) else '91-100' if (i>=91 and i<100) else '>101' for i in combine.age]




# filling missing monthly_income values with the median income for that specific age_group
combine.loc[combine.query("monthly_income == -1 and age_group == '31-40'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '31-40'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '41-50'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '41-50'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '51-60'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '51-60'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '61-70'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '61-70'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '71-80'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '71-80'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '81-90'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '81-90'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '91-100'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '91-100'").monthly_income.median()
combine.loc[combine.query("monthly_income == -1 and age_group == '>101'").index.values, 'monthly_income'] = combine.query("monthly_income != -1 and age_group == '>101'").monthly_income.median()




combine.info()

