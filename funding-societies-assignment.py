
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from scipy import stats

get_ipython().magic(u'matplotlib inline')
pd.set_option('display.max_columns', 100)



df = pd.read_csv('./loans_fs.csv', index_col=0)


# 1. Predit willingess and ability of a borrower to repay us
# 
# 2. Summarise the data and perform intial analysis on the focus that drive lending decisions (WHO SHOULD WE LEND MONEY TO?)

# dti: a ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.

# term: 36 months, 60 months, 600 months
# 
# payment_plan: y, n
# 
# grade: A, B, C, D, E, F, G
# 
# sub_grade: 1 to 5
# 
# emp_length: < 1 year to 10+ years increment by 1 year
# 
# home_ownership: rent, own, mortgage, other, none, any
# 
# verification_status: verified, source verified, not verified
# 
# loan_status: fully paid, does not meet credit policy. fully paid, current, in grace period, late (16-30 days), late (31-120 days), default, charged off, does not meet credit policy. charged off
# 
# purpose: credit card, car, small business, wedding, debt consolidation, home improvement, major purchase, medical, moving, vacation, house, renewable energy, educational, other


df.head(1)



df.info()



df.describe()



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
        'addr_state', 'dti', 'term', 'pymnt_plan']]
)



# assumptions 
# funded_amnt or funded_amnt_inv cannot be more than loan_amnt
# funded_amnt_inv cannot be more than funded_amnt

# checking out how many rows will be removed
count = df_edit.query("funded_amnt_inv > loan_amnt or funded_amnt > loan_amnt or funded_amnt_inv > funded_amnt").shape[0]
print("erroneous rows: " + str(count))

print("number of rows in raw data: " + str(df_edit.shape[0]))



# since assumed erroneous rows is only approx ~0.5% of raw data, I will remove them
df_edit = df_edit.query("funded_amnt_inv <= loan_amnt and funded_amnt <= loan_amnt and funded_amnt_inv <= funded_amnt")



df_edit.describe()



# seems there are some erroneous entry in dti
# will use IQR to remove outliers
upper_limit = np.percentile(df_edit.dti, 75) + (stats.iqr(df_edit.dti) * 1.5)
lower_limit = np.percentile(df_edit.dti, 25) - (stats.iqr(df_edit.dti) * 1.5)

df_edit = df_edit.query("dti >= {lower} and dti <= {upper}".format(lower=lower_limit, upper=upper_limit))



print("number of rows originally: " + str(df.shape[0]))
print("number of rows left: " + str(df_edit.shape[0]))



df_edit.loan_status.unique()



df_edit.term.unique()



# converting payment terms into integers
df_edit['term2'] = [int(i.split()[0]) for i in df_edit.term]

# changing payment_plan to 1 (y) and 0 (n) and filling na with 999
# defining borrowers who are willing to make repayments are loans with status "Fully Paid" and "Current"
# removing loan status "Issued" because it cannot be determine whether are they willing to pay
# creating new feature funded_amnt_lc. assuming that lc top ups loans that there aren't enough investors
df_edit = (
    df_edit
    .query("loan_status != 'Issued'")
    .pipe(lambda x: x.assign(payment_plan = np.where(x.pymnt_plan=='y',1,0)))
    .pipe(lambda x: x.assign(
            loan_status_mod = np.where(x.loan_status == 'Fully Paid',1,
                                       np.where(x.loan_status == 'Current',1,
                                                np.where(x.loan_status == 'Does not meet the credit policy. Status:Fully Paid',1,0)))))
    .pipe(lambda x: x.assign(funded_amnt_lc=x.funded_amnt-x.funded_amnt_inv))
    .drop(['term', 'pymnt_plan'], axis=1)
    .rename(columns={"term2": "term"})
    .fillna(999)
    [['member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'funded_amnt_lc', 'int_rate',
       'installment', 'grade', 'sub_grade', 'emp_title', 'emp_length',
       'home_ownership', 'annual_inc', 'verification_status', 'issue_d',
       'loan_status', 'loan_status_mod', 'url', 'desc', 'purpose', 'title', 'zip_code',
       'addr_state', 'dti', 'term', 'payment_plan']]
    )



_ = plt.figure(figsize=(20,5))
_ = plt.subplot(131)
_ = sns.boxplot(x='loan_status_mod', y='loan_amnt', data=df_edit)

_ = plt.subplot(132)
_ = sns.barplot()



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





