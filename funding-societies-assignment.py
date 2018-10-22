
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

get_ipython().magic(u'matplotlib inline')
pd.set_option('display.max_columns', 100)



df = pd.read_csv('./loans_fs.csv', index_col=0)


# At Funding Societies, we use various data sources to assess the suitability of a borrower on our platform. We predict the willingness and ability of a borrower to repay us. This is a dataset of personal loans and the variables utilised to make lending decisions. 
# 
# Your task would be to summarise the data and perform some initial analysis on it with a focus on the factors that drive lending decisions.
# 
# Feel free to use any software to process, analyse, model and present your findings. The idea here is to give you a realistic experience of the work we do. Please send across all workings including your code and any working files. 
# 
# What we want to look at is your thought process in approaching the problem and so, please explain how you are cleaning and dissecting the data and why you are using any particular methods. All the best!
# 

# loan_amnt: amount of loan applied for by the borrower
# 
# funded_amnt: total amount committed to that loan at that point of time
# 
# funded_amnt_inv: total amount committed by investors for that loan at that point in time
# 
# empy_title: the job title supplied by the Borrower when applying for the loan
# 
# verification_status: indicates if income was verified by LC, not verified, or if the income source was verified
# 
# issue_d: the month which the loan was funded
# 
# pymnt_plan: indicates if a payment plan has been put in place for the loan
# 
# dti: a ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.


df.head(3)



df.term.unique()



df.pymnt_plan.unique()



df['term2'] = [int(i.split()[0]) for i in df.term]
df = (
    df
    .pipe(lambda x: x.assign(payment_plan = np.where(x.pymnt_plan=='n',1,0)))
    .drop(['term', 'pymnt_plan'], axis=1)
    .rename(columns={"term2": "term"})
)



df.info()



df.loan_status.unique()



df.loan_status.value_counts()



fully_paid = df.query("loan_status=='Fully Paid'")
effy = df.query("loan_status =='Does not meet the credit policy. Status:Fully Paid'")



fully_paid.describe()



effy.describe()



df.describe()



df = df.query("loan_amnt>=0 and loan_status!='Issued' ")



_ = plt.figure(figsize=(20,6))
_ = plt.subplot(121)
_ = sns.distplot(df.loan_amnt)
_ = plt.title('Frequency distribution')

_ = plt.subplot(122)
_ = sns.boxplot(x='loan_amnt', data=df, orient='v')
_ = plt.title('Boxplot')





