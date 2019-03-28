
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", 100)
get_ipython().magic(u'matplotlib inline')



transactions = pd.read_csv('./user_transactions.csv')
user_base = pd.read_csv('./user_base_part1_edit.csv')
user_label = pd.read_csv('./user_label_branch_edit.csv')


# # Data Cleaning


# divide monetary value by 10000 so that it is easier to read
combine = (
    transactions
    .merge(user_base, on='user_id', how='left')
    .merge(user_label, on='user_id', how='left')
    [['user_id', 'branch_code', 'default_flag', 'number_of_cards', 'outstanding', 'credit_limit', 'bill', 'total_cash_usage',
       'total_retail_usage', 'total_usage', 'total_usage_per_limit', 'total_3mo_usage_per_limit',
       'total_6mo_usage_per_limit', 'remaining_bill', 'payment_ratio', 'payment_ratio_3month', 'payment_ratio_6month','overlimit_percentage',
       'delinquency_score', 'years_since_card_issuing',
       'remaining_bill_per_number_of_cards', 'remaining_bill_per_limit', 'utilization_3month', 'utilization_6month']]
    .pipe(lambda x: x.assign(outstanding=x.outstanding/10000))
    .pipe(lambda x: x.assign(credit_limit=x.credit_limit/10000))
    .pipe(lambda x: x.assign(bill=x.bill/10000))
    .pipe(lambda x: x.assign(total_cash_usage=x.total_cash_usage/10000))
    .pipe(lambda x: x.assign(total_retail_usage=x.total_retail_usage/10000))
    .pipe(lambda x: x.assign(total_usage=x.total_usage/10000))
    .pipe(lambda x: x.assign(remaining_bill=x.remaining_bill/10000))
    .pipe(lambda x: x.assign(remaining_bill_per_number_of_cards=x.remaining_bill_per_number_of_cards/10000))
)



combine[['total_cash_usage', 'total_retail_usage', 'total_usage']].describe()


# Since there are 45 null values in the total_cash_usage, I would need to find a way to fill in those missing values. 
# 
# Logically, it seems that total_cash_usage + total_retail_usage = total_usage. If this were true, I would be able to use total_retail_usage and total_usage to calculate total_cash_usage.


# assuming total_cash_usage + total_retail_usage = total_usage
# check out assumption
check_assumption1 = (
    combine
    [['total_cash_usage', 'total_retail_usage', 'total_usage']]
    .fillna('x')
    .query("total_cash_usage != 'x'")
    .pipe(lambda x: x.assign(total_usage_check=x.total_cash_usage+x.total_retail_usage-x.total_usage))
)



print("% of rows that does not tally: " + str(round(len(check_assumption1.query("total_usage_check !=0"))/check_assumption1.shape[0] * 100, 1)) + '%')



check_assumption1.query("total_usage_check !=0")


# After doing a check, it seems that my assumption is correct. For most of the rows that do not fit the assumption, it just seems to be a rounding issue. I will proceed to fill in the null values of total_cash_usage with total_usage - total_retail_usage. Furthermore, I would think that it is impossible to have negative values for total_usage and total_retail_usage. In order to resolve the negative values, I will mulitply those values with -1.


print("% of data points where total_retail_usage or total_usage is negative: " + str(round(len(combine.query("total_retail_usage < 0 or total_usage < 0"))/combine.shape[0] * 100, 1)) + '%')



# assumption is correct
# fill missing null values of total_cash_usage with total_usage - total_retail_usage 
# correct rows where total_usage != total_cash_usage + total_retail_usage
# mulitply -1 with values where total_usage or total_retail_usage is negaive
combine = (
    combine
    .fillna({"total_cash_usage": -999})
    .pipe(lambda x: x.assign(total_retail_usage=np.where(x.total_retail_usage<0, -1*x.total_retail_usage, x.total_retail_usage)))
    .pipe(lambda x: x.assign(total_usage=np.where(x.total_usage<0, -1*x.total_usage, x.total_usage)))
    .pipe(lambda x: x.assign(total_cash_usage=np.where(x.total_cash_usage==-999, x.total_usage-x.total_retail_usage, x.total_cash_usage)))
    .pipe(lambda x: x.assign(total_usage_check=x.total_usage-x.total_cash_usage-x.total_retail_usage))
    .pipe(lambda x: x.assign(total_usage=np.where(x.total_usage_check!=0, x.total_cash_usage+x.total_retail_usage, x.total_usage)))
    .drop('total_usage_check', axis=1)
)



combine[['total_usage_per_limit', 'total_3mo_usage_per_limit', 'total_6mo_usage_per_limit']].describe()


# Similarly, I would think that it is improbable for the columns total_usage_per_limit and total_3mo_usage_per_limit to have negative values. To correct these values, I would make the assumption that total_usage_per_limit = total_usage/credit_limit. 


# assume total_usage_per_limit is calculated as total_usage/credit_limit
check_assumption2 = (
    combine
    [['credit_limit', 'total_usage', 'total_usage_per_limit']]
    .pipe(lambda x: x.assign(total_usage_per_limit_recal=x.total_usage/x.credit_limit))
    .pipe(lambda x: x.assign(total_usage_per_limit_check=round(x.total_usage_per_limit_recal-abs(x.total_usage_per_limit), 1)))
)



check_assumption2.query("total_usage_per_limit_check != 0")


# All, but one entry agreeds with my assumption in total_usage_per_limit calculation logic. I will proceed to re-calculate the total_usage_per_limit to remove erroneous data like the above entry as well as negative values. As for resolving negative total_3mo_usage_per_limit, I will multiply -1 to the negative values.


# other than index 4819, the assumption on the calculation of total_usage_per_limit seems right
# re-calculate total_usage_per_limit using total_usage/credit_limit to get rid of negative values of total_usage_per_limit
# multiply -1 to negative total_3mo_usage_per_limit
combine = (
    combine
    .pipe(lambda x: x.assign(total_usage_per_limit=x.total_usage/x.credit_limit))
    .pipe(lambda x: x.assign(total_3mo_usage_per_limit=np.where(x.total_3mo_usage_per_limit<0, -1*x.total_3mo_usage_per_limit, x.total_3mo_usage_per_limit)))
)



combine[['payment_ratio', 'payment_ratio_3month', 'payment_ratio_6month']].describe()


# Assuming that payment is calculated as payment_ratio * bill, the payment for each month is outrageous. The median payment_ratio is approximately 27, meaning to say that 50% of the users are making payment of more than 27 times their last month bill. As such, I would assume that the unit of payment_ratio, payment_ratio_3month and payment_ratio_6month columns are in percentages. I will divide these columns by 100 to obtain the actual ratio. Furthermore, I will multiply -1 to negative payment ratios as it is logically impossible to have negative payment ratios.


# divide payment ratios columns by 100 to get the real payment_ratio
# for payment ratios that are negative, I will mulitply by -1
combine = (
    combine
    .pipe(lambda x: x.assign(payment_ratio=x.payment_ratio/100))
    .pipe(lambda x: x.assign(payment_ratio=np.where(x.payment_ratio<0, -1*x.payment_ratio, x.payment_ratio)))
    .pipe(lambda x: x.assign(payment_ratio_3month=x.payment_ratio_3month/100))
    .pipe(lambda x: x.assign(payment_ratio_3month=np.where(x.payment_ratio_3month<0, -1*x.payment_ratio_3month, x.payment_ratio_3month)))
    .pipe(lambda x: x.assign(payment_ratio_6month=x.payment_ratio_6month/100))
    .pipe(lambda x: x.assign(payment_ratio_6month=np.where(x.payment_ratio_6month<0, -1*x.payment_ratio_6month, x.payment_ratio_6month)))
)



combine[['payment_ratio', 'payment_ratio_3month', 'payment_ratio_6month']].describe()


# From the table above, we can see that there are some outliers in payment ratio columns. The maximum payment ratios is approximately 700 times the 75th percentile and it is probably impossible to make a payment 700x the customer's bill. As such, I would think that there are some erroneous data in these columns. In order to pick them out, I will use IQR to decide on my upper limit and I will treat everything above it as outliers.


# using iqr to spot outliers
def upper_limit(df, column):
    iqr = df['{}'.format(column)].quantile(q=0.75) - df['{}'.format(column)].quantile(q=0.25)
    upper_limit = df['{}'.format(column)].quantile(q=0.75) + 1.5*iqr
    print("upper limit of {} based on 1.5*IQR: ".format(column) + str(upper_limit))



upper_limit(combine, 'payment_ratio')
upper_limit(combine, 'payment_ratio_3month')
upper_limit(combine, 'payment_ratio_6month')



print("% of data points where payment_ratio > 2.5: " + str(round(len(combine.query("payment_ratio > 2.5"))/len(combine)*100, 1)) + '%')
print("% of data points where payment_ratio_3month > 2.5: " + str(round(len(combine.query("payment_ratio_3month > 1.992"))/len(combine)*100, 1)) + '%')
print("% of data points where payment_ratio_6month > 2.5: " + str(round(len(combine.query("payment_ratio_6month > 2.167"))/len(combine)*100, 1)) + '%')


# We can see that a small portion of the data are outliers. I will choose to drop them instead of replacing those values.


# since the % of outliers is low, I will drop these rows
print("% of data points that will be dropped: " + str(round(len(combine.query("payment_ratio > 2.5 or payment_ratio_3month > 1.992 or payment_ratio_6month > 2.167"))/len(combine)*100, 1)) + '%')
combine = combine.query("payment_ratio <= 2.5 and payment_ratio_3month <= 1.992 and payment_ratio_6month <= 2.167")



print("% of data points where payment_ratio equals to 0: " + str(round(len(combine.query("payment_ratio == 0"))/len(combine)*100, 1)) + '%')
print("% of data points where payment_ratio equals to 0: " + str(round(len(combine.query("payment_ratio_3month == 0"))/len(combine)*100, 1)) + '%')
print("% of data points where payment_ratio equals to 0: " + str(round(len(combine.query("payment_ratio_6month == 0"))/len(combine)*100, 1)) + '%')



combine.query("payment_ratio != 0")[['payment_ratio', 'payment_ratio_3month', 'payment_ratio_6month']].corr()


# Taking a deeper look into payment_ratio, we can see that approximately 1/3 of the data have a value of 0. It is quite unlikely for such a sizeable portion of the data to be 0 as such a phenomenon cannot be observed in payment_ratio_3month and payment_ratio_6month. I would assume that these data in payment_ratio are erroneous. Since payment_ratio and payment_ratio_3month have a strong correlation, I will be replacing the 0 values in payment_ratio with payment_ratio_3month values.


# replacing 0s in payment_ratio with payment_ratio_3month values
combine = (
    combine
    .pipe(lambda x: x.assign(payment_ratio=np.where(x.payment_ratio==0, x.payment_ratio_3month, x.payment_ratio)))
)



combine[['user_id', 'overlimit_percentage', 'delinquency_score']].describe()



combine.delinquency_score.value_counts()



combine[combine.delinquency_score.isnull()].groupby("default_flag").agg({"user_id": "count"}).reset_index()


# Since all of the data points where delinquency_score is null are non-defaulters, I will impute the null values with 0 as it is the mode of the column


# imputing delinquency_score null values with mode
combine.loc[combine[combine.delinquency_score.isnull()].index.values, 'delinquency_score'] = 0


# In order to impute the null values in overlimit_percentage, I shall make the assumption that overlimit_percentage = outstanding/credit_limit. If this assumption is true, I would be able to use this logic to impute overlimit_percentage null values.


# assume that overlimit_percentage is calculated using outstanding divide by credit_limit
# check assumption
check_assumption3 = (
    combine
    [combine.overlimit_percentage.notnull()]
    [['overlimit_percentage', 'credit_limit', 'outstanding', 'default_flag']]
    .pipe(lambda x: x.assign(overlimit_percentage_recal=np.where(x.outstanding>x.credit_limit, (x.outstanding-x.credit_limit)/x.credit_limit * 100, 0)))
    .query("overlimit_percentage != overlimit_percentage_recal")
)



check_assumption3.shape


# Approximately 1/3 of the data does not agreed with my above assumption. I will refrain from using my assumed logic to impute missing values in overlimit_percentage. Instead I will drop these rows. I will also divide the values in the column by 100 so that it would be of the same scale as the other columns


# dropping rows where there are missing values
# dividing overlimit_percentage by 100
combine.dropna(inplace=True)
combine = (
    combine
    .pipe(lambda x: x.assign(overlimit_percentage=x.overlimit_percentage/100))
)



combine[['utilization_3month', 'utilization_6month']].describe()



upper_limit(combine, 'utilization_3month')
upper_limit(combine, 'utilization_6month')



print("% of data where utilization_3month or utilization_6month are outliers: " + str(round(len(combine.query("utilization_3month > 2 or utilization_6month > 1.928"))/len(combine)*100, 1)) + '%')


# Again, we can see that there are some outliers in the data and I will choose to drop them instead of replacing their values as the proportion is very small.


# dropping these rows that has outliers
combine = combine.query("utilization_3month <= 2 and utilization_6month <= 1.928")



combine.shape



combine.info()



combine.describe()


# After cleaning the data, I am left with 15,121 rows.

# # EDA

# The data provided revolves around historical credit card transactions, usage and payment details. However, it does not seem that the default_flag refers to credit card default as the payment_ratio is not 0 for defaulters.
# 
# Taking into account that GoFin provides credit loan to users, I would assume that the default_flag refers to users who took up credit loan from GoFin default or not. The credit card details would assist GoFin in making the decision to lend money to users. My end goal in mind is to come up with a **predictive model to ascertain whether a user will default on his/her loan to GoFin**. The by-product of the model will also inform me on the **importance of each feature to determine defaulters**.
# 
# I would also assume that the branch_code refers to GoFin branches that gives out loans to users.

# ### Summary
# - Data set is highly skewed, with baseline default rate to be at 9.7%
# - Branch D and G have lower default rate than baseline while Branch E is higher than baseline. Personnel in Branch D and G are filtering out the right users to lend money to. Best practices should be shared across the branches
# - Certain features are highly correlated and I have removed them from further analysis as the analysis would be similar
# - Users who owns 2 or 3 credit cards have lower default rates as compared to the rest
# - Users who owns 2 or 3 credit cards also tend to have lower credit limit. I have inferred that this group of users have lower income (due to lower credit limit) and hence, they are more conscious in spending. As such, they do not default as much. This is a point GoFin could take note off when a user of a high income apply for a loan.
# - **delinquency_score is a great feature to ascertain whether a not a user is going to default**. If a user has a delinquency_score > 0, he/she is likely to default 80% of the time. GoFin should not loan money to anyone who has a delinquency_score > 0
# - It is interesting to note that there are defaulters even though outstanding amount is very low. This group of people seems are the **poor and needy** and they require loans to get by their daily lives. They have low spending history and have high payment_ratio than other defaulters. Despite knowing that this group might default, GoFin should nevertheless **help them financially and should spread their repayment plan across a longer period of time, while keeping interest rate low**.
# - First sign of financial duress can be first spotted 3 months ago when user spend drop slightly as compared to his/her longer term spend. GoFin could use this information to **target potential borrowers** and assist them in better financial planning to ensure that they do not default.
# - There is a group of users who spend as per normal despite having financial problems. This group of people have higher credit limit, which implies higher income. GoFin should **charge them higher interest** as this will not only disuade them from overborrowing, GoFin is also able to maximise profits as this higher income group are able to make the higher interest repayment
# - As expected, payment ratios of defaulters are lower than non-defaulters. GoFin could potentially **use these payment ratio metrics as simple method to gauge potential defaulters**. For example, if the interest rate of a loan from GoFin is above their current payment_ratio, then GoFin should not loan money out.
# - Users who took loans from Branch D and G have relatively higher total_3mo_usage_per_limit and payment_ratio as compared to other branches. This could be the reason why Branch D and G have lower default rate than baseline. 


combine.groupby("default_flag").agg({"user_id": "count"}).reset_index()



# the data is skewed towards non-defaulters
print("baseline default rate: " + str(round(1343/13778*100, 1)) + '%')



def cat_var_processing(df, column):
    processed_df = (
        df
        .groupby(["{}".format(column), "default_flag"])
        .agg({"user_id": "count"})
        .reset_index()
        .pivot_table(columns='default_flag', index='{}'.format(column), values='user_id', aggfunc=np.sum)
        .reset_index()
        .rename(columns={0: "non_default", 1: "default"})
        .fillna(0)
        .pipe(lambda x: x.assign(total=x.non_default+x.default))
        .pipe(lambda x: x.assign(total_pctg=100))
        .pipe(lambda x: x.assign(non_default_pctg=round(x.non_default/(x.total)*100, 1)))
        .pipe(lambda x: x.assign(default_pctg=x.total_pctg-x.non_default_pctg))
    )
    
    return processed_df



def plot_cat_var(df, column):  
    overall_plot = sns.barplot(x = df['{}'.format(column)], y = df.total_pctg, color = "red", label='default')
    bottom_plot = sns.barplot(x = df['{}'.format(column)], y = df.non_default_pctg, color = "green", label='non_default')
    _ = plt.title('Breakdown of default status by {}'.format(column))
    _ = plt.legend(loc=4, ncol = 1)
    _ = plt.ylabel('percentage')



branch_defaulters = cat_var_processing(combine, 'branch_code')



branch_defaulters = (
    combine
    .groupby(["branch_code", "default_flag"])
    .agg({"user_id": "count"})
    .reset_index()
    .pivot_table(columns='default_flag', index='branch_code', values='user_id', aggfunc=np.sum)
    .reset_index()
    .rename(columns={0: "non_default", 1: "default"})
    .pipe(lambda x: x.assign(total=x.non_default+x.default))
    .pipe(lambda x: x.assign(total_pctg=100))
    .pipe(lambda x: x.assign(non_default_pctg=round(x.non_default/(x.total)*100, 1)))
    .pipe(lambda x: x.assign(default_pctg=x.total_pctg-x.non_default_pctg))
)



branch_defaulters[['branch_code', 'total', 'default_pctg']]



plot_cat_var(branch_defaulters, 'branch_code')


# Most loans are given out by Branch A. Branch E has the worse default rates out of all the branches and Branch D and G are the better performing ones.
# 
# It would be good for personnel in Branch D and G to share their best practice on how to spot potential defaulters, so that default rate will be lowered.
# 
# Let's take a deeper look into the data on how Branch D and G spot potential defaulters. 
# 
# Before I continue with data exploration, I will first look at the correlation between each variables. I will remove either of the highly correlated variables as there are currently too many variables to look at and highly correlated variables will tell the same story.


# plotting a heatmap of a correlation matrix of all the continuous variables
cont_var_df = combine[['number_of_cards',
       'outstanding', 'credit_limit', 'total_cash_usage',
       'total_retail_usage', 'total_usage', 'total_usage_per_limit',
       'total_3mo_usage_per_limit', 'total_6mo_usage_per_limit',
       'bill', 'remaining_bill', 'payment_ratio', 'payment_ratio_3month',
       'payment_ratio_6month', 'overlimit_percentage', 'delinquency_score',
       'years_since_card_issuing', 'remaining_bill_per_number_of_cards',
       'remaining_bill_per_limit', 'utilization_3month', 'utilization_6month']]

_ = plt.subplots(figsize = (15,10))

corr = cont_var_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(600, 20, as_cmap=True)

_ = sns.heatmap(corr, mask=mask, annot=True, cmap =cmap)


# We can observation strong correlation between:
# - outstanding and bill, outstanding and remaining_bill. I will drop bill and remaining_bill
# - total_usage and total_retail_usage. I will drop total_usage as total_cash_usage is a subset of total_usage
# - payment_ratio and payment_ratio_3month. I will drop payment_ratio_3_month. Although I should note that I have replace 1/3 of payment_ratio with payment_ratio_3month
# - utilization_3month and utilization_6month, utilization_3month and remaining_bill_per_limit. I will drop utilization_3month


# dropping highly correlated variables
combine2 = combine.drop(['bill', 'remaining_bill', 'total_usage', 'payment_ratio_3month', 'utilization_3month'], axis=1)


# ## Categorical Features


no_cards_defaulters = cat_var_processing(combine, 'number_of_cards')



no_cards_defaulters



plot_cat_var(no_cards_defaulters, 'number_of_cards')


# I will disregard observations made on customers having 10 or more cards as the sample size is way too small.
# 
# We can observe that most customer only have 2 or 3 credit cards and default rate (8.6% and 9.1% respectively) of these group of people are below the baseline default rate of 9.7%. 
# 
# Customers having 5 or more cards tend to have a higher default rate. However, it is worth to note that the sample size of these data points are quite small as well.
# 
# Interestingly, the default rate of customers having 1 credit card is above the average default rate. It could be that due to poor credit scoring, they were unable to obtain a second credit card.
# 
# Since having 2 or 3 credit cards is the norm and the default rate of these groups are below the baseline and the fact that the data points in other groups are sparse, I will engineer a new feature to split number_of_cards to two groups, having_2_or_3_cards or not. 


combine2 = (
    combine2
    .pipe(lambda x: x.assign(credit_limit_per_card=x.credit_limit/x.number_of_cards))
    .pipe(lambda x: x.assign(having_2_or_3_cards=np.where(((x.number_of_cards == 2)|(x.number_of_cards == 3)), 1, 0)))
)



_ = sns.boxplot(x='having_2_or_3_cards', y='credit_limit_per_card', data=combine2)
_.set_yscale('log')


# We can see that the credit_limit_per_card of users having 2 or 3 cards is lower that those who have 1 credit card or more than 3 credit cards. Normally credit limit is determine by a person's income. Having a lower credit limit would imply that the user's income is lower as well. Furthermore, we know what the default rate of users having 2 or 3 credit cards are lower than baseline, while the rest are higher. We can infer that perhaps users who are of a lower income group are more down-to-earth and have tigher spends as such, they do not spend excessively that will result in a default.


delinquency_defaulters = cat_var_processing(combine, 'delinquency_score')



delinquency_defaulters



plot_cat_var(delinquency_defaulters, 'delinquency_score')


# Delinquency_score has an obvious correlation to default rate. It is obvious that GoFin should not be giving out loans to users who have 1 or more.


delinquency_more_than_0_defaulters = (
    combine2
    .pipe(lambda x: x.assign(delinquency_score_more_than_0=np.where(x.delinquency_score>=1, 1, 0)))
    .groupby(['delinquency_score_more_than_0', 'branch_code'])
    .agg({"user_id": "count"})
    .reset_index()
    .pivot_table(columns='delinquency_score_more_than_0', index='branch_code', values='user_id', aggfunc=np.sum)
    .reset_index()
    .rename(columns={0: "delinquency_score_0", 1: "delinquency_score_more_than_0"})
    .pipe(lambda x: x.assign(score_more_than_0_pctg=round(x.delinquency_score_more_than_0/(x.delinquency_score_more_than_0+x.delinquency_score_0)*100, 1)))
)



delinquency_more_than_0_defaulters.merge(branch_defaulters, on='branch_code', how='left')[['branch_code', 'score_more_than_0_pctg', 'default_pctg']]


# If Branch D, J & K were to pay more attention to delinquency_score, they would have 0 or close to 0 default rate.
# 
# Not paying attention to delinquency_score is also the main reason why Branch E have such a high default rate.
# 
# As the majority users (~99%) have a delinquency_score of 0, it would be pointless to have this feature in my model. I will drop this feature moving forward.


combine2.drop('delinquency_score', axis=1, inplace=True)


# ## Continuous Features


def plot_cont_var(df, column, boxplot=True, log=True, ax=1, x='default_flag'):
    if boxplot:
        if log:
            _ = sns.boxplot(x='{}'.format(x), y='{}'.format(column), data=df)
            _.set_yscale('symlog')
            _ = plt.title("Boxplot of {} (symlog scale)".format(column))
        else:
            _ = sns.boxplot(x='{}'.format(x), y='{}'.format(column), data=df)
            _ = plt.title("Boxplot of {}".format(column))
    else:
        if log:
            _ = sns.distplot(np.log(df.query("{} == 0".format(x))['{}'.format(column)]), ax=ax, label='non_default')
            _ = sns.distplot(np.log(df.query("{} == 1".format(x))['{}'.format(column)]), ax=ax, label='default')
            _ = plt.title("Distribution of {}".format(column))
            _ = ax.legend()
        else:
            _ = sns.distplot(df.query("{} == 0".format(x))['{}'.format(column)], ax=ax, label='non_default')
            _ = sns.distplot(df.query("{} == 0".format(x))['{}'.format(column)], ax=ax, label='default')
            _ = plt.title("Distribution of {}".format(column))
            _ = ax.legend()



_ = plt.figure(figsize=(25,15))
start = 441
for idx, col in enumerate(['outstanding', 'credit_limit', 'total_cash_usage', 'total_retail_usage',
       'total_usage_per_limit', 'total_3mo_usage_per_limit',
       'total_6mo_usage_per_limit']):
    _ = plt.subplot(start + idx)
    _ = sns.boxplot(x='default_flag', y='{}'.format(col), data=combine2)
    _ = plt.title("Boxplot of {}".format(col))

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)


# Due to data skew, it is hard to see the distribution of each feature. I will re-plot the graphs again using symlog scale or removing skewed data points. 


# re-plotting using symlog scale
_ = plt.figure(figsize=(25,15))
ax1 = plt.subplot(441)
plot_cont_var(combine2, 'outstanding', boxplot=True, log=True, ax=ax1)

ax2 = plt.subplot(442)
plot_cont_var(combine2, 'credit_limit', boxplot=True, log=True, ax=ax2)

ax3 = plt.subplot(443)
plot_cont_var(combine2, 'credit_limit_per_card', boxplot=True, log=True, ax=ax3)


# The symlog scale will set a range around zero within the plot to be linear instead of logarithmic. This allows me to handle zero values.
# 
# We can see that distribution spread of outstanding amount for defaulters is quite wide. Defaulters are skewed towards having higher outstanding amount than non-defaulters. It is interesting to note that **there are defaulters even when the outstanding amount is very low**. I would think that GoFin should pay special attention to these group of people. It seems that **these are the poor and needy** and had to take a loan with GoFin probably to get by their daily lives. From the below table, we can see that these group of people have very **low spending history and their payment_ratio is higher than those defaulters who have high outstanding amount**. We can also observe that although the payment_ratio is higher than their counterpart, payment_ratio_6month is lower. This could mean that their source of income is unstable and could only make payment when they have an inflow of cash. Perhaps, GoFin could **spread their repayment plan across a longer period of time than norm while keeping interest rate low, so to help them financially as well as to not incur defaultment**.
# 
# As for credit limit, not much observation can be made. Similar to the above inference regarding credit_limit_per_card, the credit limit for defaulter is slightly higher than non-defaulters.


# poor and needy group
(
    combine
    .query("default_flag == 1 and outstanding < {}".format(combine2.outstanding.quantile(q=0.25)))
    [['outstanding', 'credit_limit', 'total_retail_usage', 'total_3mo_usage_per_limit', 'total_6mo_usage_per_limit', 'payment_ratio', 
      'payment_ratio_6month', 'remaining_bill_per_number_of_cards', 'remaining_bill_per_limit', 'utilization_6month']]
    .describe()
)



# the rest
(
    combine
    .query("default_flag == 1 and outstanding >= {}".format(combine2.outstanding.quantile(q=0.25)))
    [['outstanding', 'credit_limit', 'total_retail_usage', 'total_3mo_usage_per_limit', 'total_6mo_usage_per_limit', 'payment_ratio', 
      'payment_ratio_6month', 'remaining_bill_per_number_of_cards', 'remaining_bill_per_limit', 'utilization_6month']]
    .describe()
)



# using iqr to spot outliers
def upper_limit(df, column):
    iqr = df['{}'.format(column)].quantile(q=0.75) - df['{}'.format(column)].quantile(q=0.25)
    upper_limit = df['{}'.format(column)].quantile(q=0.75) + 1.5*iqr
    return upper_limit



_ = plt.figure(figsize=(25,15))
ax1 = plt.subplot(441)
total_cash_usage_upper = upper_limit(combine2, 'total_cash_usage')
plot_cont_var(combine2.query("total_cash_usage <= {}".format(total_cash_usage_upper)), 'total_cash_usage', boxplot=True, log=False, ax=ax1)

ax2 = plt.subplot(442)
total_retail_usage_upper = upper_limit(combine2, 'total_retail_usage')
plot_cont_var(combine2.query("total_retail_usage <= {}".format(total_retail_usage_upper)), 'total_retail_usage', boxplot=True, log=False, ax=ax2)

ax3 = plt.subplot(443)
total_usage_per_limit_upper = upper_limit(combine2, 'total_usage_per_limit')
plot_cont_var(combine2.query("total_usage_per_limit <= {}".format(total_usage_per_limit_upper)), 'total_usage_per_limit', boxplot=True, log=False, ax=ax3)

ax4 = plt.subplot(444)
total_3mo_usage_per_limit_upper = upper_limit(combine2, 'total_3mo_usage_per_limit')
plot_cont_var(combine2.query("total_3mo_usage_per_limit <= {}".format(total_3mo_usage_per_limit_upper)), 'total_3mo_usage_per_limit', boxplot=True, log=False, ax=ax4)

ax5 = plt.subplot(445)
total_6mo_usage_per_limit_upper = upper_limit(combine2, 'total_6mo_usage_per_limit')
plot_cont_var(combine2.query("total_6mo_usage_per_limit <= {}".format(total_6mo_usage_per_limit_upper)), 'total_6mo_usage_per_limit', boxplot=True, log=False, ax=ax5)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)


# Instead of using symlog scale, I have filtered out skewed data points to plot out the distribution of these usage features. The reason why I did not use symlog was because some of these features have too many zero values and the charts will become unreadable as well (similar to the first iteration).
# 
# The distribution of total_cash_usage is as such because the majority of total_cash_usage values (~95%) are zero.
# 
# We can see that defaulters tend to have lower usage and the first sign of financial trouble can be observed 3 months ago (total_3mo_usage_per_limit). GoFin could use this information to **target potential borrowers** and assist them in better financial planning to ensure that they do not default.
# 
# It is heartening to see that a majority of the users lowered their spending when they know they have financial issues. GoFin should **take note of the group of users who still spend as per normal despite having financial problems**. From the table below, we can see that the credit limit of those who are not reducing spend is higher than those who had reduced. This shows that their income level is probably higher as well. If GoFin still decides to loan out money to this group, GoFin should **charge them a higher interest**. Not only will it **dissuade them from spending above their current limits**, GoFin would also be able to **maximise profit as this group are more probable in making the high interest repayment**, with their assumed higher income.


# defaulters who are not lowering spend
(
    combine2
    .query("default_flag == 1 and total_retail_usage > {}".format(combine2.total_retail_usage.median()))
    [['total_retail_usage', 'total_3mo_usage_per_limit', 'outstanding', 'credit_limit', 'credit_limit_per_card']]
    .describe()
)



# defaulters who are lowering spend
(
    combine2
    .query("default_flag == 1 and total_retail_usage < {}".format(combine2.total_retail_usage.median()))
    [['total_retail_usage', 'total_3mo_usage_per_limit', 'outstanding', 'credit_limit', 'credit_limit_per_card']]
    .describe()
)



_ = plt.figure(figsize=(25,15))
start = 441
for idx, col in enumerate(['years_since_card_issuing', 'payment_ratio', 'payment_ratio_6month', 'utilization_6month']):
    _ = plt.subplot(start + idx)
    _ = sns.boxplot(x='default_flag', y='{}'.format(col), data=combine2)
    _ = plt.title("Boxplot of {}".format(col))

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)


# It is expected to see that the payment ratios of defaulters are lower than non-defaulters. The difference in payment ratio between defaulters and non-defaulters is more stark when a more recent metric (payment_ratio) is used. Defaulters also tend to have higher utilization rate.
# 
# GoFin could use payment ratios as a simple metric to gauge potential defaulters. For example, if the user would incur an interest of X% when taking a loan from GoFin, GoFin could refer to his/her latest payment_ratio to check whether is the person able to repay the interest rate. If the payment_ratio is below X%, it shows that the user would probably default.

# Defaulters tend to have:
# - higher outstanding amount 
# - higher utilization rate 
# - lower usage amount
# - lower payment ratio 
# 
# Did Branch D and G managed to catch all these?


# creating new column called branch_d_g where entries that are belongs to branch d or g will be 1, the rest 0
combine3= (
    combine2
    .pipe(lambda x: x.assign(branch_d_g=np.where(x.branch_code=='d', 1,
                                                np.where(x.branch_code=='g', 1, 0))))
)



# branch D and G
combine3.query("branch_d_g == 1 and default_flag == 1")[['outstanding', 'utilization_6month', 'total_retail_usage', 'total_usage_per_limit', 'total_3mo_usage_per_limit', 'payment_ratio', 'payment_ratio_6month']].describe()



# other branches
combine3.query("branch_d_g == 0 and default_flag == 1")[['outstanding', 'utilization_6month', 'total_retail_usage', 'total_usage_per_limit', 'total_3mo_usage_per_limit', 'payment_ratio', 'payment_ratio_6month']].describe()



_ = plt.figure(figsize=(25,15))
ax1 = plt.subplot(441)
plot_cont_var(combine3.query("default_flag == 1"), 'outstanding', boxplot=True, log=True, ax=ax1, x='branch_d_g')

ax2 = plt.subplot(442)
_ = sns.boxplot(x='branch_d_g', y='utilization_6month', data=combine3.query("default_flag == 1"))
_ = plt.title("Boxplot of utilization_6month")

ax3 = plt.subplot(443)
total_retail_usage_upper = upper_limit(combine2.query("default_flag == 1"), 'total_retail_usage')
plot_cont_var(combine3.query("default_flag == 1 and total_retail_usage <= {}".format(total_retail_usage_upper)), 'total_retail_usage', boxplot=True, log=False, ax=ax3,  x='branch_d_g')

ax4 = plt.subplot(444)
total_usage_per_limit_upper = upper_limit(combine2.query("default_flag == 1"), 'total_usage_per_limit')
plot_cont_var(combine3.query("default_flag == 1 and total_usage_per_limit <= {}".format(total_usage_per_limit_upper)), 'total_usage_per_limit', boxplot=True, log=False, ax=ax4,  x='branch_d_g')

ax5 = plt.subplot(445)
total_3mo_usage_per_limit_upper = upper_limit(combine2, 'total_3mo_usage_per_limit')
plot_cont_var(combine3.query("default_flag == 1 and total_3mo_usage_per_limit <= {}".format(total_3mo_usage_per_limit_upper)), 'total_3mo_usage_per_limit', boxplot=True, log=False, ax=ax5,  x='branch_d_g')

ax6 = plt.subplot(446)
_ = sns.boxplot(x='branch_d_g', y='payment_ratio', data=combine3.query("default_flag == 1"))
_ = plt.title("Boxplot of payment_ratio")


ax7 = plt.subplot(447)
_ = sns.boxplot(x='branch_d_g', y='payment_ratio_6month', data=combine3.query("default_flag == 1"))
_ = plt.title("Boxplot of payment_ratio_6month")


plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.5, wspace=0.35)


# We can see that users who took loans from Branch D and G have relatively higher total_3mo_usage_per_limit and payment_ratio as compared to other branches. This resulted in them having lower default rates as compared to other branches.

# # Model

# ## Summary
# - Two models output were not great. Accuracy did not beat baseline accuracy of 90.3%
# - However, the recall score of the first model is good
# - When it comes to model evaluation, GoFin should be more concern with recall score as it tells us how many actual defaulters are selected
# - GoFin needs to balance the cost of have too many False Positives vs False Negatives
# - Payment ratio and usage level are the most important features to predict defaulters
# - Branch D and G did a great job in noticing them
# - Could have tried boosting models


import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, make_scorer

from sklearn.externals import joblib

from imblearn.over_sampling import SMOTE



combine2.head(2)



y = combine2.default_flag.values



# dropping total_cash_usage because majority of the values are 0
# creating new feature called is_overlimit, where 1 = overlimit_percentage > 0 and 0 = overlimit_percentage = 0
X = (
    combine2
    .pipe(lambda x: x.assign(is_overlimit=np.where(x.overlimit_percentage>0, 1, 0)))
    .drop(['user_id', 'branch_code', 'default_flag', 'number_of_cards', 'total_cash_usage', 'overlimit_percentage', 'credit_limit'], axis=1)
)



X.head()



# checking for multi-collinearity amongst X matrix
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif



# getting training set (77%) and testing set (33%)
# stratified so that the split of the imbalance dataset between train and test set will be the same
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 3, stratify=y)



# scale X matrix with StandardScaler
ss = StandardScaler()
Xss_train = ss.fit_transform(X_train)

Xss_test = ss.transform(X_test)

# SMOTE the training set as the data set is skewed towards having more non_defaulters
sm = SMOTE(random_state = 1, ratio = 'minority')
Xss_sm_train, y_sm_train = sm.fit_sample(Xss_train, y_train)



# using RandomForestClassifer as an estimator
rclf = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=1)

# train 30% of the training set
rclf.fit(Xss_sm_train, y_sm_train)



gs_params ={
    'criterion': ['gini'],
    'max_depth': [None,1,5,10],
    'max_features': ['auto',3,7],
    'n_estimators':[200, 500, 1000],
    'random_state':[1]
}

rclf_gs = GridSearchCV(RandomForestClassifier(), gs_params, n_jobs = -1, verbose = 1, cv = 3, scoring='recall')



rclf_gs.fit(Xss_sm_train, y_sm_train)



rclf_gs.best_params_



# using gs best parameters
rclf_gs_best = rclf_gs.best_estimator_



# function to evaulate performance of model
def sup_perf(model, X_test, y_test):
    model_yhat = model.predict(X_test)
    model_score = model.score(X_test, y_test)
    model_f1 = f1_score(y_test, model_yhat, average = 'binary')
    model_precision = precision_score(y_test, model_yhat, average = 'binary')
    model_recall = recall_score(y_test, model_yhat, average = 'binary')
    model_auc = roc_auc_score(y_test, model_yhat, average = 'macro')
    
    return (model_score, model_precision, model_recall, model_f1, model_auc)



score_rclf, precision_rclf, recall_rclf, f1_rclf, auc_rclf = sup_perf(rclf, Xss_test, y_test)



score_rclf_gs_best, precision_rclf_gs_best, recall_rclf_gs_best, f1_rclf_gs_best, auc_rclf_gs_best = sup_perf(rclf_gs_best, Xss_test, y_test)



rclf_yhat = rclf.predict(Xss_test)
print (classification_report(y_test, rclf_yhat, labels = [0,1], target_names=['non_default','default']))



# confusion matrix based normal random forest
conmat = pd.DataFrame(confusion_matrix(y_test, rclf_yhat, labels = [0,1]),
                      index = ['actual_non_default','actual_default'],
                      columns = ['predict_non_default','predict_default'])
conmat



rclf_gs_best_yhat = rclf_gs_best.predict(Xss_test)
print (classification_report(y_test, rclf_gs_best_yhat, labels = [0,1], target_names=['non_default','default']))



# confusion matrix based on random forest with grid search
conmat = pd.DataFrame(confusion_matrix(y_test, rclf_gs_best_yhat, labels = [0,1]),
                      index = ['actual_non_defau;t','actual_default'],
                      columns = ['predict_non_default','predict_default'])
conmat



model_comparison = pd.DataFrame({'rclf':[score_rclf, precision_rclf, recall_rclf, f1_rclf, auc_rclf],                                     'rclf (grid_search)':[score_rclf_gs_best, precision_rclf_gs_best, recall_rclf_gs_best, f1_rclf_gs_best, auc_rclf_gs_best]},                                    index = ['accuracy score','precision score (fraud)','recall score (fraud)','f1 score (fraud)','auc score (unweighted mean)'])
model_comparison



# observing feature importance of normal random forest model
feature_impt = pd.DataFrame({'importance': rclf.feature_importances_, 'features':X.columns})
feature_impt.sort_values('importance', ascending = False).reset_index(drop=True)

