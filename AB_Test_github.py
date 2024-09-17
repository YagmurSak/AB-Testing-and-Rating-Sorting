## Comparison Bidding Methods with A/B Testing

## BUSINESS PROBLEM

# This data set, which contains information about a company's website, includes information such as the number of ads
# that users have seen and clicked on, as well as the income information from there. There are two separate data sets,
# Control and Test groups. Maximum Bidding was applied to the control group, and AverageBidding was applied to the test group.

## VARIABLES

# impression: Number of ad views
# Click: Number of clicks on the displayed ad
# Purchase: Number of products purchased after clicking on the ads
# Earning: Earnings after purchasing products

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


## Data Analysis and First Insights

control_group = pd.read_excel("C:/Users/Yagmu/OneDrive/Masaüstü/DATA SCIENCE BOOTCAMP/3-Measurement Problems/ABTesti/ab_testing.xlsx", sheet_name="Control Group")

test_group = pd.read_excel("C:/Users/Yagmu/OneDrive/Masaüstü/DATA SCIENCE BOOTCAMP/3-Measurement Problems/ABTesti/ab_testing.xlsx", sheet_name="Test Group")

control_group.head()
control_group.describe().T
control_group.shape

test_group.head()
test_group.describe().T
test_group.shape


## Combining control and test groups


control_group["group"] = "control"   ## to show the difference in the new dataframe
test_group["group"] = "test"

new_df = pd.concat([control_group,test_group], ignore_index = True)


### Defining the Hypothesis of A/B Testing

#H0: M1 = M2 (There is no difference between the conversions of averagebidding and maximumbidding.)
#H1: M1 != M2 (there is)


## Averages of purchase

control_group["Purchase"].mean()   #550.8940
test_group["Purchase"].mean()      #582.1060


###### CONDUCTING HYPOTHESIS TESTING #####

# Normality Assumption

# H0: Normal distribution assumption is met.(+)
# H1:..not met.

test_stat, pvalue = shapiro(new_df.loc[new_df["group"] == "control", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value = 0,58 > 0.05. So HO CANNOT BE REJECTED.


test_stat, pvalue = shapiro(new_df.loc[new_df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


# p-value = 0.15 > 0.05. So HO CANNOT BE REJECTED.



#Variance Homogeneity Assumption

# H0: Variances are Homogeneous
# H1: Variances are Not Homogeneous


test_stat, pvalue = levene( new_df.loc[new_df["group"] ==  "control", "Purchase"],
                           new_df.loc[new_df["group"] == "test" , "Purchase"] )
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#p-value = 0.10 > 0.05. So HO CANNOT BE REJECTED.


#Assumptions met, independent two sample t-test (parametric test)

test_stat, pvalue = ttest_ind(new_df.loc[new_df["group"] == "control" , "Purchase"],
                           new_df.loc[new_df["group"] == "test", "Purchase"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value = 0.34 > 0.05. So HO CANNOT BE REJECTED.


#H0 (There is no difference between the conversions of averagebidding and maximumbidding) CANNOT BE REJECTED. SO THERE IS NO DIFFERENCE.




