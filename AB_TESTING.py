#####################################################
# AB Test: Comparing Conversion Rates of Bidding Methods
#####################################################

#####################################################
# Business Problem
#####################################################
# Facebook recently introduced a new bidding type called "average bidding"
# as an alternative to the existing "maximum bidding".
# One of our clients, bombabomba.com, decided to test this feature to see if
# "average bidding" brings more conversions than "maximum bidding".
# The A/B test has been running for 1 month, and bombabomba.com
# now expects an analysis of the results.
# The ultimate success metric for bombabomba.com is Purchase.
# Therefore, the analysis and hypothesis testing should focus on the Purchase metric.

#####################################################
# Dataset Story
#####################################################
# This dataset contains website data including impressions, clicks,
# purchases, and revenue. There are two separate datasets for the Control
# group and the Test group, stored in different sheets of the Excel file
# "ab_testing.xlsx". Maximum Bidding was applied to the Control group,
# and Average Bidding to the Test group.
#
# Variables:
# impression: Number of ad impressions
# Click: Number of clicks on the ads
# Purchase: Number of products purchased after clicks
# Earning: Revenue earned after purchases

#####################################################
# Task 1: Data Preparation and Exploration
#####################################################

import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Step 1: Read the control and test datasets
df_control = pd.read_excel("Dataset/ab_testing.xlsx", sheet_name="Control Group")
df_test = pd.read_excel("Dataset/ab_testing.xlsx", sheet_name="Test Group")

# Step 2: Data inspection function
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Describe #####################")
    print(dataframe.describe([0, 0.25, 0.50, 0.75 ,0.95, 0.99, 1]).T)

check_df(df_control)
check_df(df_test)

# Step 3: Merge datasets
df_control["group"] = "control"
df_test["group"] = "test"
df = pd.concat([df_control, df_test], axis=0, ignore_index=True)

#####################################################
# Task 2: Defining the Hypothesis
#####################################################
# H0: M1 = M2 → There is no significant difference in the mean Purchase values
# between the control and test groups.
# H1: M1 != M2 → There is a significant difference in the mean Purchase values
# between the control and test groups.

# Step 1: Mean Purchase by group
print(df.groupby("group").agg({"Purchase": "mean"}))

#####################################################
# Task 3: Hypothesis Testing
#####################################################

# Step 1: Assumption Checks

# Normality Assumption (Shapiro-Wilk Test)
test_stat, pvalue = shapiro(df.loc[df["group"] == "control", "Purchase"])
print(f'Control Group → Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
# p > 0.05 → Fail to reject H0 → Normally distributed

test_stat, pvalue = shapiro(df.loc[df["group"] == "test", "Purchase"])
print(f'Test Group → Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
# p > 0.05 → Fail to reject H0 → Normally distributed

# Variance Homogeneity (Levene Test)
test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print(f'Levene Test → Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
# p > 0.05 → Fail to reject H0 → Variances are equal

# Step 2: Independent Two-Sample t-Test (parametric)
test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)
print(f'Two-Sample t-Test → Test Stat = {test_stat:.4f}, p-value = {pvalue:.4f}')
# p-value > 0.05 → Fail to reject H0

#####################################################
# Task 4: Analysis of the Results
#####################################################

# Step 1: Summary of Statistical Findings
# - Test Used: Independent Two-Sample t-Test
# - Reason: Both the Normality and Variance Homogeneity assumptions were satisfied.
# - p-value = 0.3493 (> 0.05) → Fail to reject H0

# Step 2: Interpretation
# There is no statistically significant difference in the average Purchase values
# between the control group (Maximum Bidding) and the test group (Average Bidding).
# This means that, based on the data collected over the 1-month experiment,
# Average Bidding did not lead to higher conversions compared to Maximum Bidding.

# Step 3: Business Impact
# Since the ultimate goal for bombabomba.com is to maximize conversions (Purchases),
# and no significant improvement was detected, there is no data-driven reason to switch
# to the Average Bidding method at this time.

# Step 4: Recommendation
# - Continue using the existing Maximum Bidding strategy unless:
#       a) Additional experiments over a longer time period yield different results
#       b) Other KPIs (e.g., cost per conversion, ROI) show improvements with Average Bidding
# - If management is still interested in Average Bidding, run a longer test,
#   possibly with a larger sample size and seasonal variations accounted for.


import matplotlib.pyplot as plt
import numpy as np

# Mean, std, and confidence intervals
summary_stats = df.groupby("group")["Purchase"].agg(["mean", "std", "count"])
summary_stats["sem"] = summary_stats["std"] / np.sqrt(summary_stats["count"])
summary_stats["ci95_low"] = summary_stats["mean"] - 1.96 * summary_stats["sem"]
summary_stats["ci95_high"] = summary_stats["mean"] + 1.96 * summary_stats["sem"]
print(summary_stats)

# Effect size (Cohen's d) Cohen’s D → Measures the size of the difference (P-Value only answers the question “Is there”).
def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

d_value = cohens_d(df.loc[df["group"]=="test", "Purchase"],
                   df.loc[df["group"]=="control", "Purchase"])
print(f"Cohen's d = {d_value:.4f}")

# Visualization: Box plot
#Logic: It is not enough to look at P-Value; The size of the difference, confidence intervals and visualization support the business decision.

plt.figure(figsize=(6,4))
df.boxplot(column="Purchase", by="group")
plt.title("Purchase Distribution by Group")
plt.suptitle("")
plt.ylabel("Purchase")
plt.show()

# Visualization: Bar plot with error bars
plt.figure(figsize=(6,4))
plt.bar(summary_stats.index, summary_stats["mean"], yerr=summary_stats["sem"], capsize=5)
plt.title("Mean Purchase with 95% CI")
plt.ylabel("Purchase")
plt.show()

#Summary logic flow
#Prepare data → Sept and Combine Control/Test Groups.

#Hypothesis → H0: No difference, H1: There is a difference.

#Assumption controls → Normal distribution and variance homogeneity.

#Appropriate statistical test → Independent T-test.

#Analysis of the results → P-Value, Effect you, confidence interval, visualization.


#Make your job decision → If the new method has not made a significant difference, continue to the current method.
