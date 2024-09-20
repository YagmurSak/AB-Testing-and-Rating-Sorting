### PROJECT: RATING PRODUCT & SORTING REVIEWS IN AMAZON ###
#
###BUSINESS PROBLEM

# One of the most important problems in e-commerce is the correct calculation of the scores given to products after sales.
#The solution to this problem means providing more customer satisfaction for the e-commerce site, highlighting the product for
#sellers and a smooth shopping experience for buyers. Another problem is the correct ordering of the comments given to the products.
#Since the prominence of misleading comments will directly affect the sales of the product, it will cause both financial loss and
# loss of customers. In the solution of these 2 basic problems, the e-commerce site and sellers will increase their sales while
# customers will complete the purchasing journey without any problems.

# Dataset Story
#################################################

# This dataset, which contains Amazon product data, includes product categories and various metadata.
# The product with the most comments in the electronics category has user ratings and comments.

# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: User Name
# helpful: Helpful review rating
# reviewText: Review
# overall: Product rating
# summary: Review summary
# unixReviewTime: Review time
# reviewTime: Review time Raw
# day_diff: Number of days since review
# helpful_yes: Number of times the review was found helpful
# total_vote: Number of votes given to the review

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv(r"C:\Users\Yagmu\OneDrive\Masaüstü\DATA SCIENCE BOOTCAMP\3-Measurement Problems\RatingProductSortingReviewsinAmazon-221119-111357\Rating Product&SortingReviewsinAmazon\amazon_review.csv")
df.head()

## Average of Product Rating

df["overall"].mean()

## Weighted Point Average by Date

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 100, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 100) & (dataframe["day_diff"] <= 200), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 200) & (dataframe["day_diff"] <= 500), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 500), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

## Identifying useless reviews

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

## Defining functions

def score_pos_neg_diff(up,down):
    return up - down


def score_average_rating(up,down):
    if up + down == 0:
        return 0
    return up / (up + down)

def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


## Difference between useful and useless review

df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"],
                                                                 x["helpful_no"]), axis=1)


## Ratio of helpful reviews to all reviews

df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"],
                                                                   x["helpful_no"]), axis=1 )


## Wilson lower score (The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"],
                                                                   x["helpful_no"]), axis=1 )

## 20 Reviews to be displayed on the Product Detail Page for the Product

df.sort_values("wilson_lower_bound" , ascending=False)[:20]

df["wilson_lower_bound"].max()


