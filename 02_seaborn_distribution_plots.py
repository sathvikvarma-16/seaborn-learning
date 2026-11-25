# ============================================================
# SEABORN PHASE 2: DISTRIBUTION PLOTS
# ============================================================
# Install if needed:
# pip install numpy pandas matplotlib seaborn

# LEARNING OBJECTIVES
# 1. histplot()  : frequency distribution using bins
# 2. kdeplot()   : smooth estimated density
# 3. boxplot()   : median, quartiles, IQR, whiskers, potential outliers
# 4. violinplot(): distribution shape across categories
# 5. ecdfplot()  : cumulative distribution

# ML relevance:
# Inspect feature distributions, skewness, spread, unusual values,  missingness, and differences between target classes before modeling.
# A plot is exploratory evidence, not proof. An unusual value is not automatically an error, and not every ML feature must be normal.
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", context="notebook")

# ------------------------------------------------------------
# 0. PRACTICE DATASET
# ------------------------------------------------------------

# Synthetic customer data. It is for learning, not real-world inference.

rng = np.random.default_rng(42) # Creates a random number generator.
n = 500 # represents customers we want to generate

customer_type = rng.choice(["Standard", "Premium"], n, p=[0.65, 0.35]) # rng.choice() → Randomly selects values from the given options.
# ["Standard", "Premium"] → Two possible customer types.
# n generates 500 random selections || p represents probability of selecting each type

age = np.clip(rng.normal(35, 10, n), 18, 75).round().astype(int) # 	 Restrict values within a given range
# 35 -> average age (mean) | 10 -> std deviation (spread of ages) | n -> number of ages to generate

income = rng.lognormal(mean=10.5, sigma=0.55, size=n) # Generates random numbers following a log-normal distribution.
# mean=10.5	Mean of the underlying normal distribution -> It is the mean of the logarithms of the generated values.
# sigma=0.55	Standard deviation of the underlying normal distribution

spend = rng.normal(1500, 350, n) + np.where(customer_type == "Premium", 600, 0) # If the customer is "Premium", add ₹600, otherwise, add ₹0.
# This code generates random customer spending amounts, with Premium customers spending an additional ₹600 on average.
# 1500 → Average spending amount. || 350 → Standard deviation, which controls how much spending varies

spend = np.clip(spend, 100, None) # np.clip(array, minimum, maximum)

# Imagine we have 500 customers. We want to create a dataset where some customers leave the company and others stay.
# This is called customer churn.
# - churn = 1 means the customer leaves.
# - churn = 0 means the customer stays.

churn_probability = 0.12 + 0.12 * (customer_type == "Standard") + 0.08 * (spend < 1300) # This line calculates the probability of each customer leaving.
# part a : each customer starts with 0.12% chance of leaving
# This checks whether each customer is a Standard customer.

'''
For example:
customer_type = ["Standard", "Premium", "Standard", "Premium"]

The comparison produces:
[True, False, True, False]

In Python calculations:
- True behaves like 1.
- False behaves like 0.
Now multiply by 0.12:
0.12 * [True, False, True, False]

Result:
[0.12, 0.00, 0.12, 0.00]

So Standard customers get an additional 12% churn probability.

Part C: (spend < 1300)
spend < 1300

This checks whether each customer's spending is below ₹1,300.
For example:
spend = [1000, 1500, 900, 2000]

The comparison produces:
[True, False, True, False]

Now multiply by 0.08:
0.08 * [True, False, True, False]

Result:
[0.08, 0.00, 0.08, 0.00]
So customers spending below ₹1,300 get an additional 8%.

Let's combine all three parts
Suppose we have these four customers:
Customer	Type	Spending
A	     Standard	₹1,000
B	     Premium	₹1,500
C	     Standard	₹900
D	     Premium	₹2,000

Their churn probabilities will be:
Customer	Base 12%	Standard bonus	Low spending bonus	Total
A	          12%	        12%	             8%	            32%
B	          12%        	0%             	 0%	            12%
C	          12%	        12%	             8%	            32%
D	          12%	        0%	             0%	            12%

For example, Customer A:
0.12 + 0.12(1) + 0.08(1) = 0.32
That means Customer A has a 32% chance of leaving.
Important: This is only the probability. We haven't decided whether the customer actually leaves yet.
'''

churn = rng.binomial(1, np.clip(churn_probability, 0.02, 0.85)) # This line generates the actual result: 0 or 1.

'''
Part A: np.clip(churn_probability, 0.02, 0.85)
You already calculated the churn probabilities in Step 1.
For example:
churn_probability = [0.32, 0.12, 0.32, 0.12]

np.clip() keeps probabilities within a specified range.
np.clip(churn_probability, 0.02, 0.85)

Here:
- 0.02 means minimum probability is 2%.
- 0.85 means maximum probability is 85%.


Part B: rng.binomial(1, probability)
This is the most important part.

Think of flipping a coin, but instead of having a 50% chance, we give each customer their own probability of leaving.
The syntax is:
rng.binomial(number_of_trials, probability)
In our code:
rng.binomial(1, churn_probability)
- 1 → Perform one random trial for each customer.
- churn_probability → The probability of getting the result 1. 
result 1 -> leaves
result 0 -> customer stays
A 32% probability does not guarantee that the customer will leave. It means there is a 32% chance of getting 1.
'''


df = pd.DataFrame({  # Create a DataFrame using the generated customer data
    "age": age,  # Add customer ages
    "income": income.round(2),  # Add income values rounded to 2 decimal places
    "monthly_spend": spend.round(2),  # Add monthly spending rounded to 2 decimal places
    "customer_type": customer_type,  # Add customer types (Standard or Premium)
    "churn": churn  # Add churn status (1 = left, 0 = stayed)
})

# Add a few extreme values and one missing value for demonstration.
df.loc[[10, 40, 90], "income"] *= 4  # Multiply income by 4 for rows 10, 40, and 90
df.loc[25, "income"] = np.nan  # Set the income value in row 25 as missing (NaN)

print("\nDATA PREVIEW")  # Print the heading for the data preview
print(df.head())  # Display the first 5 rows of the DataFrame

print("\nDATA SUMMARY")  # Print the heading for the data summary
print(df.describe(include="all"))  # Display summary statistics for all columns

print("\nMISSING VALUES")  # Print the heading for missing values
print(df.isna().sum())  # Count the missing values in each column


# ============================================================
# 1. HISTOGRAM: sns.histplot()
# ============================================================
# A histogram divides numerical values into intervals (bins).
# Bar height is the count by default.
#
# Common uses:
# - inspect distribution shape, skewness, peaks and tails
# - compare numerical distributions across categories
#
# Useful parameters:
# bins        : number of bins or explicit bin boundaries
# binwidth    : width of each bin
# binrange    : range covered by bins
# stat        : "count", "frequency", "probability", "percent", "density"
# kde         : overlay a KDE curve
# hue         : split observations by category
# multiple    : "layer", "stack", "dodge", or "fill"
# element     : "bars", "step", or "poly"
# cumulative  : cumulative histogram
# common_bins : use common bin edges for hue groups
# common_norm : normalize groups together or separately
#
# Interpretation:
# right tail longer = right-skew; left tail longer = left-skew.
# Several peaks may indicate subgroups. Isolated bars need investigation.
# Histogram appearance depends on bin choice, so compare reasonable bins.

# 1A. Basic histogram.
plt.figure(figsize=(9, 5))  # Set the figure size to 9 inches wide and 5 inches tall
sns.histplot(data=df, x="income", bins=30)  # Create a histogram of income values using 30 bins
plt.title("Income Distribution")  # Set the chart title
plt.xlabel("Income")  # Set the x-axis label
plt.ylabel("Count")  # Set the y-axis label (number of customers)
plt.tight_layout()  # Adjust spacing to prevent labels and titles from overlapping
plt.show()  # Display the histogram

# 1B. Histogram with a KDE overlay. kde=True adds a KDE (Kernel Density Estimation) curve over the histogram.
# 1. Histogram: Shows how many customers fall into each income range using bars.
# 2. KDE curve: Shows a smooth curve representing the overall distribution of income.
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="monthly_spend", bins=25, kde=True)
plt.title("Monthly Spend: Histogram + KDE") 
plt.tight_layout()
plt.show()


# 1C. Density histogram. Total bar area is approximately 1.
# Useful for comparing distributions with different sample sizes.
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="monthly_spend", bins=25, stat="density")
# Displays density instead of the actual customer count on the Y-axis.
plt.title("Monthly Spend Density Histogram") 
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# 1D. Overlay distributions by category.
plt.figure(figsize=(9, 5))  # Set the figure size to 9 inches wide and 5 inches tall

sns.histplot(
    data=df,  # Use the customer DataFrame
    x="monthly_spend",  # Display monthly spending on the X-axis
    hue="customer_type",  # Use different colors for Standard and Premium customers
    bins=25,  # Divide the spending range into 25 intervals
    multiple="layer",  # Draw both customer distributions on top of each other
    alpha=0.4,  # Set transparency to 40% so overlapping bars are visible
    kde=True  # Add a smooth density curve for each customer type
)
plt.title("Monthly Spend by Customer Type")
plt.tight_layout()
plt.show()

# 1E. Stacked histogram.
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,  # Use the customer DataFrame
    x="monthly_spend",  # Display monthly spending on the X-axis
    hue="customer_type",  # Separate Standard and Premium customers using different colors
    bins=25,  # Divide the spending range into 25 intervals
    multiple="stack"  # Stack the customer type bars on top of each other
)
plt.title("Stacked Monthly Spend Distribution")
plt.tight_layout()
plt.show()

# 1F. Step histogram, useful when filled bars overlap.
plt.figure(figsize=(9, 5))
sns.histplot(
    data=df,  # Use the customer DataFrame
    x="income",  # Display income values on the X-axis
    hue="customer_type",  # Use different colors for Standard and Premium customers
    bins=25,  # Divide the income range into 25 intervals
    element="step",  # Draw histogram bars as step-shaped outlines
    fill=False,  # Do not fill the histogram bars with color
    common_bins=True  # Use the same bin intervals for both customer types
)
plt.title("Income by Customer Type")
plt.tight_layout()
plt.show()

# 1G. Cumulative histogram: shows the running total of observations as we move from lower values to higher values.
'''
For example:
Monthly spending	Number of customers
₹500–₹1,000	             20
₹1,000–₹1,500          	 30
₹1,500–₹2,000	         25
₹2,000–₹2,500	         15

Each bar shows the count within that particular interval.

2. What is a cumulative histogram?
Instead of showing individual counts, it keeps adding the previous counts.
Monthly spending	Normal count	Cumulative count
₹500–₹1,000	           20	             20
₹1,000–₹1,500	       30	             50
₹1,500–₹2,000	       25	             75
₹2,000–₹2,500	       15	             90
'''
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="monthly_spend", bins=25, cumulative=True)
plt.title("Cumulative Monthly Spend")
plt.ylabel("Cumulative Count")
plt.tight_layout()
plt.show()

# 1H. Choose bin width instead of number of bins.
plt.figure(figsize=(9, 5))  # Set the figure size to 9 × 5 inches
sns.histplot(
    data=df,  # Use the customer DataFrame
    x="age",  # Display customer ages on the X-axis
    binwidth=5,  # Make each bin cover a 5-year age interval -> binwidth specifies how wide each histogram interval should be.
    # 15–20	Customers aged 15 to 20
    binrange=(15, 80)  # Set the histogram range from age 15 (starting boundary) to 80 (ending boundary)
)
plt.title("Age Distribution: 5-Year Bins")  # Set the chart title
plt.tight_layout()  # Adjust spacing
plt.show()  # Display the histogram


# ============================================================
# 2. KDE PLOT: sns.kdeplot()
# ============================================================
# KDE = Kernel Density Estimation.
# It estimates a smooth density curve from observed numerical values.
# It is NOT a count plot and the curve is NOT the raw observations.
#
# Useful parameters:
# hue         : separate curves by category
# fill        : shade the area under the curve
# bw_adjust   : smoothing control; smaller = more detail,
#               larger = smoother
# cut         : how far curve extends beyond observed data
# clip        : restrict estimate to a range
# common_norm : normalize groups together or separately
# cumulative  : cumulative density estimate
#
# KDE can be unstable for small samples. Smoothing may hide details
# or create misleading shapes. Inspect the sample size and histogram too.

# 2A. Basic KDE.
plt.figure(figsize=(9, 5))
sns.kdeplot(data=df, x="monthly_spend") 
plt.title("KDE of Monthly Spend")
plt.tight_layout()
plt.show()

# 2B. Filled KDE.
plt.figure(figsize=(9, 5))
sns.kdeplot(data=df, x="monthly_spend", fill=True, alpha=0.4)
plt.title("Filled KDE")
plt.tight_layout()
plt.show()

# 2C. Compare categories. common_norm=False normalizes each group separately.
plt.figure(figsize=(9, 5))
sns.kdeplot(
    data=df, x="monthly_spend", hue="customer_type",
    fill=True, alpha=0.25, common_norm=False # -> each groups normalized seperately if ture both groups normalized together
)
plt.title("Monthly Spend KDE by Customer Type")
plt.tight_layout()
plt.show()

# 2D. Compare smoothing bandwidths.
# Bandwidth controls how much smoothing is applied when Seaborn creates the KDE curve.
# there is small bandwidth, large bandwidth, small bw - less smoothing and more detail
fig, axes = plt.subplots(1, 3, figsize=(15, 4)) # 1 row, 3 columns
for ax, bandwidth in zip(axes, [0.5, 1.0, 2.0]):
    sns.kdeplot(
        data=df, x="monthly_spend", bw_adjust=bandwidth, 
        fill=True, ax=ax # -> use same axes -> fill = True means fill with color
    )
    ax.set_title(f"bw_adjust = {bandwidth}") # varies bandwidth in each case
plt.suptitle("KDE Smoothing Comparison")
plt.tight_layout()
plt.show()

# 2E. cut=0 prevents the curve extending beyond observed values.
plt.figure(figsize=(9, 5))
sns.kdeplot(data=df, x="age", fill=True, cut=0) # cut=0 controls how far the KDE curve extends beyond the minimum and maximum data values.
# cut=0	Stop the KDE curve at the data boundaries
# cut=1	Extend approximately 1 bandwidth beyond each boundary
# cut=2	Extend approximately 2 bandwidths beyond each boundary
# cut=3	Default, extend approximately 3 bandwidths
plt.title("Age KDE Restricted to Data Range")
plt.tight_layout()
plt.show()

# 2F. Cumulative KDE.
plt.figure(figsize=(9, 5))
sns.kdeplot(data=df, x="monthly_spend", cumulative=True)
plt.title("Cumulative KDE")
plt.ylabel("Cumulative Probability")
plt.tight_layout()
plt.show()


# ============================================================
# 3. BOXPLOT: sns.boxplot()
# ============================================================
# A boxplot summarizes a distribution:
# Q1 = 25th percentile; median = 50th; Q3 = 75th percentile.
# IQR = Q3 - Q1 (middle 50% of observations).
# Default whiskers extend to the most extreme values within 1.5*IQR.
# Points beyond whiskers are called fliers/potential outliers.
#
# IMPORTANT: A flier is not automatically a bad value. Investigate it.
#
# Useful parameters:
# hue         : compare subgroups
# order       : set category order
# whis        : whisker rule (default 1.5 IQR)
# showfliers  : show/hide flier markers (does NOT remove data)
# fliersize   : marker size for fliers
# width       : box width
# dodge       : separate hue groups
# orient      : orientation
#
# Read a vertical box:
# bottom/top of box = Q1/Q3; center line = median;
# whiskers = range allowed by whisker rule; dots = fliers.

# 3A. Basic boxplot.
plt.figure(figsize=(8, 5))  # Set the figure size to 8 inches wide and 5 inches tall
sns.boxplot(data=df, y="income")  # Create a vertical box plot to show the income distribution
plt.title("Income Boxplot")  # Set the chart title
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()  # Display the box plot

# 3B. Compare numerical feature across categories.
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="customer_type", y="monthly_spend")
# Places customer categories (Standard and Premium) on the X-axis, y="monthly_spend": Places monthly spending values on the Y-axis
plt.title("Monthly Spend by Customer Type")
plt.tight_layout()
plt.show()

# 3C. Compare by category and target.
plt.figure(figsize=(9, 5))  # Set the figure size to 9 inches wide and 5 inches tall
sns.boxplot(
    data=df,  # Use the customer DataFrame
    x="customer_type",  # Display customer types on the X-axis
    y="monthly_spend",  # Display monthly spending on the Y-axis
    hue="churn",  # Separate each customer type by churn status (0 = stayed, 1 = left)
    # hue is used to separate data into different groups using different colours.
    # 0 = Customer stayed, 1 = Customer left.
    # Without hue: Compare Standard vs Premium, With hue="churn": Compare Standard vs Premium, and also separate stayed vs left customers.
    palette="Set2"  # Apply the Set2 color palette - Soft, pleasant colours
    # palette = Which colour scheme should be used for those groups?
)
plt.title("Monthly Spend by Customer Type and Churn")  # Set the chart title
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()  # Display the box plot


# 3D. Hide fliers visually. The rows remain in the DataFrame.
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, y="income", showfliers=False) # if showfliers = True, displays outliers as individual dots - helps identify unusual highs or low values
# if showfliers = false -> Hides the individual outlier dots.
plt.title("Income Boxplot (Fliers Hidden, Not Deleted)")
plt.tight_layout()
plt.show()

# 3E. Wider whisker rule. By default, Seaborn uses whis=1.5. => his means whiskers extend up to 1.5 times the IQR from Q1 and Q3 =>IQR = Q3 − Q1
# Lower whisker: Extends towards the lowest non-outlier value.
# Upper whisker: Extends towards the highest non-outlier value.
plt.figure(figsize=(8, 5))  # Set the figure size to 8 inches wide and 5 inches tall
sns.boxplot(data=df, y="income", whis=2.0)  # Create a box plot and extend whiskers up to 2 times the IQR
plt.title("Income Boxplot with whis=2.0")  # Set the chart title
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()  # Display the box plot


# 3F. Horizontal boxplot. - Seaborn automatically decides whether to draw a horizontal or vertical box plot 
# based on which variable you assign to x and y.
'''
Code arrangement	Output
x="category", y="numeric"	Vertical
x="numeric", y="category"	Horizontal
'''
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="monthly_spend", y="customer_type")
plt.title("Horizontal Monthly Spend Boxplot")
plt.tight_layout()
plt.show()


# ============================================================
# 4. VIOLINPLOT: sns.violinplot()
# ============================================================
# A violin plot displays a KDE-like distribution shape by category.
# Wider sections indicate higher estimated density.
# It combines distribution shape with optional internal summaries.

# Useful parameters:
# hue          : additional grouping variable
# split        : split two hue distributions into opposite halves
# inner        : "box", "quart", "point", or None
# cut          : curve extension beyond observed data
# bw_adjust    : KDE smoothing
# density_norm : "area", "count", or "width" in newer Seaborn
# dodge        : separate hue groups
# order        : category order
#
# Violin width is estimated density, not a count of individual points.
# For small samples, violin shapes may be unreliable.

# 4A. Basic violin.
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="customer_type", y="monthly_spend")
plt.title("Monthly Spend Violin Plot") # Wider portions indicate more observations concentrated around those values.
plt.tight_layout() # Narrower portions indicate fewer observations.
plt.show() # Think of it as a density distribution displayed symmetrically around its centre.

# 4B. Show an internal box. - Adds a miniature box plot inside each violin.
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df, x="customer_type", y="monthly_spend", inner="box" # This combines distribution shape with box plot information.
)
plt.title("Violin Plot with Inner Box")
plt.tight_layout()
plt.show()
 
# 4C. Show quartiles. Shows three horizontal lines:
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df, x="customer_type", y="monthly_spend", inner="quart"
)
# 25th percentile (Q1), 50th percentile (Median), 75th percentile (Q3)
plt.title("Violin Plot with Quartiles")
plt.tight_layout()
plt.show()

# 4D. Split violin for a two-category hue variable.
plt.figure(figsize=(8, 5))
sns.violinplot(
    data=df, x="customer_type", y="monthly_spend",
    hue="churn", # It separates the data using colours, In your dataset - churn=0: Customers who stayed, - churn=1: Customers who left.
    # So each customer type gets separate violins for the two churn groups.
    split=True, # With split=True: The two hue groups share one violin shape, with one group on each side, when it was false Separate violins are drawn side by side.
    inner="quart", 
    cut=0 # Stops at the minimum and maximum observed values
    # if cut=1 => Extends approximately one bandwidth beyond the data.
    # cut=2 => Extends approximately two bandwidths.
)
plt.title("Split Violin by Churn")
plt.tight_layout()
plt.show()


# 4E. Change smoothing.

plt.figure(figsize=(8, 5))  # Set the figure size to 8 inches wide and 5 inches tall
sns.violinplot(
    data=df,  # Use the customer DataFrame
    x="customer_type",  # Display customer types (Standard and Premium) on the X-axis
    y="income",  # Display income values on the Y-axis
    bw_adjust=1.5,  # Increase KDE bandwidth by 1.5 times to create a smoother violin shape
    cut=0,  # Stop the violin at the minimum and maximum observed income values
    inner="box"  # Show a miniature box plot inside each violin
)
plt.title("Income Violin with Adjusted Smoothing")  # Set the chart title
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()  # Display the violin plot

# 4F. Width normalization in newer Seaborn versions.
# "width": equal maximum width; "count": width reflects sample count;
# "area": equal total area. If unsupported, upgrade Seaborn or omit it.

plt.figure(figsize=(8, 5))  # Set the figure size to 8 inches wide and 5 inches tall
sns.violinplot(
    data=df,  # Use the customer DataFrame
    x="customer_type",  # Display customer types (Standard and Premium) on the X-axis
    y="monthly_spend",  # Display monthly spending values on the Y-axis
    density_norm="width",  # Give each violin the same maximum width, regardless of group size
    inner="quart"  # Display Q1, median, and Q3 as three lines inside each violin
)
plt.title("Violin Plot with density_norm='width'")  # Set the chart title
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()  # Display the violin plot


# ============================================================
# 5. ECDF PLOT: sns.ecdfplot()
# ============================================================
# ECDF = Empirical Cumulative Distribution Function.
# It shows the proportion of observations <= each x value.
# At y=0.75, about 75% of observations are at or below that x value.

# Unlike a histogram, ECDF does not require choosing bins.
# Unlike KDE, ECDF does not smooth the observations.

# Useful parameters:
# hue            : separate curves by category
# stat           : "proportion" (default) or "count"
# complementary : show proportion above a value
# weights        : observation weights, if appropriate

# ECDF is useful for percentiles, threshold questions, and comparing
# cumulative distributions.

# 5A. Basic ECDF. It shows the proportion (or percentage) of observations that are less than or equal to a particular value.
'''
In simple words, it helps answer:
- What percentage of customers spend ₹1,500 or less?
- How many customers have an income below ₹50,000?
- What percentage of customers are within a particular age range?
'''
plt.figure(figsize=(8, 5))
sns.ecdfplot(data=df, x="monthly_spend")
# - X-axis: Monthly spending values.
# Y-axis: Cumulative proportion of customers, ranging from 0 to 1.
# ex : Suppose the graph reaches 0.50 at ₹1,500 => That means 50% of customers spend ₹1,500 or less.
# Y-axis value |	Meaning 
# 0.25	       | 25% of customers spend this amount or less
# 0.50	       | 50% of customers spend this amount or less
plt.title("Monthly Spend ECDF")
plt.ylabel("Cumulative Proportion")
plt.tight_layout()
plt.show()

# 5B. Compare groups.
plt.figure(figsize=(8, 5))
sns.ecdfplot(data=df, x="monthly_spend", hue="customer_type")
plt.title("Monthly Spend ECDF by Customer Type")
plt.tight_layout()
plt.show()

# 5C. Complementary ECDF: proportion greater than a value.
plt.figure(figsize=(8, 5))
sns.ecdfplot(data=df, x="monthly_spend", complementary=True)
# if true : Shows the proportion of customers spending above a particular amount.
# complementary=False (default): Shows the proportion of customers spending at or below a particular amount.
'''
ECDF type           	  |            Interpretation
Normal ECDF = 0.70	      |      70% spend ₹1,500 or less
Complementary ECDF = 0.30 |	    30% spend more than ₹1,500
'''
plt.title("Complementary ECDF")
plt.ylabel("Proportion Above Value")
plt.tight_layout()
plt.show()

# 5D. Cumulative count instead of proportion. - stat="count" gives you actual cumulative customer counts instead of percentages or proportions.
plt.figure(figsize=(8, 5))
sns.ecdfplot(data=df, x="age", stat="count") 
plt.title("Age ECDF (Cumulative Count)")
plt.ylabel("Cumulative Count")
plt.tight_layout()
plt.show()

#  Parameter	 Y-axis
# "proportion"	 0 to 1
# "percent"	     0 to 100
# "count"	    Number of observations


# ============================================================
# 6. SIDE-BY-SIDE COMPARISON
# ============================================================
# Same numerical variable, five different perspectives.

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
sns.histplot(data=df, x="monthly_spend", bins=25, ax=axes[0, 0])
axes[0, 0].set_title("Histogram")
sns.kdeplot(data=df, x="monthly_spend", fill=True, ax=axes[0, 1])
axes[0, 1].set_title("KDE")
sns.boxplot(data=df, x="monthly_spend", ax=axes[0, 2])
axes[0, 2].set_title("Boxplot")
sns.violinplot(
    data=df, x="customer_type", y="monthly_spend",
    inner="quart", ax=axes[1, 0]
)
axes[1, 0].set_title("Violin Plot")
sns.ecdfplot(data=df, x="monthly_spend", ax=axes[1, 1])
axes[1, 1].set_title("ECDF")
axes[1, 2].axis("off")
axes[1, 2].text(
    0.05, 0.85,
    "QUICK GUIDE\\n\\n"
    "Histogram: bin frequencies\\n\\n"
    "KDE: smooth density estimate\\n\\n"
    "Boxplot: quartiles and fliers\\n\\n"
    "Violin: distribution shape\\n\\n"
    "ECDF: cumulative proportion",
    va="top", fontsize=11
)

plt.suptitle("Five Views of a Numerical Distribution")
plt.tight_layout()
plt.show()


# ============================================================
# 7. DISTRIBUTIONS IN CLASSIFICATION / ML
# ============================================================
# Compare numerical features across a categorical target.
# This can reveal overlap or differences between classes.
# Visual separation alone does NOT prove predictive value.
# Confirm feature usefulness with appropriate validation.

# 7A. Histogram by churn class.
plt.figure(figsize=(9, 5))  # Set the figure size to 9 inches wide and 5 inches tall
sns.histplot(
    data=df,  # Use the customer DataFrame
    x="monthly_spend",  # Display monthly spending values on the X-axis
    hue="churn",  # Separate customers into churn groups (0 = stayed, 1 = left)
    bins=25,  # Divide the monthly spending range into 25 intervals
    stat="density",  # Display probability density instead of the actual count
    common_norm=False,  # Normalize each churn group independently so each has an area of 1
    element="step",  # Display histogram bars as step-shaped outlines
    fill=False  # Remove the fill colour and display only the outlines
)
plt.title("Monthly Spend Distribution by Churn")  # Set the chart title
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()  # Display the histogram

# 7B. KDE by churn class.
plt.figure(figsize=(9, 5))
sns.kdeplot(
    data=df, x="income", hue="churn", common_norm=False
)
plt.title("Income KDE by Churn Class")
plt.tight_layout()
plt.show()

# 7C. Boxplot by target.
plt.figure(figsize=(8, 5))  # Set the figure size to 8 inches wide and 5 inches tall
sns.boxplot(data=df, x="churn", y="monthly_spend")  # Create box plots comparing monthly spending for each churn group
# You will get two box plots:
# - Churn = 0: Monthly spending distribution of customers who stayed.
# - Churn = 1: Monthly spending distribution of customers who left.
plt.title("Monthly Spend by Churn Class")  # Set the chart title
plt.xlabel("Churn (0 = stayed, 1 = churned)")  # Set the X-axis label explaining the churn categories
plt.tight_layout()  # Adjust spacing to prevent overlapping elements
plt.show()  # Display the box plot

# 7D. Countplot to inspect class balance.
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="churn")
plt.title("Target Class Distribution")
plt.xlabel("Churn (0 = stayed, 1 = churned)")
plt.ylabel("Number of rows")
plt.tight_layout()
plt.show()

# ============================================================
# 8. NUMERICAL STATISTICS TO SUPPORT VISUAL ANALYSIS
# ============================================================
# Mean: arithmetic average; sensitive to extreme values.
# Median: middle value; more resistant to extreme values.
# Standard deviation: spread around the mean.
# Quartiles: 25th, 50th, and 75th percentiles.
# Skewness: asymmetry of a distribution.
# Positive skew: longer right tail; negative skew: longer left tail.
#
# These summaries complement plots; none alone determines whether
# a feature is useful for a model.

numeric_columns = ["age", "income", "monthly_spend"]

print("\\nDESCRIPTIVE STATISTICS")
print(df[numeric_columns].describe())

print("\\nMEAN")
print(df[numeric_columns].mean())

print("\\nMEDIAN")
print(df[numeric_columns].median())

print("\\nSTANDARD DEVIATION")
print(df[numeric_columns].std())

print("\\nSKEWNESS")
print(df[numeric_columns].skew())


# ============================================================
# 9. MISSING VALUES AND PLOTTING
# ============================================================
# Seaborn generally omits missing values for the variable plotted.
# Therefore, a plot may represent fewer rows than the DataFrame.
# Inspect missingness before interpreting the graph.
# Do not fill missing values just to make a plot; missing-data
# handling is a separate topic.

print("\\nMISSING VALUES")
print(df[numeric_columns].isna().sum())

print("\\nNON-MISSING COUNTS")
print(df[numeric_columns].count())

# Explicitly select non-missing income observations for this plot.
income_available = df.dropna(subset=["income"])

plt.figure(figsize=(8, 5))
sns.histplot(data=income_available, x="income", bins=30)
plt.title("Income Distribution: Non-Missing Observations")
plt.tight_layout()
plt.show()


# ============================================================
# 10. IMPORTANT ML INTERPRETATION NOTES
# ============================================================
# 1. Histogram: frequency/counts inside intervals (bins).
# 2. KDE: estimated smooth density, not raw counts.
# 3. Boxplot: median, Q1, Q3, IQR, whiskers, potential fliers.
# 4. Violin: estimated distribution shape; width indicates density.
# 5. ECDF: cumulative proportion at or below each value.
#
# OUTLIERS:
# A boxplot flier may be a valid rare event, a unit mismatch,
# a data-entry error, or an unusual but meaningful observation.
# Investigate its source and domain meaning before treating it.
#
# NORMALITY:
# Not every ML algorithm requires normally distributed features.
# Do not transform every skewed feature automatically.
#
# TRANSFORMATIONS:
# A log transform can sometimes help with strongly right-skewed
# positive data, but check zeros/negative values and evaluate the
# effect in the context of the model.
#
# SCALING:
# Scaling changes feature magnitudes; it does not automatically
# make a distribution normal.
#
# SAMPLE SIZE:
# KDE and violin estimates may be unreliable with very few rows.
#
# CLASS COMPARISON:
# Differences between class distributions are clues for exploration,
# not proof of causation or predictive performance.


# ============================================================
# 11. PRACTICE CHECKLIST
# ============================================================
# [ ] 1. Plot age with 10, 20, and 30 histogram bins.
# [ ] 2. Change binwidth and explain how the display changes.
# [ ] 3. Add kde=True to a histogram.
# [ ] 4. Compare categories with hue in histplot().
# [ ] 5. Compare multiple="stack" and multiple="layer".
# [ ] 6. Create a filled KDE plot.
# [ ] 7. Compare bw_adjust values 0.5, 1, and 2.
# [ ] 8. Create separate KDE curves using hue.
# [ ] 9. Identify potential fliers in an income boxplot.
# [ ] 10. Compare spend across customer types using boxplot.
# [ ] 11. Hide fliers and explain why data was not deleted.
# [ ] 12. Create a violin plot with inner="box".
# [ ] 13. Create a split violin plot with a two-level hue.
# [ ] 14. Create an ECDF and interpret y=0.75.
# [ ] 15. Compare all five plots for the same feature.
# [ ] 16. Compare a numerical feature across target classes.
# [ ] 17. Inspect missing values before interpreting plots.
# [ ] 18. Explain why a potential outlier is not automatically an error.
# [ ] 19. Explain why not every ML feature must be normal.
# [ ] 20. Choose the appropriate distribution plot for a given question.


# ============================================================
# 12. PHASE 2 COMPLETION CHECK
# ============================================================
# You are ready for Phase 3 when you can:
# - choose the correct distribution plot for a question
# - explain histogram bins and KDE smoothing
# - read median, quartiles, IQR, and whiskers
# - interpret violin width and ECDF values
# - compare distributions across categories and target classes
# - recognize skewness and investigate unusual values
# - check missing values before drawing conclusions
#
# RECOMMENDED TIME: 2-3 days
# Day 1: histplot() and kdeplot()
# Day 2: boxplot() and violinplot()
# Day 3: ecdfplot(), comparisons, and ML interpretation
#
# NEXT PHASE 3: CATEGORICAL PLOTS
# stripplot(), swarmplot(), boxenplot(), pointplot(), catplot()
# ============================================================
# END OF PHASE 2
# ============================================================
