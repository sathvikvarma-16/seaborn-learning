# ============================================================
# SEABORN PHASE 4: RELATIONSHIPS AND CORRELATION
# ============================================================

# LEVEL: Beginner to Advanced (ML Engineering Preparation)

# TOPICS COVERED:

# 1. Understanding relationships between variables
# 2. pairplot()
# 3. jointplot()
# 4. regplot()
# 5. lmplot()
# 6. heatmap()
# 7. Pearson correlation
# 8. Spearman correlation
# 9. Kendall correlation (basic awareness)
# 10. Correlation matrices
# 11. Correlation interpretation
# 12. Positive, negative and zero correlation
# 13. Linear and nonlinear relationships
# 14. Multicollinearity
# 15. Feature-target relationships
# 16. Regression visualization
# 17. Hue and categorical grouping
# 18. Pairwise feature analysis
# 19. Correlation-based feature selection
# 20. Missing values and correlation
# 21. EDA dashboard
# 22. ML Engineering applications
# 23. Common mistakes and best practices

# INSTALLATION:
# pip install numpy pandas matplotlib seaborn
# ============================================================


# ============================================================
# 0. IMPORT LIBRARIES AND SETUP
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(
    style="whitegrid",
    context="notebook"
)
rng = np.random.default_rng(42)

# ============================================================
# 1. UNDERSTANDING RELATIONSHIPS BETWEEN VARIABLES
# ============================================================

# A relationship describes how two or more variables
# change in relation to one another.

# Examples:

# Study Hours and Marks
# Advertising and Sales
# Experience and Salary
# Temperature and Electricity Consumption

# In ML, relationships help us understand:

# 1. Whether a feature is associated with the target.
# 2. Whether two features contain similar information.
# 3. Whether relationships are linear or nonlinear.
# 4. Whether unusual observations exist.
# 5. Whether transformations may be useful.
# 6. Whether multicollinearity may be a concern.

# IMPORTANT:
# Correlation measures association, Correlation does NOT prove causation.
# A strong correlation does not automatically mean that one variable causes another.


# ============================================================
# 2. CREATE A SAMPLE DATASET
# ============================================================

# We create a student performance dataset.

# Numerical features, study_hours, attendance, assignment_score, previous_marks, marks
# Categorical features: department,mgender, result
# Target: marks
# The data is generated for learning purposes only.

n = 300
study_hours = rng.uniform(1, 10, n)
attendance = np.clip(
    55 + study_hours * 4 + rng.normal(0, 10, n),
    30,
    100
)
assignment_score = np.clip(
    30 + study_hours * 6 + rng.normal(0, 12, n),
    0,
    100
)
previous_marks = np.clip(
    rng.normal(65, 15, n),
    0,
    100
)
# Generate marks using several features and random noise.
marks = np.clip(
    10
    + study_hours * 4
    + attendance * 0.15
    + assignment_score * 0.25
    + previous_marks * 0.2
    + rng.normal(0, 8, n),
    0,
    100
)
df = pd.DataFrame({
    "study_hours": study_hours,
    "attendance": attendance,
    "assignment_score": assignment_score,
    "previous_marks": previous_marks,
    "marks": marks,
    "department": rng.choice(
        ["CSE", "ECE", "EEE", "Mechanical"],
        n
    ),
    "gender": rng.choice(
        ["Male", "Female"],
        n
    )
})
df["result"] = np.where(
    df["marks"] >= 50,
    "Pass",
    "Fail"
)
print("\nDATASET")
print(df.head())
print("\nDATASET INFORMATION")
print(df.info())
print("\nSTATISTICAL SUMMARY")
print(df.describe())


# ============================================================
# 3. UNDERSTANDING CORRELATION
# ============================================================

# Correlation measures the strength and direction of association between two variables.
# The commonly used Pearson correlation coefficient ranges from -1 to +1.
# +1: Perfect positive linear correlation.
# 0: No linear correlation.
# -1: Perfect negative linear correlation.

# POSITIVE CORRELATION:  When one variable increases, the other tends to increase.
# Example:
# Study hours and marks.

# NEGATIVE CORRELATION:
# When one variable increases, the other tends to decrease.
# Example: Product price and quantity demanded in some market situations.
# ZERO CORRELATION: No linear association is detected.

# IMPORTANT:
# Zero Pearson correlation does not necessarily
# mean that no relationship exists.
# A nonlinear relationship may still be present.


# ============================================================
# 4. PAIRPLOT()
# ============================================================

# pairplot() creates pairwise visualizations
# for multiple numerical variables.
# It provides:
# Diagonal: Distribution of each individual variable.
# Off-diagonal:  Relationship between pairs of variables.
# It is especially useful during initial EDA.


# ------------------------------------------------------------
# 4.1 BASIC PAIRPLOT
# ------------------------------------------------------------

numerical_columns = [
    "study_hours",
    "attendance",
    "assignment_score",
    "previous_marks",
    "marks"
]

# graph is a Seaborn Pairplot containing 5 numerical features and 25 individual plots (5 × 5 grid).
sns.pairplot( # Seaborn takes your five numerical columns and automatically compares:
    # Each variable with itself.
    # Each variable with every other variable.
    data=df,
    vars=numerical_columns
)
# X-axis and Y-axis contain these five variables: 
# 1. study_hours – Hours students spend studying.
# 2. attendance – Students' attendance percentage.
# 3. assignment_score – Assignment marks.
# 4. previous_marks – Marks obtained previously.
# 5. marks – Students' current marks.
# same variable - histogram | different variables - scatter plot 
plt.show()


# ------------------------------------------------------------
# 4.2 PAIRPLOT WITH HUE
# ------------------------------------------------------------

# hue separates observations by a categorical variable.
# Here, observations are colored according to their result.

sns.pairplot(
    data=df,
    vars=numerical_columns,
    hue="result",
    palette="Set1"
)
plt.show()


# ------------------------------------------------------------
# 4.3 CONTROL DIAGONAL DISTRIBUTIONS
# ------------------------------------------------------------

# diag_kind:
#
# "hist" -> Histogram.
# "kde"  -> Kernel density estimate.

sns.pairplot(
    data=df,
    vars=numerical_columns,
    hue="result", 
    diag_kind="hist" # Controls the type of diagonal plots.    
)

plt.show()


# ------------------------------------------------------------
# 4.4 USE KDE ON THE DIAGONAL
# ------------------------------------------------------------

sns.pairplot(
    data=df,
    vars=numerical_columns,
    hue="result",
    diag_kind="kde" # instead of histogram displays kernel density estimation
)
plt.show()


# ------------------------------------------------------------
# 4.5 CHANGE OFF-DIAGONAL PLOT
# ------------------------------------------------------------

# kind:
#
# "scatter" -> Scatter plots.
# "kde"     -> Density contours.
# "hist"    -> Bivariate histograms.
# "reg"     -> Regression plots.

# Note:
# Availability and behavior of some options can depend on the Seaborn version.

sns.pairplot(
    data=df,
    vars=numerical_columns,
    hue="result", # color seperation
    kind="reg", # Displays scatterplots with regression lines
    # The regression line helps you identify the general trend or relationship between two numerical variables.
    diag_kind="hist"
)

plt.show()


# ------------------------------------------------------------
# 4.6 SELECT ONLY IMPORTANT FEATURES
# ------------------------------------------------------------

# A pairplot with too many features can become crowded.
# Select a smaller number of relevant variables.

sns.pairplot(
    data=df,
    vars=[
        "study_hours",
        "attendance",
        "marks"
    ],
    hue="result"
)
plt.show()

# ------------------------------------------------------------
# 4.7 CONTROL PLOT SIZE AND TRANSPARENCY
# ------------------------------------------------------------

sns.pairplot(
    data=df,
    vars=numerical_columns,
    hue="result",
    height=2.5, # How  tall ? Controls the height of each individual plot in inches.
    aspect=1,   # How wide relative to height ? Controls the width-to-height ratio of each individual plot -> Each subplot is twice as wide as its height.
    plot_kws={  # to customize the appearance of the plots outside the diagonal, such as scatterplots.
        "alpha": 0.6, # slightly transparent
        "s": 35 # controls size of each scatterpoint
    }
)
plt.show()


# ------------------------------------------------------------
# 4.8 IMPORTANT PAIRPLOT PARAMETERS
# ------------------------------------------------------------

# data:
# Input DataFrame.

# vars:
# Select numerical variables to visualize.

# x_vars and y_vars:
# Select different variables for rows and columns.

# hue:
# Group observations by a categorical variable.

# kind:
# Controls off-diagonal visualization.

# diag_kind:
# Controls diagonal visualization.

# corner=True:
# Displays only one triangular half of the pairplot.

# height:
# Height of each subplot.

# aspect:
# Width-to-height ratio of each subplot.

# palette: "Set1": Bright colors | "Set2": Soft colors | "Set3": Multiple pastel colors | "deep": Seaborn's default color palette | "viridis": Color gradient.
# Controls group colors.

# plot_kws:
# Parameters passed to off-diagonal plots.

# diag_kws: customizes the appearance of plots on the diagonal -> fill=True: Fills the area under the KDE curve.
# It is useful when diag_kind="hist" or diag_kind="kde".
# Parameters passed to diagonal plots.


# ------------------------------------------------------------
# 4.9 PAIRPLOT WITH CORNER=True
# ------------------------------------------------------------

# Pairwise relationships are symmetric.
# The upper and lower triangles contain repeated information.
# corner=True removes the upper triangle.

sns.pairplot(
    data=df,
    vars=numerical_columns,
    hue="result",
    corner=True
)
plt.show()


# ------------------------------------------------------------
# 4.10 DIFFERENT X AND Y VARIABLES
# ------------------------------------------------------------

# x_vars and y_vars allow us to create a smaller rectangular grid of pairwise plots.

sns.pairplot(
    data=df,
    x_vars=["study_hours", "attendance"], # columns displayed on x-axis
    y_vars=["marks", "assignment_score"], # columns displayed on y-axis
    hue="result" # Colors the points based on student results (Pass or Fail).
) 
# x_vars determines the columns (left to right) || y_vars determines the rows (top to bottom)
plt.show()


# ============================================================
# 5. JOINTPLOT()
# ============================================================

# jointplot() visualizes the relationship between two numerical variables and their individual distributions.

# Main structure:

# Center: Relationship between X and Y.
# Top:Distribution of X.
# Right:Distribution of Y.

# Useful for understanding:
# - Relationship strength and direction.
# - Distribution shape.
# - Clusters.
# - Outliers.
# - Density concentration.


# ------------------------------------------------------------
# 5.1 BASIC JOINTPLOT
# ------------------------------------------------------------

sns.jointplot(
    data=df,
    x="study_hours",
    y="marks" # -> equivalent to kind="scatter" -> default kind is scatterplot()
)
# Central plot: Shows the relationship between two variables
# Top plot: Shows the distribution of the X-axis variable.
# Right plot: Shows the distribution of the Y-axis variable.
plt.show()


# ------------------------------------------------------------
# 5.2 JOINTPLOT WITH KIND="SCATTER"
# ------------------------------------------------------------

sns.jointplot(
    data=df,
    x="study_hours",
    y="marks",
    kind="scatter" # Type of central plot
)
plt.show()


# ------------------------------------------------------------
# 5.3 JOINTPLOT WITH KIND="REG"
# ------------------------------------------------------------

# Adds a regression line to the central relationship plot.

sns.jointplot(
    data=df,
    x="study_hours",
    y="marks",
    kind="reg" # creates a scatterplot with a regression line.
)
# The line represents the overall linear trend.
# The line helps identify whether marks tend to increase or decrease as study hours increase.
plt.show()


# ------------------------------------------------------------
# 5.4 JOINTPLOT WITH KIND="RESID"
# ------------------------------------------------------------

# Displays regression residuals in the central plot.
# Residual: Actual value - Predicted value.
# Residual plots help inspect patterns that a fitted linear relationship may not explain.

sns.jointplot(
    data=df,
    x="study_hours",
    y="marks",
    kind="resid"
)
plt.show()


# ------------------------------------------------------------
# 5.5 JOINTPLOT WITH KIND="KDE"
# ------------------------------------------------------------

# KDE represents estimated data density.
# Dense areas indicate where observations are concentrated.

sns.jointplot(
    data=df,
    x="study_hours",
    y="marks",
    kind="kde",
    fill=True
)

plt.show()


# ------------------------------------------------------------
# 5.6 JOINTPLOT WITH KIND="HIST"
# ------------------------------------------------------------

sns.jointplot(
    data=df,
    x="study_hours",
    y="marks",
    kind="hist"
)
plt.show()


# ------------------------------------------------------------
# 5.7 JOINTPLOT WITH KIND="HEX"
# ------------------------------------------------------------

# Hexbin plots group observations into hexagonal bins.
# Useful when a scatter plot becomes overcrowded because of a large number of observations.

sns.jointplot(
    data=df,
    x="study_hours",
    y="marks",
    kind="hex"
)
plt.show()


# ------------------------------------------------------------
# 5.8 JOINTPLOT WITH HUE
# ------------------------------------------------------------

# hue separates observations by category.

sns.jointplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="result",
    kind="scatter"
)
plt.show()


# ------------------------------------------------------------
# 5.9 CONTROL JOINTPLOT SIZE
# ------------------------------------------------------------

sns.jointplot(
    data=df,
    x="study_hours",  # X-axis: Study hours
    y="marks",        # Y-axis: Marks
    kind="reg",       # Scatterplot with regression line - Centre: Study hours vs marks with a regression line. 
    # Top: Distribution of study hours.
    # Top: Distribution of study hours.
    height=7,         # Overall figure height (in inches) / smaller, larger, medium plot is depends on size of height
    ratio=5,          # Central plot size relative to marginal plots - lesser the ratio central plot is relatively lesser
    space=0.2         # space=0 -> no gap, space=0.2 -> small gap, space=0.5 -> larger  gap
)
plt.show()

# ------------------------------------------------------------
# 5.10 IMPORTANT JOINTPLOT PARAMETERS
# ------------------------------------------------------------

# data: Input DataFrame.
# x and y: Numerical variables.
# kind: scatter, reg, resid, kde, hist or hex.
# hue:Categorical grouping variable.
# height: Overall figure size.
# ratio: Relative size of the central plot compared with marginal plots.
# space: Space between central and marginal plots.
# marginal_ticks: Controls ticks on marginal axes.
# joint_kws: Parameters for the central plot.
# marginal_kws: Parameters for marginal distributions.


# ============================================================
# 6. REGPLOT()
# ============================================================

# regplot() displays a scatter plot and fits a regression model to visualize the relationship.
# It is commonly used for understanding linear trends.
# It is a visualization tool.
# It is not a replacement for a complete ML model.


# ------------------------------------------------------------
# 6.1 BASIC REGPLOT
# ------------------------------------------------------------

sns.regplot(
    data=df,
    x="study_hours",
    y="marks"
)
plt.title("Study Hours vs Marks with Regression Line")
plt.show()


# ------------------------------------------------------------
# 6.2 UNDERSTANDING THE REGRESSION LINE 
# sns.regplot() creates a scatterplot with a regression line to show the relationship between two numerical variables.
# ------------------------------------------------------------

# The regression line represents a fitted linear trend.
# Positive slope: Y tends to increase as X increases.
# Negative slope: Y tends to decrease as X increases.
# Flat slope: Little linear trend is detected.
# The shaded region generally represents a confidence interval around the estimated mean.


# ------------------------------------------------------------
# 6.3 DISABLE CONFIDENCE INTERVAL
# ------------------------------------------------------------
sns.regplot(
    data=df,
    x="study_hours",  # X-axis: Study hours
    y="marks",        # Y-axis: Marks
    ci=None           # Hides the confidence interval shaded region
)
plt.title("Regression Plot Without Confidence Interval")
plt.show()

# Normally, Seaborn displays a shaded region around the regression line representing the confidence interval.
# - ci=95 (default): Displays a 95% confidence interval.
# - ci=None: Removes the confidence interval shading.


# ------------------------------------------------------------
# 6.4 CHANGE CONFIDENCE INTERVAL
# ------------------------------------------------------------

sns.regplot(
    data=df,
    x="study_hours",
    y="marks",
    ci=95
)
plt.title("Regression Plot with 95% Confidence Interval")
plt.show()


# NOTE:
# In regplot(), ci is used for the regression confidence interval in commonly used Seaborn versions.
# This differs from newer errorbar syntax used by functions such as lineplot and barplot.


# ------------------------------------------------------------
# 6.5 CHANGE SCATTER POINT APPEARANCE
# ------------------------------------------------------------

sns.regplot(
    data=df,                   # DataFrame containing student data
    x="study_hours",           # X-axis: Students' study hours
    y="marks",                 # Y-axis: Students' marks
    scatter_kws={
        "color": "blue",       # Sets scatter points color to blue
        "alpha": 0.5,          # Makes scatter points slightly transparent
        "s": 40                # Sets the size of scatter points
    },
    line_kws={
        "color": "red",        # Sets regression line color to red
        "linewidth": 2         # Sets regression line thickness
    }
)
plt.title("Customized Regression Plot")
plt.show()


# ------------------------------------------------------------
# 6.6 USE ORDER FOR POLYNOMIAL REGRESSION
# ------------------------------------------------------------

# order=1 : Linear relationship.
# order=2: Quadratic relationship.
# order=3:Cubic relationship.
# Higher polynomial orders can overfit.
# Use them only when justified by the data.

sns.regplot(
    data=df,
    x="study_hours",
    y="marks",
    order=2
)
plt.title("Polynomial Regression Visualization")
plt.show()

# ------------------------------------------------------------
# 6.7 LOWESS REGRESSION
# ------------------------------------------------------------

# lowess=True fits a locally weighted smoothing curve.
# Useful for exploring nonlinear trends.
# It is primarily a visualization technique.

sns.regplot(
    data=df,
    x="study_hours",
    y="marks",
    lowess=True, # creates a smooth curved regression line instead of a straight linear regression line.
    # lowess=False → Normal straight regression line (default).
    scatter_kws={"alpha": 0.5}
)
plt.title("LOWESS Relationship")
plt.show()


# ------------------------------------------------------------
# 6.8 LOGISTIC REGRESSION VISUALIZATION
# ------------------------------------------------------------

# Logistic regression can be visualized using: logistic=True - fits a logistic regression curve to estimate probabilities.
# Here we create a binary target for demonstration.

df["high_performance"] = (
    df["marks"] >= 70
).astype(int)
sns.regplot(
    data=df,
    x="study_hours",
    y="high_performance", 
    logistic=True, # Fits logistic regression instead of linear regression
    y_jitter=0.03,  # Slightly spreads binary points vertically to reduce overlap
    ci=96 # Displays the 95% confidence interval by default, ci=None	Hides the confidence interval
)

# IMPORTANT:
# logistic=True requires statsmodels.
# Install it if needed: pip install statsmodels
# This is a simple visualization.
# It is not a complete classification workflow.

# ------------------------------------------------------------
# 6.9 X_ESTIMATOR - applies a function to the Y-values for each unique X-value and plots the resulting estimated value.
# ------------------------------------------------------------

# x_estimator aggregates repeated X values before plotting them.
# This can be useful when many observations share the same X value.

repeated_df = pd.DataFrame({
    "hours": np.repeat([1, 2, 3, 4, 5], 20),
    "marks": np.concatenate([
        rng.normal(40, 5, 20),
        rng.normal(50, 5, 20),
        rng.normal(60, 5, 20),
        rng.normal(70, 5, 20),
        rng.normal(80, 5, 20)
    ])
})
sns.regplot(
    data=repeated_df,
    x="hours",
    y="marks",
    x_estimator=np.mean
)
plt.title("Regression with X Aggregation")
plt.show()


# ------------------------------------------------------------
# 6.10 IMPORTANT REGPLOT PARAMETERS
# ------------------------------------------------------------

# data: Input DataFrame.
# x and y: Variables to compare.
# hue:Not supported by regplot itself.
# ci: Confidence interval.
# order:Polynomial regression order.
# logistic:Logistic regression visualization.
# lowess: Locally weighted smoothing.
# robust:Robust regression.
# logx: Fits a regression using a logarithmic X transformation.
# x_estimator: Aggregates observations with repeated X values.
# scatter_kws: Customizes scatter points.
# line_kws: Customizes the regression line.
# truncate: Controls whether the regression line is limited to the observed X range.


# regplot() → One regression plot.
# lmplot() → Regression plots with categorical grouping and multiple subplots.

# ============================================================
# 7. LMPlot()
# ============================================================

# lmplot() is a figure-level regression visualization.
# It is similar to regplot(), but it supports:
# hue: Separate regression lines by category.
# col: Separate plots into columns.
# row: Separate plots into rows.
# It is useful for comparing relationships across multiple groups.


# ------------------------------------------------------------
# 7.1 BASIC LMPLOT
# ------------------------------------------------------------

sns.lmplot(
    data=df,
    x="study_hours",
    y="marks"
)
plt.show()


# ------------------------------------------------------------
# 7.2 LMPLOT WITH HUE
# ------------------------------------------------------------

sns.lmplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="gender"
)

plt.show()


# ------------------------------------------------------------
# 7.3 LMPLOT WITH COL
# ------------------------------------------------------------

# sns.lmplot() creates separate regression plots for different categories using col, row, or hue.
sns.lmplot( # It creates two separate regression plots side by side:
    data=df,
    x="study_hours",
    y="marks",
    col="gender" # horizontal
)
plt.show()


# ------------------------------------------------------------
# 7.4 LMPLOT WITH ROW
# ------------------------------------------------------------

sns.lmplot(
    data=df,
    x="study_hours",
    y="marks",
    row="gender" # vertical
)
plt.show()


# ------------------------------------------------------------
# 7.5 LMPLOT WITH ROW AND COL
# ------------------------------------------------------------

sns.lmplot(
    data=df,
    x="study_hours",   # X-axis: Study hours
    y="marks",         # Y-axis: Marks
    row="gender",      # Separate rows for Male and Female
    col="department",  # Separate columns for each department
    height=3,          # Height of each subplot (in inches)
    aspect=1           # Width-to-height ratio of each subplot
)
plt.show()

# How the grid is arranged
#	    CSE	     ECE	 EEE Mechanical
# Male	Plot 1	Plot 2	Plot 3	Plot 4
# Female	Plot 5	Plot 6	Plot 7	Plot 8


# ------------------------------------------------------------
# 7.6 LMPLOT WITH LOWESS
# ------------------------------------------------------------

sns.lmplot(
    data=df,
    x="study_hours",
    y="marks",
    lowess=True
)
plt.show()


# ------------------------------------------------------------
# 7.7 LMPLOT WITH ORDER
# ------------------------------------------------------------
'''
Parameter	Meaning
order=1	   Linear regression (straight line)
order=2    Quadratic regression
order=3	   Cubic regression
order=4	   Fourth-degree polynomial regression
'''

sns.lmplot(
    data=df,
    x="study_hours",
    y="marks",
    order=2             # Fits a second-degree polynomial curve
)
plt.show()


# ------------------------------------------------------------
# 7.8 IMPORTANT LMPLOT PARAMETERS
# ------------------------------------------------------------

# data: Input DataFrame.

# x and y: Variables
# hue: Separate regression lines by category.
# col: Create separate columns. horizontal
# row: Create separate rows. vertical
# col_wrap: Wrap columns into multiple rows.
# height: Height of each facet.
# aspect: Width-to-height ratio.
# order: Polynomial regression order.
# logistic: Logistic regression visualization.
# lowess: Locally weighted smoothing.
# ci: Confidence interval.
# scatter_kws: Scatter point customization.
# line_kws: Regression line customization.


# ============================================================
# 8. REGPLOT VS LMPLOT
# ============================================================

# regplot():
# - Axes-level function.
# - Works with an existing Matplotlib Axes.
# - Does not provide hue, row and col faceting.

# lmplot():
# - Figure-level function.
# - Supports hue, row and col.
# - Creates its own figure.

# Use regplot when: You need one regression plot inside a dashboard.

# Use lmplot when: You need to compare regression relationships across multiple categories.


# ============================================================
# 9. PEARSON CORRELATION
# ============================================================

# Pearson correlation measures the strength and direction of a LINEAR relationship between two numerical variables.
# It ranges from -1 to +1.
# +1: Perfect positive linear relationship.
# -1: Perfect negative linear relationship.
# 0: No linear relationship detected.
# Pearson is sensitive to extreme outliers.
# It may not capture nonlinear relationships.


# ------------------------------------------------------------
# 9.1 CALCULATE PEARSON CORRELATION - ex : it helps determine whether students who study more hours tend to score higher marks.
# ------------------------------------------------------------

pearson_corr = df[numerical_columns].corr(
    method="pearson"
)
print("\nPEARSON CORRELATION")
print(pearson_corr)

# ------------------------------------------------------------
# 9.2 CORRELATION BETWEEN TWO VARIABLES
# ------------------------------------------------------------

study_marks_corr = df[
    "study_hours"
].corr(
    df["marks"],
    method="pearson"
)
print("\nSTUDY HOURS AND MARKS CORRELATION:")
print(study_marks_corr)


# ------------------------------------------------------------
# 9.3 INTERPRETING CORRELATION
# ------------------------------------------------------------

# A rough descriptive guide: 0.00 to 0.19: Very weak linear association.
# 0.20 to 0.39: Weak linear association 
# Moderate linear association : 0.40 to 0.59
# Strong linear association : 0.60 to 0.79
# Very strong linear association : 0.80 to 1.00

# The same applies to negative values using absolute magnitude.
# IMPORTANT:
# These ranges are informal guidelines.
# There are no universal thresholds.
# Interpret correlation according to the problem context.


# ============================================================
# 10. SPEARMAN CORRELATION - measures the strength and direction of the relationship between two variables based on their ranks.
# ============================================================
'''
Unlike Pearson, which measures linear relationships using actual values, 
Spearman measures whether variables tend to increase or decrease together in a consistent order.
'''

# Spearman correlation measures monotonic association.
# It works using ranks rather than original values.
# A monotonic relationship means: As X increases, Y generally increases. OR As X increases, Y generally decreases.
# The relationship does not have to be linear.

# Spearman is useful when:
# - Relationships are monotonic but nonlinear, Variables contain ordinal information, Data is not well suited to Pearson assumptions.
# Spearman is still sensitive to unusual observations
# in some situations, but rank-based methods are
# generally less affected by extreme magnitudes.


# ------------------------------------------------------------
# 10.1 CALCULATE SPEARMAN CORRELATION
# ------------------------------------------------------------

spearman_corr = df[numerical_columns].corr(
    method="spearman"
)
print("\nSPEARMAN CORRELATION")
print(spearman_corr)


# ------------------------------------------------------------
# 10.2 COMPARE PEARSON AND SPEARMAN
# ------------------------------------------------------------

# Compare Pearson and Spearman correlations of study_hours with all selected numerical columns

correlation_comparison = pd.DataFrame({
    "Pearson": df[numerical_columns].corr(
        method="pearson"       # Calculates Pearson correlation (linear relationship)
    ).loc["study_hours"],     # Selects the correlation values of study_hours with every column
    "Spearman": df[numerical_columns].corr(
        method="spearman"      # Calculates Spearman correlation (rank-based monotonic relationship)
    ).loc["study_hours"]      # Selects the correlation values of study_hours with every column
})

# Display the comparison DataFrame
print(correlation_comparison)
print("\nPEARSON VS SPEARMAN")
print(correlation_comparison)


# ============================================================
# 11. KENDALL CORRELATION
# ============================================================

# Kendall correlation is another rank-based correlation measure.
# It measures ordinal association using concordant and discordant pairs.

# It can be useful for: Ordinal variables, Small datasets, Rank-based association analysis.
# For most beginner EDA tasks, Pearson and Spearman are sufficient to understand.
# You should know Kendall exists, but you do not need to master its mathematical derivation now.

kendall_corr = df[numerical_columns].corr(
    method="kendall"
)
print("\nKENDALL CORRELATION")
print(kendall_corr)


# ============================================================
# 12. PEARSON VS SPEARMAN VS KENDALL
# ============================================================

# Pearson: Measures linear association Uses original numerical values
# Spearman : Measures monotonic association, Uses ranks.
# Kendall: Measures rank association using pairs.
# Choose based on the relationship and data.
# Do not blindly use Pearson for every dataset.


# ============================================================
# 13. HEATMAP()
# ============================================================

# heatmap() displays values using colors.
# It is commonly used for:
# - Correlation matrices.
# - Feature relationships.
# - Missing value patterns.
# - Statistical summaries.
# A correlation heatmap is especially useful
# for examining relationships among numerical features.


# ------------------------------------------------------------
# 13.1 BASIC HEATMAP
# ------------------------------------------------------------

sns.heatmap(
    pearson_corr # Correlation matrix containing Pearson correlation values
)
plt.title("Basic Correlation Heatmap")
plt.show()

# Each square represents the correlation between two numerical variables.
# Each square has a color representing the correlation strength and direction.
# Values closer to +1 indicate strong positive correlation.
# Values closer to -1 indicate strong negative correlation.
# Values closer to 0 indicate weak linear correlation.


# ------------------------------------------------------------
# 13.2 ANNOTATIONS
# ------------------------------------------------------------

# annot=True displays numerical values inside cells.
sns.heatmap(
    pearson_corr,
    annot=True
)
plt.title("Correlation Heatmap with Values")
plt.show()


# ------------------------------------------------------------
# 13.3 FORMAT ANNOTATIONS
# ------------------------------------------------------------

# fmt=".2f":
# Displays two decimal places.
sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f"
)

plt.title("Correlation Heatmap with Two Decimal Places")
plt.show()


# ------------------------------------------------------------
# 13.4 CHANGE COLOR MAP
# ------------------------------------------------------------

# cmap controls the color mapping.
sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()


# ------------------------------------------------------------
# 13.5 CENTER AT ZERO
# ------------------------------------------------------------
# center=0: Places zero at the center of the color scale.
# Useful for correlation matrices containing, both positive and negative values.

sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap Centered at Zero")
plt.show()


# ------------------------------------------------------------
# 13.6 CONTROL MINIMUM AND MAXIMUM
# ------------------------------------------------------------

# vmin and vmax control color scale limits.
# For correlation: -1 to +1 is the natural range.

sns.heatmap(
    pearson_corr,       # Correlation matrix containing Pearson correlation values
    annot=True,         # Displays correlation values inside each cell
    fmt=".2f",          # Formats values to 2 decimal places (e.g., 0.85)
    cmap="coolwarm",    # Uses cool colors for negative and warm colors for positive correlations
    center=0,           # Places the neutral color at correlation value 0
    vmin=-1,            # Sets the minimum color scale value to -1
    vmax=1              # Sets the maximum color scale value to +1
    # customized Pearson correlation heatmap with a fixed color scale from -1 to +1.
)
plt.title("Correlation Heatmap with Fixed Scale")
plt.show()


# ------------------------------------------------------------
# 13.7 CHANGE CELL BOUNDARIES
# ------------------------------------------------------------

sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5, # Controls the thickness of the lines separating the cells. (0/0.5/1/2) 
    linecolor="white" # Controls the color of the borders between cells.
)
plt.title("Heatmap with Cell Boundaries")
plt.show()


# ------------------------------------------------------------
# 13.8 CHANGE FIGURE SIZE
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))
sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True #     square=True  # Makes every heatmap cell square, perfectly square-shaped (equal width and height).
    # square=False: Cells can be rectangular (default).
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 13.9 HIDE COLOR BAR
# ------------------------------------------------------------

sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    cbar=False # hides the color bar (color scale) displayed beside the heatmap.
    # cbar controls whether the color scale legend is displayed.
    # cbar=True: Displays the color bar (default).
)
plt.title("Heatmap Without Color Bar")
plt.show()


# ------------------------------------------------------------
# 13.10 IMPORTANT HEATMAP PARAMETERS
# ------------------------------------------------------------

# data: 2D DataFrame or array.
# annot: Displays values inside cells.
# fmt: Formats annotation values.
# cmap: Controls color mapping.
# center: Centers color scale around a value.
# vmin and vmax: Set color scale limits.
# linewidths: Controls cell border thickness.
# linecolor: Controls cell border color.
# square: Makes cells square.
# cbar: Displays or hides the color bar.
# mask: Hides selected cells.
# annot_kws: Controls annotation text formatting.
# cbar_kws: Controls color bar formatting.


# ============================================================
# 14. UNDERSTANDING THE CORRELATION MATRIX
# ============================================================

# A correlation matrix contains pairwise correlations between selected numerical variables.
# The diagonal values are always 1 because every variable is perfectly correlated with itself.
# The matrix is symmetric : Correlation(A, B) = Correlation(B, A)
# Therefore, the upper and lower triangles contain repeated information.


# ------------------------------------------------------------
# 14.1 CORRELATION MATRIX
# ------------------------------------------------------------

corr_matrix = df[numerical_columns].corr(
    method="pearson"
)
print("\nCORRELATION MATRIX")
print(corr_matrix)


# ------------------------------------------------------------
# 14.2 UPPER TRIANGULAR MASK
# ------------------------------------------------------------

'''
A correlation matrix contains repeated information.
For example:
Original correlation matrix :

                Study hours	    Attendance	    Marks
Study hours	    1.00	        0.65	        0.80
Attendance    	0.65	        1.00	        0.55
Marks	        0.80	        0.55	        1.00

Notice:
Study hours vs Attendance = 0.65
Attendance vs Study hours = 0.65
Both are the same information!
So, we hide the upper triangle and display only the lower triangle.
'''

# Step 1: Create a matrix of True values
np.ones_like(corr_matrix, dtype=bool) # creates an array with the same shape as corr_matrix, dtype=bool means its values will be Boolean (True or False).
# Step 2 : create an upper triangular mask
mask = np.triu( # np.triu() selects the upper triangle, including the diagonal, It creates the mask that tells Seaborn which cells to hide.
    # In a Seaborn heatmap, True means hide that cell, and False means show it.
    np.ones_like( 
        corr_matrix,
        dtype=bool
    )
)
plt.figure(figsize=(9, 7))
sns.heatmap(
    corr_matrix, # 	Correlation matrix to visualize
    mask=mask, # Hides cells where mask is True
    annot=True, # Displays correlation values in cells
    fmt=".2f", # Displays values to 2 decimal places
    cmap="coolwarm", # Sets the color palette
    center=0, # Centers the color scale at zero
    vmin=-1, # Minimum color scale value
    vmax=1, # Maximum color scale value
    square=True, # Makes cells square
    linewidths=0.5 # Sets border thickness between cells
)
plt.title("Lower Triangle Correlation Heatmap")
plt.tight_layout()
plt.show()

# if you want to show the diagonal and hide only upper triangle use :
'''
mask = np.triu(
    np.ones_like(corr_matrix, dtype=bool),
    k=1 # k=1 starts masking one diagonal above the main diagonal.
)
'''
# We use np.triu() and mask to remove duplicate correlation values, making the heatmap cleaner and easier to read.


# ------------------------------------------------------------
# 14.3 SPEARMAN CORRELATION HEATMAP
# ------------------------------------------------------------

spearman_matrix = df[numerical_columns].corr(
    method="spearman"
)
plt.figure(figsize=(9, 7))
sns.heatmap(
    spearman_matrix,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Spearman Correlation Matrix")
plt.tight_layout()
plt.show()


# ============================================================
# 15. NONLINEAR RELATIONSHIPS
# ============================================================

# Pearson correlation mainly captures linear association.
# Consider: Y = X squared
# This relationship is nonlinear.
# Pearson correlation may be close to zero in a symmetric dataset even though X and Y are strongly related.
# Therefore:
# A low Pearson correlation does not prove that two variables are unrelated.


# ------------------------------------------------------------
# 15.1 CREATE A NONLINEAR DATASET
# ------------------------------------------------------------

x = np.linspace(-5, 5, 200)
y = x ** 2 + rng.normal(0, 1, 200)
nonlinear_df = pd.DataFrame({
    "x": x,
    "y": y
})


# ------------------------------------------------------------
# 15.2 VISUALIZE THE NONLINEAR RELATIONSHIP
# ------------------------------------------------------------

sns.scatterplot(
    data=nonlinear_df,
    x="x",
    y="y"
)
plt.title("Nonlinear Relationship")
plt.show()


# ------------------------------------------------------------
# 15.3 COMPARE CORRELATION
# ------------------------------------------------------------

print("\nNONLINEAR DATASET PEARSON CORRELATION")

print(
    nonlinear_df["x"].corr(
        nonlinear_df["y"],
        method="pearson"
    )
)
print("\nNONLINEAR DATASET SPEARMAN CORRELATION")
print(
    nonlinear_df["x"].corr(
        nonlinear_df["y"],
        method="spearman"
    )
)


# IMPORTANT:
# Always inspect relationships visually.
# Correlation coefficients alone may hide important
# nonlinear patterns.


# ============================================================
# 16. MULTICOLLINEARITY
# ============================================================

# Multicollinearity occurs when two or more input features are strongly linearly related.
# Example:
# Feature A = Study hours
# Feature B = Study hours multiplied by 2

# These features contain very similar information.
# Possible effects:
# - Unstable coefficient estimates in linear models.
# - Difficulty interpreting individual coefficients.
# - Redundant features.
# - Increased model complexity.

# IMPORTANT: Multicollinearity is particularly important for coefficient-based models such as linear regression.

# It does not automatically damage every ML model.
# Tree-based models, for example, can often work reasonably well with correlated features.

# Correlation is one way to detect possible multicollinearity, but it does not detect every form of dependence.


# ------------------------------------------------------------
# 16.1 CREATE CORRELATED FEATURES
# ------------------------------------------------------------

multi_df = pd.DataFrame()
multi_df["study_hours"] = rng.uniform(
    1,
    10,
    300
)
multi_df["study_hours_double"] = (
    multi_df["study_hours"] * 2
    + rng.normal(0, 0.2, 300)
)
multi_df["attendance"] = rng.uniform(
    40,
    100,
    300
)
multi_df["marks"] = (
    multi_df["study_hours"] * 5
    + multi_df["attendance"] * 0.3
    + rng.normal(0, 5, 300)
)

multi_corr = multi_df.corr()
print("\nMULTICOLLINEARITY EXAMPLE")
print(multi_corr)


# ------------------------------------------------------------
# 16.2 VISUALIZE MULTICOLLINEARITY
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
sns.heatmap(
    multi_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)

plt.title("Multicollinearity Example")
plt.show()


# ------------------------------------------------------------
# 16.3 SCATTER PLOT OF CORRELATED FEATURES
# ------------------------------------------------------------

sns.scatterplot(
    data=multi_df,
    x="study_hours",
    y="study_hours_double"
)
plt.title("Relationship Between Correlated Features")
plt.show()


# ------------------------------------------------------------
# 16.4 HOW TO HANDLE MULTICOLLINEARITY
# ------------------------------------------------------------

# Possible approaches:
# 1. Remove redundant features when appropriate.
# 2. Use domain knowledge to choose meaningful features.
# 3. Use regularization:
#    Ridge Regression can help stabilize coefficient estimates.
# 4. Use dimensionality reduction:
#    PCA can combine correlated information into components.
# 5. Examine Variance Inflation Factor (VIF)
#    when using coefficient-based regression models.

# IMPORTANT:
# Do not automatically remove every highly correlated feature. Check the target relationship, domain meaning, model type and validation results.


# ============================================================
# 17. FEATURE-TARGET RELATIONSHIPS
# ============================================================

# In supervised ML:

# X = Input features.
# y = Target variable.

# We want to understand whether input features are associated with the target.
# For regression: Target is numerical.
# For classification: Target is categorical.
# Visualization helps with initial feature analysis.
# Actual feature selection must be validated using appropriate training and validation procedures.


# ------------------------------------------------------------
# 17.1 NUMERICAL FEATURE VS NUMERICAL TARGET
# ------------------------------------------------------------

sns.regplot(
    data=df,
    x="study_hours",
    y="marks"
)
plt.title("Study Hours vs Target Marks")
plt.show()


# --------------------------------------------------------------------------------------
# 17.2 MULTIPLE FEATURE-TARGET RELATIONSHIPS - multiple regression plots using sub plots
# --------------------------------------------------------------------------------------    

# Create a figure containing 4 subplots arranged in 2 rows and 2 columns.
# figsize controls the overall figure width and height in inches.

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 9)
)

# List of numerical features to compare with marks.
features = [
    "study_hours",
    "attendance",
    "assignment_score",
    "previous_marks"
]

# axes.flatten() converts the 2D axes array into a 1D array.
# zip() pairs each subplot (ax) with one feature.
for ax, feature in zip(
    axes.flatten(),
    features
):
    # Create a regression plot for each feature against marks.
    sns.regplot(
        data=df,
        x=feature,                   # Feature on the X-axis
        y="marks",                   # Marks on the Y-axis
        scatter_kws={"alpha": 0.4},  # Makes scatter points 40% opaque
        line_kws={"color": "red"},   # Sets regression line color to red
        ax=ax                        # Draws the plot on the current subplot
    )
    # Set a separate title for each subplot.
    ax.set_title(
        f"{feature} vs Marks"
    )
# Automatically adjusts spacing between subplots.
plt.tight_layout()
# Display all four regression plots together.
plt.show()


# ============================================================
# 18. CORRELATION-BASED FEATURE ANALYSIS
# ============================================================

# Correlation can help identify:
# - Features associated with a numerical target.
# - Redundant numerical features.
# - Possible multicollinearity.

# Example:
# Rank numerical features by absolute Pearson
# correlation with the target marks.



# ------------------------------------------------------------
# 18.1 VISUALIZE FEATURE-TARGET CORRELATION
# ------------------------------------------------------------

# Create a figure with width 8 inches and height 5 inches.
plt.figure(figsize=(8, 5))
# Create a horizontal bar plot showing each feature's correlation with marks.
sns.barplot(
    x=target_correlations.values,  # Correlation values on the X-axis
    y=target_correlations.index    # Feature names on the Y-axis
)
# Add a title to the plot.
plt.title("Feature Correlation with Target")
# Set the X-axis label.
plt.xlabel("Pearson Correlation")
# Set the Y-axis label.
plt.ylabel("Feature")
# Automatically adjust spacing to prevent labels from getting cut off.
plt.tight_layout()
# Display the bar plot.
plt.show()

# IMPORTANT:
# Correlation-based ranking is only an initial screening tool.
# A low Pearson correlation does not mean a feature has no predictive value.
# A feature may have:
# - A nonlinear relationship.
# - An interaction with another feature.
# - Predictive information that Pearson does not capture.
# Correlation does not measure feature importance for every model.


# ============================================================
# 19. MISSING VALUES AND CORRELATION
# ============================================================

# Create a copy of numerical columns.
missing_df = df[numerical_columns].copy()

# Randomly select 5% of rows and make their attendance missing.
missing_rows = missing_df.sample(
    frac=0.05,
    random_state=42
).index
missing_df.loc[missing_rows, "attendance"] = np.nan

# Count missing values in each column.
print("\nMISSING VALUES:")
print(missing_df.isnull().sum())

# Calculate correlation even when some values are missing.
missing_corr = missing_df.corr()

print("\nCORRELATION WITH MISSING VALUES:")
print(missing_corr)

# Visualize missing values as a heatmap - This code visualizes where missing values are present in your dataset using a heatmap.

plt.figure(figsize=(10, 5))
# Create a heatmap showing missing and available values.
sns.heatmap(
    missing_df.isnull(),  # True = missing, False = available
    cbar=False,           # Hide the color bar
    yticklabels=False,    # Hide row labels
    cmap="viridis"        # Set the heatmap color palette
)
# Add a title.
plt.title("Missing Value Pattern")
# Display the heatmap.
plt.show()

# ============================================================
# 20. REGRESSION DIAGNOSTICS: RESIDUAL VISUALIZATION
# ============================================================

# Residual = Actual Value - Predicted Value
# Residual plots help us inspect whether a simple
# fitted relationship leaves systematic patterns.

# A random-looking residual distribution around zero is often desirable for a basic linear model.
# Patterns may suggest:
# - Nonlinearity.
# - Unequal variance.
# - Missing features.
# - Outliers.

# A Seaborn residual plot is a visualization.
# It does not replace a complete regression diagnostic.

# Get study hours as X (input feature).
x_values = df["study_hours"].to_numpy()
# Get actual marks as Y (target variable).
actual_values = df["marks"].to_numpy()

# ------------------------------------------------------------
# 20.1 LINEAR REGRESSION
# ------------------------------------------------------------

# Find the best-fitting straight line.
# 1 means we want a straight-line (linear) relationship.
# slope = slope of the line.
# intercept = point where the line crosses the Y-axis.

slope, intercept = np.polyfit(
    x_values,
    actual_values,
    1
)
# Predict marks using the regression equation.
predicted_values = slope * x_values + intercept #  y=mx+c

# ------------------------------------------------------------
# 20.2 CALCULATE RESIDUALS
# ------------------------------------------------------------

# Calculate the difference between actual and predicted marks.
residual_values = actual_values - predicted_values
# Store predicted marks and residuals in a DataFrame.
residual_df = pd.DataFrame({
    "predicted_marks": predicted_values,
    "residuals": residual_values
})

# ------------------------------------------------------------
# 20.3 RESIDUALS VS PREDICTED VALUES
# ------------------------------------------------------------

# Create a scatter plot to examine prediction errors.
sns.scatterplot(
    data=residual_df,
    x="predicted_marks",
    y="residuals",
    alpha=0.6
)
# Draw a horizontal reference line at zero.
plt.axhline(
    0,
    color="red",
    linestyle="--"
)
plt.title("Residuals vs Predicted Values")
plt.xlabel("Predicted Marks")
plt.ylabel("Residuals")
plt.show()


# ------------------------------------------------------------
# 20.4 RESIDUAL DISTRIBUTION
# ------------------------------------------------------------

# Display the distribution of prediction errors.
sns.histplot(
    data=residual_df,
    x="residuals",
    kde=True
)
plt.title("Residual Distribution")
plt.xlabel("Residuals")
plt.show()

# ============================================================
# 21. COMBINED RELATIONSHIP AND CORRELATION DASHBOARD
# ============================================================

# A practical EDA dashboard combines:
# 1. Feature distributions.
# 2. Pairwise relationships.
# 3. Regression relationships.
# 4. Correlation matrix.
#
# This gives an initial understanding of the dataset.


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 11)
)

# Relationship between study hours and marks.
sns.regplot(
    data=df,
    x="study_hours",
    y="marks",
    scatter_kws={"alpha": 0.4},
    ax=axes[0, 0]
)
axes[0, 0].set_title(
    "Study Hours vs Marks"
)
# Relationship between attendance and marks.
sns.scatterplot(
    data=df,
    x="attendance",
    y="marks",
    hue="result",
    alpha=0.7,
    ax=axes[0, 1]
)
axes[0, 1].set_title(
    "Attendance vs Marks"
)
# Pairwise relationship of selected variables.
sns.scatterplot(
    data=df,
    x="assignment_score",
    y="marks",
    hue="gender",
    alpha=0.6,
    ax=axes[1, 0]
)
axes[1, 0].set_title(
    "Assignment Score vs Marks"
)
# Correlation matrix.
sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    ax=axes[1, 1]
)
axes[1, 1].set_title(
    "Pearson Correlation"
)
plt.tight_layout()
plt.show()


# ============================================================
# 22. FIGURE-LEVEL VS AXES-LEVEL FUNCTIONS
# ============================================================

# Axes-level functions: sns.regplot(), sns.heatmap()
# They can draw on an existing Matplotlib Axes using ax=ax.
# Figure-level functions:
# sns.pairplot()
# sns.jointplot()
# sns.lmplot()
# They create and manage their own figures.
# Do not pass ax=ax to figure-level functions.


# ============================================================
# 23. IMPORTANT COMPARISON OF FUNCTIONS
# ============================================================

# pairplot(): Pairwise relationships among multiple numerical variables.
# jointplot(): One relationship plus marginal distributions.
# regplot(): Scatter plot with fitted regression line.
# lmplot(): Regression relationships with hue and faceting.
# heatmap(): Color-based visualization of a matrix, commonly used for correlation.

# WHEN TO USE WHICH?
# Many numerical variables: pairplot(), Two numerical variables and their distributions:
# jointplot() :  One regression relationship:
# regplot() : Regression comparisons across categories:
# lmplot() : Correlation matrix,heatmap().


# ============================================================
# 24. COMMON MISTAKES TO AVOID
# ============================================================

# 1. Assuming correlation proves causation.
# 2. Assuming zero Pearson correlation means
#    no relationship exists.
# 3. Using pairplot on too many features,
#    It can become slow and crowded.
# 4. Ignoring outliers.
#    Pearson correlation can be strongly affected by them.
# 5. Using correlation as the only feature selection method.
# 6. Confusing correlation with feature importance.
# 7. Ignoring multicollinearity.
#    Especially when interpreting linear model coefficients.
# 8. Interpreting confidence intervals as the full range of observed data.
# 9. Assuming regression lines prove a causal relationship.
# 10. Forgetting that pairplot, jointplot and lmplot are figure-level functions.
# 11. Using categorical variables directly in a numerical correlation matrix without appropriate encoding and interpretation.
# 12. Ignoring missing data when interpreting correlations.
# 13. Assuming every strongly correlated feature must be removed.
# 14. Treating Pearson, Spearman and Kendall as identical.


# ============================================================
# 25. PRACTICE TASKS
# ============================================================

# Complete these tasks independently.

# TASK 1:
# Create a pairplot for all numerical features.

# TASK 2:
# Add result as hue in pairplot.

# TASK 3:
# Use corner=True.

# TASK 4:
# Change the diagonal from histogram to KDE.

# TASK 5:
# Create a jointplot between attendance and marks.

# TASK 6:
# Experiment with scatter, reg, hist and kde
# as jointplot kinds.

# TASK 7:
# Create a regression plot between study_hours and marks.

# TASK 8:
# Disable the confidence interval in regplot.

# TASK 9:
# Try polynomial regression with order=2.

# TASK 10:
# Compare regression lines by gender using lmplot.

# TASK 11:
# Calculate Pearson correlation.

# TASK 12:
# Calculate Spearman correlation.

# TASK 13:
# Compare Pearson and Spearman results.

# TASK 14:
# Create a correlation heatmap.

# TASK 15:
# Add annotations and format values to two decimals.

# TASK 16:
# Create an upper-triangle mask.

# TASK 17:
# Create a nonlinear dataset and inspect its correlation.

# TASK 18:
# Create two strongly correlated features.

# TASK 19:
# Identify possible multicollinearity using a heatmap.

# TASK 20:
# Rank numerical features by absolute correlation with the target.

# TASK 21:
# Visualize residuals against predicted values.

# TASK 22:
# Build a complete relationship EDA dashboard.


# ============================================================
# 26. FINAL REVISION CHECKLIST
# ============================================================

# BASIC:

# [ ] Understand relationships between variables.
# [ ] Understand positive and negative correlation.
# [ ] Understand Pearson correlation.
# [ ] Understand Spearman correlation.
# [ ] Know what Kendall correlation measures.


# SEABORN FUNCTIONS:

# [ ] pairplot()
# [ ] jointplot()
# [ ] regplot()
# [ ] lmplot()
# [ ] heatmap()



# ADVANCED VISUALIZATION:

# [ ] Hue and category comparisons.
# [ ] Pairplot corner option.
# [ ] Jointplot kinds.
# [ ] Polynomial regression visualization.
# [ ] LOWESS visualization.
# [ ] Confidence intervals.
# [ ] Correlation matrix.
# [ ] Upper-triangle masking.


# ML CONCEPTS:

# [ ] Feature-target relationships.
# [ ] Correlation-based feature analysis.
# [ ] Multicollinearity.
# [ ] Nonlinear relationships.
# [ ] Residual analysis.
# [ ] Missing data and correlation.
# [ ] Limits of correlation-based feature selection.

# PRACTICAL SKILLS:

# [ ] Choose the correct visualization.
# [ ] Interpret the chart.
# [ ] Explain the correlation coefficient.
# [ ] Identify possible redundant features.
# [ ] Create a relationship EDA dashboard.


# ============================================================
# 27. WHAT YOU SHOULD BE ABLE TO DO AFTER PHASE 4
# ============================================================

# By completing this phase, you should be able to:
# 1. Explore pairwise relationships among numerical features.
# 2. Understand how numerical features relate to a target.
# 3. Distinguish linear and nonlinear relationships.
# 4. Calculate and interpret Pearson and Spearman correlation.
# 5. Build and interpret correlation heatmaps.
# 6. Identify possible multicollinearity.
# 7. Understand why correlation alone is insufficient for feature selection.
# 8. Compare relationships across categorical groups.
# 9. Inspect simple regression residuals.
# 10. Build a useful relationship-analysis dashboard.

# These skills will be useful when you begin:

# - Data Cleaning
# - Exploratory Data Analysis
# - Feature Engineering
# - Feature Selection
# - Scikit-learn
# - Regression and Classification

# ============================================================
# END OF SEABORN PHASE 4
# ============================================================
