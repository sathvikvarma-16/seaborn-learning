# ============================================================
# PHASE 7: SEABORN FOR EDA AND MACHINE LEARNING
# ============================================================

# OBJECTIVE:
# Learn how to use Seaborn effectively during Exploratory Data Analysis (EDA) before developing Machine Learning models.
# This phase connects everything you learned in Seaborn with practical data-analysis and ML workflows.

# TOPICS:

# 1. Understanding a dataset before modeling
# 2. Numerical and categorical feature analysis
# 3. Target-variable analysis
# 4. Class distribution and imbalance
# 5. Feature-target relationships
# 6. Correlation and feature redundancy
# 7. Outlier analysis
# 8. Before-and-after data-cleaning comparisons
# 9. Regression and residual visualization
# 10. Feature selection support using visualization
# 11. Complete EDA workflow

# PREVIOUSLY COMPLETED: NumPy and Pandas, Matplotlib, Seaborn setup and fundamental plots, Distribution and categorical plots
#  Relationship and correlation plots, Seaborn customization, Multi-plot layouts and faceting
# We will use those functions directly.
# We will not repeat their individual explanations.

# IMPORTANT:
# EDA is not simply creating graphs, It is the process of understanding data, identifying problems, discovering patterns and preparing data for machine learning.


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Seaborn version:", sns.__version__)

sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="deep"
)

# ============================================================
# 2. CREATE A SAMPLE MACHINE LEARNING DATASET
# ============================================================

# We will create a synthetic student-performance dataset.
# Our objective: Predict whether a student passes or fails.
# Target variable: result
# Features: study_hours, attendance, assignment_score, previous_marks, department, gender
# This is illustrative data, not a real student dataset.

np.random.seed(42)
number_of_students = 500
department = np.random.choice(
    ["CSE", "ECE", "MECH"],
    number_of_students,
    p=[0.50, 0.30, 0.20]
)
gender = np.random.choice(
    ["Male", "Female"],
    number_of_students
)
study_hours = np.random.uniform(
    0,
    10,
    number_of_students
)
attendance = np.random.uniform(
    40,
    100,
    number_of_students
)
assignment_score = np.random.uniform(
    30,
    100,
    number_of_students
)
previous_marks = np.random.uniform(
    25,
    100,
    number_of_students
)
marks = (
    10
    + study_hours * 4
    + attendance * 0.15
    + assignment_score * 0.20
    + previous_marks * 0.15
    + np.random.normal(0, 7, number_of_students)
)
marks = np.clip(marks, 0, 100)
result = np.where(
    marks >= 55,
    "Pass",
    "Fail"
)
df = pd.DataFrame({
    "department": department,
    "gender": gender,
    "study_hours": study_hours,
    "attendance": attendance,
    "assignment_score": assignment_score,
    "previous_marks": previous_marks,
    "marks": marks,
    "result": result
})
print("\nDataset preview:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nDataset information:")
df.info()


# ============================================================
# 3. UNDERSTAND THE DATASET BEFORE VISUALIZATION
# ============================================================

# Before creating graphs, understand the structure and quality of the dataset.


# ------------------------------------------------------------
# 3.1 CHECK THE FIRST AND LAST ROWS
# ------------------------------------------------------------

print(df.head())
print(df.tail())


# ------------------------------------------------------------
# 3.2 CHECK THE SHAPE
# ------------------------------------------------------------

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# ------------------------------------------------------------
# 3.3 CHECK COLUMN TYPES
# ------------------------------------------------------------

print(df.dtypes)

# Numerical columns: study_hours, attendance, assignment_score, previous_marks, marks
# Categorical columns: department, gender, result


# ------------------------------------------------------------
# 3.4 CHECK SUMMARY STATISTICS
# ------------------------------------------------------------

print(df.describe())

# describe() gives:
# count  = number of non-missing observations
# mean   = average
# std    = standard deviation
# min    = minimum
# 25%    = first quartile
# 50%    = median
# 75%    = third quartile
# max    = maximum

# For categorical columns:
print(df.describe(include="object"))


# ------------------------------------------------------------
# 3.5 CHECK MISSING VALUES
# ------------------------------------------------------------

print(df.isnull().sum())

# Missing values can affect:
# - Statistical summaries
# - Visualizations
# - Feature engineering
# - Model training

# Learn how to handle them properly in your Data Cleaning and Missing Data Handling phases.


# ------------------------------------------------------------
# 3.6 CHECK DUPLICATES
# ------------------------------------------------------------

print("Duplicate rows:", df.duplicated().sum())

# Duplicate observations can distort:
# - Distributions
# - Counts
# - Statistical estimates
# - Model training


# ============================================================
# 4. SEPARATE NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

# In EDA, it is useful to analyze numerical and categorical variables separately.

# Numerical variables: Represent measurable quantities.
# Categorical variables: Represent groups or labels.

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)


# ============================================================
# 5. UNIVARIATE ANALYSIS
# ============================================================

# UNIVARIATE ANALYSIS: Analysis of ONE variable at a time.

# Questions:
# - What is the distribution?
# - What is the average?
# - Are there outliers?
# - Is the distribution skewed?
# - Are there unusual values?

# This is generally one of the first steps in EDA.


# ------------------------------------------------------------
# 5.1 NUMERICAL FEATURE DISTRIBUTIONS
# ------------------------------------------------------------

for column in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x=column,
        kde=True,
        bins=25
    )
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# What to observe:
# 1. Shape of distribution.
# 2. Concentration of observations.
# 3. Possible skewness.
# 4. Multiple peaks.
# 5. Unusual values.

# KDE provides a smoothed representation of the distribution.
# It does not show the exact observation counts.


# ------------------------------------------------------------
# 5.2 BOX PLOTS FOR NUMERICAL FEATURES
# ------------------------------------------------------------

# Inspect outliers in each numerical column of the student dataset.

for column in numerical_columns:
    plt.figure(figsize=(8, 3))  # Create a separate figure for each numerical column
    sns.boxplot(
        data=df,             # Use the student dataset
        x=column             # Display the selected numerical column on the x-axis
    )
    plt.title(f"Outlier Inspection: {column}")  # Show which student feature is being inspected
    plt.xlabel(column)  # Label the x-axis with the feature name
    plt.tight_layout()  # Adjust spacing to prevent labels and titles from overlapping
    plt.show()  # Display the boxplot

# Observe:
# - Median
# - First and third quartiles
# - Interquartile range
# - Whiskers
# - Potential outliers

# A point outside the whiskers is a potential outlier.
# It is not automatically an error.


# ------------------------------------------------------------
# 5.3 CATEGORICAL FEATURE COUNTS
# ------------------------------------------------------------

for column in categorical_columns:
    plt.figure(figsize=(7, 4))
    sns.countplot(
        data=df,
        x=column,
        order=df[column].value_counts().index
    )
    plt.title(f"Category Counts: {column}")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

# Observe:
# - Number of observations per category.
# - Rare categories.
# - Imbalanced categories.
# - Unexpected category labels.


# ============================================================
# 6. BIVARIATE ANALYSIS
# ============================================================

# BIVARIATE ANALYSIS: Analysis of TWO variables together.

# Questions:
# - Are two numerical features related?
# - Does a numerical feature differ across categories?
# - Does one feature change with the target?

# Bivariate analysis helps identify potentially useful relationships before model development.


# ------------------------------------------------------------
# 6.1 NUMERICAL VS NUMERICAL
# ------------------------------------------------------------

sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="result",
    alpha=0.6
)
plt.title("Study Hours vs Marks")
plt.show()

# Observe:
# - Positive or negative relationship.
# - Nonlinear patterns.
# - Clusters.
# - Unusual observations.
# - Whether groups appear separated.
# A visible relationship does not prove causation.


# ------------------------------------------------------------
# 6.2 NUMERICAL VS CATEGORICAL
# ------------------------------------------------------------

sns.boxplot(
    data=df,
    x="department",
    y="marks",
    hue="gender"
)
plt.title("Marks by Department and Gender")
plt.show()

# Observe:
# - Differences in medians.
# - Differences in spread.
# - Potential outliers.
# - Differences between groups.


# ------------------------------------------------------------
# 6.3 CATEGORICAL VS CATEGORICAL
# ------------------------------------------------------------

sns.countplot(
    data=df,
    x="department",
    hue="result"
)
plt.title("Pass and Fail Counts by Department")
plt.show()

# Observe:
# - Category counts.
# - Differences between groups.
# - Whether one category has fewer observations.

# Raw counts can be misleading when group sizes differ.
# Consider proportions when comparing groups of different sizes.


# ------------------------------------------------------------
# 6.4 GROUP-WISE PROPORTIONS
# ------------------------------------------------------------

result_proportions = pd.crosstab(
    df["department"],
    df["result"],
    normalize="index"
)
print(result_proportions)

result_proportions.plot(
    kind="bar",
    stacked=True,
    figsize=(8, 5)
)
plt.title("Pass and Fail Proportions by Department")
plt.ylabel("Proportion")
plt.xlabel("Department")
plt.legend(title="Result")
plt.tight_layout()
plt.show()

# normalize="index": Divides each department's counts by its row total.
# Each department therefore adds up to 1.
# This makes proportions more suitable than raw counts when department sizes are different.


# ============================================================
# 7. TARGET VARIABLE ANALYSIS
# ============================================================

# In supervised machine learning, the target is the variable that the model is trained to predict.
# Here: Target = result
# The target variable must be understood before modeling.

# Questions:
# - How many classes exist?
# - How many observations belong to each class?
# - Is the target imbalanced?
# - Are some classes extremely rare?


# ------------------------------------------------------------
# 7.1 TARGET COUNTS
# ------------------------------------------------------------

print(df["result"].value_counts())
sns.countplot(
    data=df,
    x="result",
    order=df["result"].value_counts().index
)
plt.title("Target Class Distribution")
plt.show()


# ------------------------------------------------------------
# 7.2 TARGET PROPORTIONS
# ------------------------------------------------------------

target_proportions = (
    df["result"]
    .value_counts(normalize=True)
)
print(target_proportions)
print(
    "Target percentages:\n",
    target_proportions * 100
)


# ------------------------------------------------------------
# 7.3 WHAT IS CLASS IMBALANCE?
# ------------------------------------------------------------

# Class imbalance occurs when one target class has significantly more observations than another.
# Example: Pass = 90% | Fail = 10%

# A model that predicts Pass for every observation could achieve 90% accuracy while completely failing to identify Fail cases.

# Therefore: Accuracy alone may be misleading.
# In such situations, examine : Precision, Recall, F1-score, Confusion matrix, Precision-recall curve
# These metrics will be covered in the ML evaluation phase.

# Important:
# This dataset may not be severely imbalanced.
# We are learning how to inspect class distribution.


# ------------------------------------------------------------
# 7.4 TARGET VS NUMERICAL FEATURES
# ------------------------------------------------------------

for column in numerical_columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=df,
        x="result",
        y=column
    )
    plt.title(f"{column} by Target Class")
    plt.tight_layout()
    plt.show()

# This helps us investigate whether numerical features show different distributions across target classes.
# A feature with visibly different distributions may be useful for prediction.
# But visual separation alone does not prove predictive value.


# ------------------------------------------------------------
# 7.5 TARGET VS CATEGORICAL FEATURES
# ------------------------------------------------------------

for column in ["department", "gender"]:
    plt.figure(figsize=(7, 4))
    sns.countplot(
        data=df,
        x=column,
        hue="result"
    )
    plt.title(f"{column} vs Target")
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. MULTIVARIATE ANALYSIS
# ============================================================

# MULTIVARIATE ANALYSIS: Analysis involving THREE OR MORE variables.
# It helps us understand how several features interact.

# Examples: Study hours vs marks, grouped by gender | Marks distributions across departments and result | Correlations among multiple numerical features.


# ------------------------------------------------------------
# 8.1 PAIRPLOT
# ------------------------------------------------------------

sns.pairplot(
    data=df,
    vars=[
        "study_hours",
        "attendance",
        "assignment_score",
        "previous_marks",
        "marks"
    ],
    hue="result",
    corner=True,
    plot_kws={
        "alpha": 0.5,
        "s": 25
    }
)
plt.show()

# Use pairplot when you want a quick overview of relationships among several numerical features.
# Be careful with large datasets because pairplots can become computationally expensive and cluttered.


# ------------------------------------------------------------
# 8.2 MULTIVARIATE SCATTERPLOT
# ------------------------------------------------------------

sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="result",
    style="gender",
    size="attendance",
    sizes=(30, 150),
    alpha=0.7
)
plt.title("Multivariable Student Performance Analysis")
plt.show()

# x: Study hours.
# y: Marks.
# hue: Result class.
# style: Gender.
# size: Attendance.
# Several variables can be represented in one visualization.
# However, avoid adding so many dimensions that the chart becomes difficult to interpret.


# ------------------------------------------------------------
# 8.3 FACETED ANALYSIS
# ------------------------------------------------------------

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    hue="result",
    kind="scatter",
    height=4,
    aspect=1.1
)
plt.show()

# This helps compare the relationship between study hours and marks separately across departments.


# ============================================================
# 9. CORRELATION AND FEATURE REDUNDANCY
# ============================================================

# Correlation measures the strength and direction of a relationship between numerical variables.

# Common methods: Pearson, Measures linear correlation.
# Spearman: Measures monotonic association using ranks.
# Correlation values range from -1 to +1.
# +1 = perfect positive correlation.
# -1 = perfect negative correlation.
#  0 = no linear correlation for Pearson.

# Important:
# Zero Pearson correlation does not rule out a nonlinear relationship.


# ------------------------------------------------------------
# 9.1 PEARSON CORRELATION
# ------------------------------------------------------------

pearson_corr = df[numerical_columns].corr(
    method="pearson"
)
print(pearson_corr)
sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Pearson Correlation Matrix")
plt.show()


# ------------------------------------------------------------
# 9.2 SPEARMAN CORRELATION
# ------------------------------------------------------------

spearman_corr = df[numerical_columns].corr(
    method="spearman"
)
print(spearman_corr)
sns.heatmap(
    spearman_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Spearman Correlation Matrix")
plt.show()

# Pearson: Useful for measuring linear relationships.
# Spearman: Useful for monotonic relationships, including some nonlinear relationships.
# Neither method proves causation.


# ------------------------------------------------------------
# 9.3 IDENTIFY HIGHLY CORRELATED FEATURE PAIRS
# ------------------------------------------------------------

# We can inspect the upper triangle of the correlation matrix to avoid printing duplicate feature pairs.
# Extract only the upper triangle of the Pearson correlation matrix.
# This avoids duplicate pairs (e.g., age-study_hours and study_hours-age).

upper_triangle = pearson_corr.where(
    np.triu(
        np.ones(pearson_corr.shape),  # Create a square matrix filled with 1s
        k=1                          # Keep only values above the main diagonal
    ).astype(bool)                   # Convert 1s and 0s into True and False
)
# Extract valid correlation pairs and sort them by absolute correlation strength.
high_correlation_pairs = (
    upper_triangle
    .stack()                         # Convert the matrix into a Series of variable pairs
    .sort_values(
        ascending=False,              # Sort from strongest to weakest
        key=abs                        # Sort using absolute values, ignoring the sign
    )
)
print(high_correlation_pairs)
print("\nFeature correlation pairs:")
print(high_correlation_pairs)

# A high absolute correlation may indicate redundancy.

# However:
# - High correlation does not automatically mean a feature should be removed.
# - Low correlation does not automatically mean a featureis useless.
# - Correlation with the target is not the same as model feature importance.
# Feature selection requires further analysis and validation.


# ------------------------------------------------------------
# 9.4 CORRELATION WITH THE TARGET
# ------------------------------------------------------------

# Our target is categorical.
# For this illustrative binary target, create a numerical representation for exploratory correlation only.

df["result_encoded_for_eda"] = (
    df["result"]
    .map({
        "Fail": 0,
        "Pass": 1
    })
)
target_corr = df[
    numerical_columns + ["result_encoded_for_eda"]
].corr()
print(target_corr["result_encoded_for_eda"].sort_values(
    ascending=False
))
sns.heatmap(
    target_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Feature and Encoded Target Correlation")
plt.show()

# This is only a simple exploratory analysis.
# For a binary target, Pearson correlation with its 0/1 encoding is related to point-biserial correlation.
# For multiclass targets, do not blindly convert labels into arbitrary integers and interpret Pearson correlation.
# Use suitable statistical methods and model-based evaluation for more complex target types.


# ============================================================
# 10. OUTLIER ANALYSIS
# ============================================================

# Outliers are observations that are unusually far from the general pattern of the data.
# Possible causes:
# - Data-entry errors.
# - Measurement errors.
# - Rare but valid observations.
# - Natural variation.

# Do not automatically remove outliers.
# First investigate whether they are valid observations.


# ------------------------------------------------------------
# 10.1 VISUAL OUTLIER INSPECTION
# ------------------------------------------------------------

sns.boxplot(
    data=df,
    x="marks"
)
plt.title("Marks Outlier Inspection")
plt.show()


# ------------------------------------------------------------
# 10.2 IQR METHOD
# ------------------------------------------------------------

# Calculate the first quartile (25% of students scored at or below this value).
Q1 = df["marks"].quantile(0.25)
# Calculate the third quartile (75% of students scored at or below this value).
Q3 = df["marks"].quantile(0.75)
# Calculate the Interquartile Range (middle 50% of students' marks).
IQR = Q3 - Q1
# Calculate the lower limit for detecting potential low outliers.
lower_bound = Q1 - 1.5 * IQR
# Calculate the upper limit for detecting potential high outliers.
upper_bound = Q3 + 1.5 * IQR
# Select students whose marks fall outside the calculated limits.
outliers = df[
    (df["marks"] < lower_bound)  # Marks below the lower limit
    |
    (df["marks"] > upper_bound)  # Marks above the upper limit
]
# Display the calculated quartiles and outlier detection limits.
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
# Display the number of students identified as potential outliers.
print("Potential outlier count:", len(outliers))
# Display the complete records of students identified as potential outliers.
print(outliers)

# IQR = Q3 - Q1
# Lower fence = Q1 - 1.5 * IQR
# Upper fence = Q3 + 1.5 * IQR
# Observations outside these fences are flagged as potential outliers.
# The IQR rule is a screening method, not a universal definition of invalid data.


# ------------------------------------------------------------
# 10.3 OUTLIERS ACROSS GROUPS
# ------------------------------------------------------------

sns.boxplot(
    data=df,
    x="department",
    y="marks"
)
plt.title("Potential Outliers by Department")
plt.show()

# Group-wise inspection may reveal that an observation is unusual in one group but normal in another.
# Consider the context of the feature before deciding what to do with an outlier.


# ============================================================
# 11. BEFORE-AND-AFTER DATA CLEANING COMPARISON
# ============================================================

# Visualization is useful for checking the effect of data-cleaning operations.
# We will create a separate example with missing values and an extreme value.
# This section demonstrates visualization only.
# Detailed cleaning methods will be covered separately.


# ------------------------------------------------------------
# 11.1 CREATE A DATASET WITH ISSUES
# ------------------------------------------------------------

# Create a separate copy of the original student dataset.
# This keeps the original df unchanged while introducing artificial data-quality issues.
dirty_df = df.copy()
# Introduce missing values in the study_hours column.
dirty_df.loc[
    dirty_df.sample(
        frac=0.05,          # Randomly select 5% of the student records.
        random_state=42     # Ensure the same records are selected every time.
    ).index,
    "study_hours"           # Specify the column where missing values will be introduced.
] = np.nan                  # Replace the selected values with NaN.

# Introduce one extreme value in the attendance column for outlier demonstration.
dirty_df.loc[
    dirty_df.index[0],      # Select the first student's record.
    "attendance"            # Specify the column to modify.
] = 250                     # Set attendance to 250, an unrealistic value if attendance is a percentage.

# Count the number of missing values in every column.
print(dirty_df.isnull().sum())  # Display missing-value counts column by column.


# ------------------------------------------------------------
# 11.2 VISUALIZE BEFORE CLEANING
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)
sns.histplot(
    data=dirty_df,
    x="study_hours",
    kde=True,
    ax=axes[0]
)
axes[0].set_title("Study Hours Before Cleaning")
sns.boxplot(
    data=dirty_df,
    x="attendance",
    ax=axes[1]
)
axes[1].set_title("Attendance Before Cleaning")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 11.3 APPLY SIMPLE DEMONSTRATION CLEANING
# ------------------------------------------------------------

cleaned_df = dirty_df.copy()
# Fill missing study hours using the median.
cleaned_df["study_hours"] = (
    cleaned_df["study_hours"]
    .fillna(
        cleaned_df["study_hours"].median()
    )
)

# Remove observations with attendance outside the known valid range of 0 to 100.
# This rule is justified here because attendance is represented as a percentage.

cleaned_df = cleaned_df[
    cleaned_df["attendance"].between(0, 100)
].copy()


# ------------------------------------------------------------
# 11.4 VISUALIZE AFTER CLEANING
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 4)
)

sns.histplot(
    data=cleaned_df,
    x="study_hours",
    kde=True,
    ax=axes[0]
)
axes[0].set_title("Study Hours After Cleaning")
sns.boxplot(
    data=cleaned_df,
    x="attendance",
    ax=axes[1]
)
axes[1].set_title("Attendance After Cleaning")
plt.tight_layout()
plt.show()

# Compare:
# - Missing-value effects.
# - Distribution changes.
# - Invalid-value removal.
# - Number of rows before and after cleaning.

print("Before cleaning:", dirty_df.shape)
print("After cleaning:", cleaned_df.shape)

# IMPORTANT:
# We used the full dataset for this demonstration.
# In a real ML workflow, preprocessing parameters such as medians should be learned from training data only.
# Do not calculate imputation values using the complete dataset before splitting into training and test sets.
# This avoids data leakage.


# ============================================================
# 12. REGRESSION AND RESIDUAL VISUALIZATION
# ============================================================

# Regression visualization helps inspect the relationship between predicted values and actual values.
# Residual = Actual value - Predicted value
# Residual analysis helps identify patterns that may indicate problems with a regression model.
# We will create a simple illustrative linear prediction.
# This is NOT a trained Scikit-learn model.


# ------------------------------------------------------------
# 12.1 CREATE SIMPLE PREDICTIONS
# ------------------------------------------------------------

x_values = df["study_hours"].to_numpy()  # Convert study_hours into a NumPy array (independent variable).
actual_values = df["marks"].to_numpy()   # Convert actual marks into a NumPy array (dependent variable).
slope, intercept = np.polyfit(  # Calculate the slope and intercept of the best-fit straight line.
    x_values,                   # Input x-values (study hours).
    actual_values,              # Corresponding actual y-values (marks).
    1                           # Fit a polynomial of degree 1 (linear regression).
)
predicted_values = (
    slope * x_values + intercept         # Calculate predicted marks using the linear equation y = mx + c.
)
residual_values = (
    actual_values - predicted_values     # Calculate prediction errors (actual marks - predicted marks).
)
residual_df = pd.DataFrame({             # Create a DataFrame to organize actual, predicted, and residual values.
    "actual_marks": actual_values,       # Store the original marks of each student.
    "predicted_marks": predicted_values, # Store the marks predicted by the regression line.
    "residuals": residual_values         # Store the difference between actual and predicted marks.
})
print(residual_df.head())  # Display the first five rows to inspect the prediction errors.


# ------------------------------------------------------------
# 12.2 ACTUAL VS PREDICTED
# ------------------------------------------------------------

sns.scatterplot(
    data=residual_df,
    x="actual_marks",
    y="predicted_marks",
    alpha=0.6
)
minimum_value = min(
    residual_df["actual_marks"].min(),
    residual_df["predicted_marks"].min()
)
maximum_value = max(
    residual_df["actual_marks"].max(),
    residual_df["predicted_marks"].max()
)
plt.plot(
    [minimum_value, maximum_value],
    [minimum_value, maximum_value],
    linestyle="--",
    color="red"
)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Values")
plt.show()

# The diagonal line represents perfect predictions.
# Points closer to the line have smaller prediction errors.
# This plot alone is not enough to evaluate a model.


# ------------------------------------------------------------
# 12.3 RESIDUALS VS PREDICTED
# ------------------------------------------------------------

sns.scatterplot(
    data=residual_df,
    x="predicted_marks",
    y="residuals",
    alpha=0.6
)
plt.axhline(
    0,
    linestyle="--",
    color="red"
)
plt.xlabel("Predicted Marks")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.show()

# What to observe:
# Ideally, residuals should be distributed around zero without a clear systematic pattern.
# Possible warning signs:
# 1. Curved pattern: May indicate a nonlinear relationship.
# 2. Funnel shape: May indicate non-constant residual variance.
# 3. Strong clusters: May indicate groups or missing variables.
# 4. Extreme residuals: May indicate observations poorly predicted by the model.
# These patterns are diagnostic clues, not automatic proof of a particular model problem.


# ------------------------------------------------------------
# 12.4 RESIDUAL DISTRIBUTION
# ------------------------------------------------------------

sns.histplot(
    data=residual_df,
    x="residuals",
    kde=True,
    bins=25
)
plt.axvline(
    0,
    linestyle="--",
    color="red"
)
plt.title("Residual Distribution")
plt.show()

# Inspect the shape and spread of residuals.
# Do not assume that every machine learning model requires  perfectly normally distributed residuals.
# Requirements depend on the model and the statistical inference being performed.


# ============================================================
# 13. FEATURE SELECTION SUPPORT USING VISUALIZATION
# ============================================================

# Feature selection means choosing useful input variables for a machine learning model
# Seaborn can help us investigate:
# - Relationships with the target.
# - Feature redundancy.
# - Differences between target classes.
# - Outliers.
# - Distribution patterns.

# However:
# A graph should not be the only basis for selecting or removing a feature.


# ------------------------------------------------------------
# 13.1 COMPARE NUMERICAL FEATURES WITH THE TARGET
# ------------------------------------------------------------

for column in numerical_columns:
    sns.boxplot(
        data=df,
        x="result",
        y=column
    )
    plt.title(f"{column} vs Result")
    plt.show()

# Questions:
# - Do target classes have different distributions?
# - Are the medians different?
# - Is there substantial overlap?
# - Are there unusual observations?


# ------------------------------------------------------------
# 13.2 INSPECT FEATURE CORRELATIONS
# ------------------------------------------------------------

sns.heatmap(
    pearson_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Feature Redundancy Inspection")
plt.show()

# Highly correlated features may contain overlapping information.
# But do not remove features solely because their correlation is high.

# Consider:
# - Model type.
# - Domain knowledge.
# - Predictive performance.
# - Validation results.
# - Multicollinearity concerns.


# ------------------------------------------------------------
# 13.3 COMPARE DISTRIBUTIONS BY TARGET
# ------------------------------------------------------------

sns.displot(
    data=df,
    x="study_hours",
    hue="result",
    kind="kde",
    fill=True,
    common_norm=False
)
plt.show()

# Compare how study-hours distributions differ between Pass and Fail observations.
# This may reveal useful patterns, but it does not prove that study hours alone can predict the result.


# ============================================================
# 14. DATA LEAKAGE: AN IMPORTANT ML CONCEPT
# ============================================================

# Data leakage occurs when information that would not legitimately be available during prediction is used model training.

# Examples:

# 1. Using information recorded after the target event.
# 2. Calculating preprocessing statistics using the entire dataset before train/test splitting.
# 3. Including a feature that directly reveals the target.
# 4. Performing feature selection using test-set information.
# Visualization can help detect suspicious relationships, but it cannot reliably detect every form of leakage.

# IMPORTANT WORKFLOW:
# 1. Understand the data.
# 2. Identify the target.
# 3. Split into training and test sets.
# 4. Fit preprocessing only on training data.
# 5. Transform validation/test data using fitted parameters.
# 6. Train and evaluate the model.
# Later, Scikit-learn pipelines will help implement this workflow safely.


# ============================================================
# 15. COMPLETE EDA WORKFLOW: THE ORDER TO FOLLOW
# ============================================================

# When you receive a new dataset, follow this order.

# STEP 1: UNDERSTAND THE DATA
# - Inspect head() and tail().
# - Check shape.
# - Check column names.
# - Check data types.
# - Understand what each column represents.

# STEP 2: CHECK DATA QUALITY
# - Missing values.
# - Duplicate rows.
# - Invalid values.
# - Inconsistent categories.
# - Incorrect data types.

# STEP 3: SEPARATE FEATURE TYPES
# - Numerical columns.
# - Categorical columns.
# - Datetime columns.
# -Target variable.

# STEP 4: UNIVARIATE ANALYSIS
# - Histograms.
# - KDE plots.
# - Boxplots.
# - Countplots.

# STEP 5: BIVARIATE ANALYSIS
# - Numerical vs numerical.
# - Numerical vs categorical.
# - Categorical vs categorical.

# STEP 6: TARGET ANALYSIS
# - Target distribution.
# - Class proportions.
# - Feature distributions by target.

# STEP 7: MULTIVARIATE ANALYSIS
# - Pairplots.
# - Correlation heatmaps.
# - Grouped visualizations.
# - Faceted plots.

# STEP 8: OUTLIER ANALYSIS
# - Boxplots.
# - IQR screening.
# - Domain-specific checks.

# STEP 9: CLEANING AND VALIDATION
# - Apply justified cleaning.
# - Compare before and after.
# - Check whether cleaning changed distributions.

# STEP 10: PREPARE FOR MACHINE LEARNING
# - Identify potential features.
# - Check for leakage.
# - Split data correctly.
# - Build preprocessing pipelines.
# - Train and evaluate models.
# This is a general workflow.
# The exact order can change depending on the dataset.


# ============================================================
# 16. COMMON EDA MISTAKES
# ============================================================

# 1. Creating graphs without asking a specific question.
# 2. Treating correlation as causation.
# 3. Removing every point identified as an outlier.
# 4. Ignoring missing values.
# 5. Comparing raw counts when group sizes differ.
# 6. Ignoring target class imbalance.
# 7. Using too many variables in one visualization.
# 8. Assuming visual separation guarantees good prediction.
# 9. Removing features based only on correlation.
# 10. Ignoring data leakage.
# 11. Using the complete dataset to calculate preprocessing values before splitting into training and test sets.
# 12. Treating synthetic demonstration data as real evidence.
# 13. Using a chart as a replacement for statistical validation and model evaluation.


# ============================================================
# 17. PRACTICE TASKS
# ============================================================

# Complete these tasks independently.

# DATASET UNDERSTANDING

# TASK 1:
# Inspect a dataset using head(), info() and describe().

# TASK 2:
# Identify numerical and categorical features.

# TASK 3:
# Check missing values and duplicate rows.


# UNIVARIATE ANALYSIS

# TASK 4:
# Create histograms for all numerical features.

# TASK 5:
# Create boxplots for all numerical features.

# TASK 6:
# Create countplots for categorical features.


# BIVARIATE ANALYSIS

# TASK 7:
# Compare two numerical features using scatterplot().

# TASK 8:
# Compare a numerical feature across categories.

# TASK 9:
# Compare two categorical variables using countplot().

# TASK 10:
# Calculate group-wise proportions using crosstab().


# TARGET ANALYSIS

# TASK 11:
# Inspect target counts and percentages.

# TASK 12:
# Visualize numerical features against the target.

# TASK 13:
# Visualize categorical features against the target.


# MULTIVARIATE ANALYSIS

# TASK 14:
# Create a pairplot using selected numerical features.

# TASK 15:
# Create a multivariable scatterplot.

# TASK 16:
# Create a faceted plot using department and gender.


# CORRELATION AND OUTLIERS

# TASK 17:
# Compare Pearson and Spearman correlations.

# TASK 18:
# Identify highly correlated feature pairs.

# TASK 19:
# Identify potential outliers using the IQR method.

# TASK 20:
# Compare outlier distributions across categories.


# CLEANING AND ML PREPARATION

# TASK 21:
# Create a dataset with missing values.

# TASK 22:
# Visualize the data before and after cleaning.

# TASK 23:
# Create a simple regression prediction and residual plot.

# TASK 24:
# Explain why correlation alone is not enough for
# feature selection.

# TASK 25:
# Explain how data leakage can happen during preprocessing.


# ============================================================
# 18. PHASE 7 REVISION CHECKLIST
# ============================================================

# [ ] Understand the purpose of EDA.
# [ ] Inspect dataset structure and quality.
# [ ] Separate numerical and categorical features.
# [ ] Perform univariate analysis.
# [ ] Perform bivariate analysis.
# [ ] Perform multivariate analysis.
# [ ] Understand target-variable analysis.
# [ ] Inspect class imbalance.
# [ ] Compare features across target classes.
# [ ] Interpret correlation matrices.
# [ ] Identify possible feature redundancy.
# [ ] Inspect potential outliers.
# [ ] Compare data before and after cleaning.
# [ ] Understand actual vs predicted plots.
# [ ] Understand residual plots.
# [ ] Understand visualization's role in feature selection.
# [ ] Understand data leakage.
# [ ] Know the general EDA workflow.
# [ ] Complete the practice tasks.


# ============================================================
# 19. WHAT YOU SHOULD BE ABLE TO DO AFTER PHASE 7
# ============================================================

# You should be able to take an unfamiliar tabular dataset and perform a structured exploratory analysis.
# Specifically, you should be able to:
# 1. Understand its structure.
# 2. Identify data-quality problems.
# 3. Analyze numerical and categorical variables.
# 4. Understand the target variable.
# 5. Investigate relationships between features.
# 6. Identify potential outliers.
# 7. Inspect correlations and possible redundancy.
# 8. Compare data before and after cleaning.
# 9. Interpret basic regression diagnostics.
# 10. Prepare an informed plan for machine learning.

# You do not need to memorize every plotting parameter.
# Focus on choosing the correct visualization and what it tells you.


# ============================================================
# PHASE 7 COMPLETE
# ============================================================
