# ============================================================
# SEABORN PHASE 3: CATEGORICAL DATA VISUALIZATION
# ============================================================
#
# LEVEL: Beginner to Advanced (ML Engineering Preparation)

# TOPICS COVERED:

# 1. Understanding categorical data
# 2. stripplot()
# 3. swarmplot()
# 4. boxenplot()
# 5. pointplot()
# 6. catplot()
# 7. Categorical ordering
# 8. Hue and grouped comparisons
# 9. Dodge and jitter
# 10. Native scale and categorical axes
# 11. Figure-level vs Axes-level functions
# 12. Combining categorical plots
# 13. Categorical visualization for EDA
# 14. Categorical visualization for ML
# 15. Class-wise feature analysis
# 16. Outlier and distribution interpretation
# 17. Common mistakes and best practices
#
# IMPORTANT:
# This file uses generated sample data.
# It does not require downloading an external dataset.

# Install if required:
# pip install numpy pandas matplotlib seaborn
#
# ============================================================


# ============================================================
# 0. IMPORT LIBRARIES AND SETUP
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set a consistent visualization theme
sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="Set2"
)

# Set random seed for reproducible sample data
rng = np.random.default_rng(42)

print("Seaborn Version:", sns.__version__)


# ============================================================
# 1. UNDERSTANDING CATEGORICAL DATA
# ============================================================

# Categorical variables represent groups or labels.

# Examples:
# Gender       -> Male, Female
# Department   -> CSE, ECE, Mechanical
# Result       -> Pass, Fail
# Product      -> Laptop, Mobile, Tablet
# CustomerType -> New, Returning

# Categorical variables can be:

# 1. Nominal:
#    Categories without a natural order.
#    Example: Color, Department, Country.

# 2. Ordinal:
#    Categories with a meaningful order.
#    Example: Low, Medium, High.

# Numerical variables:
#    Age, Salary, Marks, Study Hours.

# Categorical plots help us understand:
# - Differences between categories.
# - Distribution within each category.
# - Spread and variability.
# - Outliers.
# - Relationships between categorical and numerical variables.
# - Differences between classes in ML datasets.


# ============================================================
# 2. CREATE A SAMPLE STUDENT DATASET
# ============================================================

number_of_students = 300

departments = ["CSE", "ECE", "EEE", "Mechanical"]
genders = ["Male", "Female"]
study_levels = ["Low", "Medium", "High"]
results = ["Fail", "Pass"]

df = pd.DataFrame({
    "student_id": range(1, number_of_students + 1),
    "department": rng.choice(
        departments,
        number_of_students
    ),
    "gender": rng.choice(
        genders,
        number_of_students
    ),
    "study_level": rng.choice(
        study_levels,
        number_of_students,
        p=[0.25, 0.45, 0.30]
    ),
    "study_hours": np.clip(
        rng.normal(4.5, 1.8, number_of_students),
        0.5,
        10
    ),
    "attendance": np.clip(
        rng.normal(80, 12, number_of_students),
        35,
        100
    ),
    "marks": np.clip(
        rng.normal(70, 15, number_of_students),
        0,
        100
    ),
    "age": rng.normal(
        20,
        1.5,
        number_of_students
    ).round(0)
})

# Create result category based on marks
df["result"] = np.where(
    df["marks"] >= 50,
    "Pass",
    "Fail"
)

# Convert categorical columns to category dtype
categorical_columns = [
    "department",
    "gender",
    "study_level",
    "result"
]

for column in categorical_columns:
    df[column] = df[column].astype("category")

print("\nSTUDENT DATASET")
print(df.head())

print("\nDATASET INFORMATION")
print(df.info())

print("\nCATEGORICAL VALUE COUNTS")
for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts())


# ============================================================
# 3. STRIPPLOT()
# ============================================================

# stripplot() displays individual observations as points.
#
# It is useful for:
# - Seeing individual data points.
# - Comparing numerical values across categories.
# - Identifying clusters.
# - Understanding the spread of observations.
#
# Example:
# Compare marks across different departments.


# ------------------------------------------------------------
# 3.1 BASIC STRIPPLOT
# ------------------------------------------------------------

sns.stripplot(
    data=df,
    x="department",  
    y="marks"
)

plt.title("Marks Across Departments")
plt.xlabel("Department")
plt.ylabel("Marks")
plt.show()


# ------------------------------------------------------------
# 3.2 UNDERSTANDING JITTER
# ------------------------------------------------------------

# Problem:
# Many observations may have identical or similar values.
# Their points can overlap.

# jitter=True:
# Adds a small random horizontal displacement.
# This makes overlapping observations easier to see.

sns.stripplot(
    data=df,
    x="department",
    y="marks",
    jitter=True
)
plt.title("Strip Plot with Jitter")
plt.show()


# ------------------------------------------------------------
# 3.3 CONTROL JITTER
# ------------------------------------------------------------

# A numerical jitter value controls the amount of displacement.
# Smaller jitter:
# Points remain closer to their category position.
# Larger jitter:
# Points spread farther horizontally.

sns.stripplot(
    data=df,
    x="department",
    y="marks",
    jitter=0.25
)   
plt.title("Strip Plot with Controlled Jitter")
plt.show()


# ------------------------------------------------------------
# 3.4 STRIPPLOT WITH HUE
# ------------------------------------------------------------

# hue separates observations using another categorical variable.

sns.stripplot(
    data=df,
    x="department",
    y="marks",
    hue="gender", # color difference among points
    dodge=True, # dodge=True is used to separate overlapping categorical plots side by side instead of placing them on top of each other.
    jitter=True, # jitter=True is used to add small random horizontal or vertical variations to data points so that overlapping points become easier to see.
    alpha=0.7,
    palette="Set1"
)

plt.title("Marks by Department and Gender")
plt.show()


# ------------------------------------------------------------
# 3.5 IMPORTANT STRIPPLOT PARAMETERS
# ------------------------------------------------------------

# data:
# DataFrame containing observations.

# x:
# Categorical variable on horizontal axis.

# y:
# Numerical variable on vertical axis.

# hue:
# Additional categorical grouping.

# jitter:
# Spreads overlapping points.

# dodge:
# Separates hue groups.

# alpha:
# Controls transparency.

# size:
# Controls marker size.

# palette:
# Controls category colors.

# order:
# Controls category order.

# orient:
# Controls plot orientation.

# legend:
# Controls whether a legend is displayed.


# ------------------------------------------------------------
# 3.6 HORIZONTAL STRIPPLOT
# ------------------------------------------------------------

sns.stripplot(
    data=df,
    x="marks",
    y="department",
    jitter=True,
    alpha=0.7
)

plt.title("Horizontal Strip Plot")
plt.show()


# ============================================================
# 4. SWARMPLOT()
# ============================================================

# swarmplot() displays individual observations.

# Unlike stripplot(), it attempts to position points so
# that they do not overlap.

# Useful when:
# - Dataset is relatively small.
# - Individual observations matter.
# - You want to see the distribution of points clearly.

# Important:
# swarmplot can become slow or crowded with large datasets.


# ------------------------------------------------------------
# 4.1 BASIC SWARMPLOT
# ------------------------------------------------------------

sns.swarmplot(
    data=df.sample(100, random_state=42),
    x="department",
    y="marks"
)
plt.title("Marks Across Departments Using Swarm Plot")
plt.show()


# ------------------------------------------------------------
# 4.2 SWARMPLOT WITH HUE
# ------------------------------------------------------------

sample_df = df.sample(100, random_state=42)
sns.swarmplot(
    data=sample_df,
    x="department",
    y="marks",
    hue="gender",
    dodge=True,
    palette="Set1",
    size=4
)

plt.title("Swarm Plot by Department and Gender")
plt.show()


# ------------------------------------------------------------
# 4.3 SWARMPLOT WITH ANOTHER PLOT
# ------------------------------------------------------------

# A swarm plot can be combined with a box plot.

# Box plot:
# Shows median, quartiles and possible outliers.

# Swarm plot:
# Shows individual observations.

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=sample_df,
    x="department",
    y="marks",
    color="lightgray",
    showfliers=False,
    ax=ax
)

sns.swarmplot(
    data=sample_df,
    x="department",
    y="marks",
    color="black",
    alpha=0.7,
    size=3,
    ax=ax
)

ax.set_title("Box Plot Combined with Swarm Plot")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 4.4 STRIPPLOT VS SWARMPLOT
# ------------------------------------------------------------

# stripplot():
# - Uses jitter to spread observations.
# - Can still have overlapping points.
# - Generally faster for larger datasets.
#
# swarmplot():
# - Attempts to prevent point overlap.
# - Shows individual observations clearly.
# - Can be computationally expensive for large datasets.
#
# Use stripplot for larger datasets.
# Use swarmplot when the number of observations is manageable.


# ============================================================
# 5. BOXENPLOT()
# ============================================================

# boxenplot() is also called a letter-value plot.

# It is designed to show the distribution of numerical data
# across categories, particularly for larger datasets.

# It displays multiple nested quantile regions.

# Compared with a traditional boxplot:
# - A boxplot mainly emphasizes quartiles and median.
# - A boxenplot shows additional quantile information.
# - It can provide more detail about distribution tails.
# Useful for large datasets and detailed distribution analysis.


# ------------------------------------------------------------
# 5.1 BASIC BOXENPLOT
# ------------------------------------------------------------

sns.boxenplot(
    data=df,
    x="department",
    y="marks"
)
plt.title("Marks Distribution Using Boxen Plot")
plt.show()


# ------------------------------------------------------------
# 5.2 BOXENPLOT WITH HUE
# ------------------------------------------------------------

sns.boxenplot(
    data=df,
    x="department",
    y="marks",
    hue="gender",
    palette="Set2"
)

plt.title("Marks Distribution by Department and Gender")
plt.show()


# ------------------------------------------------------------
# 5.3 HORIZONTAL BOXENPLOT
# ------------------------------------------------------------

sns.boxenplot(
    data=df,
    x="marks",
    y="department"
)

plt.title("Horizontal Boxen Plot")
plt.show()


# ------------------------------------------------------------
# 5.4 BOXENPLOT PARAMETERS
# ------------------------------------------------------------

# data:
# Input DataFrame.

# x and y:
# Categorical and numerical variables.

# hue:
# Additional category.

# order:
# Controls category order.

# palette:
# Controls colors.

# orient:
# Horizontal or vertical orientation.

# k_depth:
# Controls the depth of quantile levels.
# The default is generally suitable for normal use.

# width:
# Controls the width of the boxes.

# dodge:
# Separates groups when hue is used.

# showfliers:
# Controls whether extreme observations are displayed.

# ------------------------------------------------------------
# 5.4 BOXENPLOT PARAMETERS
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxenplot(
    data=df,
    x="department",        # Categorical variable
    y="marks",             # Numerical variable
    hue="gender",          # Additional category
    order=["CSE", "ECE", "EEE", "Mechanical"],  # Category order
    palette="Set2",        # Colors
    orient="v",            # Vertical orientation
    k_depth="tukey",       # Controls quantile depth
    width=0.8,             # Width of boxes
    dodge=True,            # Separates gender groups
    showfliers=True        # Displays extreme observations
)
plt.title("Marks Distribution Across Departments and Gender")
plt.xlabel("Department")
plt.ylabel("Marks")
plt.legend(title="Gender")
plt.show()


# ============================================================
# 6. POINTPLOT()
# ============================================================

# pointplot() displays an estimated numerical value for
# each category using points and uncertainty intervals.
#
# It is useful for:
# - Comparing group averages.
# - Understanding trends across categories.
# - Comparing multiple groups.
#
# By default, pointplot generally estimates the mean.
# It also displays an uncertainty interval by default.

sns.boxenplot(
    data=df,
    x="Category",           # Categorical variable
    y="Sales",              # Numerical variable
    hue="Region",           # Additional category
    order=["A", "B", "C"],  # Category order
    palette="Set2",         # Color palette
    orient="v",             # Vertical orientation ("h" = horizontal)
    k_depth="tukey",        # Controls quantile depth
    width=0.8,              # Width of boxes
    dodge=True,             # Separates hue groups
    showfliers=True         # Displays extreme observations
)

plt.title("Boxen Plot")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()

# ------------------------------------------------------------
# 6.1 BASIC POINTPLOT
# ------------------------------------------------------------

sns.pointplot(
    data=df,
    x="department",
    y="marks"
)
plt.title("Average Marks by Department")
plt.show()


# ------------------------------------------------------------
# 6.2 POINTPLOT WITH HUE
# ------------------------------------------------------------

sns.pointplot(
    data=df,
    x="department",
    y="marks",
    hue="gender",
    dodge=True, # seperates gender groups side by side
    markers=["o", "s"],
    linestyles=["-", "--"],
    palette="Set1"
)

plt.title("Average Marks by Department and Gender")
plt.show()


# ------------------------------------------------------------
# 6.3 CHANGE ESTIMATOR
# ------------------------------------------------------------

# By default, pointplot calculates the mean.
# We can use another estimator, such as median.
sns.pointplot(
    data=df,
    x="department",
    y="marks",
    estimator=np.median,
    errorbar=None
)
plt.title("Median Marks by Department")
plt.show()


# ------------------------------------------------------------
# 6.4 UNDERSTANDING ERROR BARS
# ------------------------------------------------------------

# errorbar controls the uncertainty interval.
# Common modern Seaborn options:

# errorbar="ci":
# Confidence interval.

# errorbar=("ci", 95):
# 95% confidence interval.

# errorbar="sd":
# Standard deviation.

# errorbar="se":
# Standard error.

# errorbar=None:
# No uncertainty interval.

# Important:
# Confidence intervals, standard deviation and standard error
# represent different statistical concepts.
# Do not interpret them as interchangeable.

sns.pointplot(
    data=df,
    x="department",
    y="marks",
    errorbar=("ci", 95)
)
plt.title("Average Marks with 95% Confidence Intervals")
plt.show()


# ============================================================
# 7. CATPLOT()
# ============================================================

# catplot() is a figure-level interface for categorical plots.

# It can create different categorical visualizations.

# Supported kinds include:

# kind="strip"      -> stripplot
# kind="swarm"      -> swarmplot
# kind="box"        -> boxplot
# kind="boxen"      -> boxenplot
# kind="violin"     -> violinplot
# kind="point"      -> pointplot
# kind="bar"        -> barplot
# kind="count"      -> countplot

# Main advantage:
# catplot() makes it easier to create multiple plots
# separated by categories using row and col.


# ------------------------------------------------------------
# 7.1 BASIC CATPLOT
# ------------------------------------------------------------

sns.catplot(
    data=df,
    x="department",
    y="marks",
    kind="strip"
)
plt.show()


# ------------------------------------------------------------
# 7.2 CATPLOT WITH KIND="BOX"
# ------------------------------------------------------------

sns.catplot(
    data=df,
    x="department",
    y="marks",
    kind="box"
)
plt.show()


# ------------------------------------------------------------
# 7.3 CATPLOT WITH KIND="BOXEN"
# ------------------------------------------------------------

sns.catplot(
    data=df,
    x="department",
    y="marks",
    kind="boxen"
)
plt.show()


# ------------------------------------------------------------
# 7.4 CATPLOT WITH KIND="POINT"
# ------------------------------------------------------------

sns.catplot(
    data=df,
    x="department",
    y="marks",
    hue="gender",
    kind="point"
)

plt.show()


# ------------------------------------------------------------
# 7.5 CATPLOT WITH KIND="BAR"
# ------------------------------------------------------------

sns.catplot(
    data=df,
    x="department",
    y="marks",
    hue="gender",
    kind="bar"
)

plt.show()


# ------------------------------------------------------------
# 7.6 CATPLOT WITH KIND="COUNT"
# ------------------------------------------------------------

sns.catplot(
    data=df,
    x="department",
    hue="gender",
    kind="count"
)

plt.show()


# ============================================================
# 8. FACETING WITH CATPLOT
# ============================================================

# Faceting means splitting a dataset into multiple smaller
# visualizations based on categorical variables.

# catplot() supports:

# col:
# Creates separate plots for each category.

# row:
# Creates separate rows for each category.

# col_wrap:
# Controls how many columns are displayed before wrapping.

# height:
# Controls the height of each individual plot.

# aspect:
# Controls width relative to height.

# ------------------------------------------------------------
# 8.1 SEPARATE PLOTS USING COL
# ------------------------------------------------------------
# Create separate boxplots for each gender using catplot
sns.catplot(
    data=df,               # DataFrame containing student data
    x="department",        # X-axis: Departments
    y="marks",             # Y-axis: Students' marks
    col="gender",          # Creates separate plots for Male and Female (horizontal)
    kind="box",            # Specifies that the plot type is a boxplot
    height=4,              # Sets the height of each plot to 4 inches
    aspect=1               # Sets width-to-height ratio to 1 (square plot)
)

plt.show()


# ------------------------------------------------------------
# 8.2 SEPARATE PLOTS USING ROW
# ------------------------------------------------------------

sns.catplot(
    data=df,
    x="department",
    y="marks",
    row="gender", # vertical 
    kind="strip",
    height=4,
    aspect=1.3
)

plt.show()


# ------------------------------------------------------------
# 8.3 USING ROW AND COL TOGETHER
# ------------------------------------------------------------

# Create a grid of boxplots based on gender and result
sns.catplot(
    data=df,               # DataFrame containing student data
    x="department",        # X-axis: Departments
    y="marks",             # Y-axis: Students' marks
    row="gender",          # Creates separate rows for Male and Female
    col="result",          # Creates separate columns for Pass and Fail
    kind="box",            # Specifies boxplot as the plot type
    height=3.5,            # Sets the height of each subplot to 3.5 inches
    aspect=1               # Sets width-to-height ratio to 1 (square plots)
)
plt.show()                # Displays the complete grid of plots


# ------------------------------------------------------------
# 8.4 USING COL_WRAP
# ------------------------------------------------------------

# Create separate stripplots for each study level
sns.catplot(
    data=df,               # DataFrame containing student data
    x="department",        # X-axis: Departments
    y="marks",             # Y-axis: Students' marks
    col="study_level",     # Creates separate plots for Low, Medium and High study levels
    col_wrap=2,            # Displays a maximum of 2 plots per row
    kind="strip",          # Creates stripplots to display individual data points
    height=4,              # Sets the height of each subplot to 4 inches
    aspect=1.2              # Sets width to 1.2 times the height
)

plt.show()                 # Displays all the plots


# ============================================================
# 9. CATEGORICAL ORDERING
# ============================================================

# By default, categories may be displayed in their existing categorical order or an inferred order.
# We can explicitly define the order.
# This is particularly important for ordinal categories.

# ------------------------------------------------------------
# 9.1 DEFINE CATEGORY ORDER
# ------------------------------------------------------------

# Define the order in which study levels should appear on the X-axis
study_order = ["Low", "Medium", "High"]

# Create a stripplot to visualize individual students' marks by study level
sns.stripplot(
    data=df,                  # DataFrame containing student data
    x="study_level",          # X-axis: Study levels (Low, Medium, High)
    y="marks",                # Y-axis: Students' marks
    order=study_order,        # Displays study levels in the specified order
    jitter=True               # Slightly spreads overlapping dots horizontally
)
# If High has more dots towards 80–100, students in that group tend to score higher.
# If Low has more dots towards 30–50, students in that group tend to score lower.
# If all three groups have similar distributions, study level might not show a strong relationship with marks.
plt.title("Marks by Study Level")  # Sets the graph title
plt.xlabel("Study Level")           # Sets the X-axis label
plt.ylabel("Marks")                 # Sets the Y-axis label 
plt.show()                          # Displays the graph

# ------------------------------------------------------------
# 9.2 ORDER DEPARTMENTS
# ------------------------------------------------------------

department_order = [
    "CSE",
    "ECE",
    "EEE",
    "Mechanical"
]

sns.boxplot(
    data=df,
    x="department",
    y="marks",
    order=department_order
)

plt.title("Marks by Department in Custom Order")
plt.show()


# ------------------------------------------------------------
# 9.3 ORDER CATEGORIES BY THEIR MEAN
# ------------------------------------------------------------

# Calculate average marks for each department.

department_means = (
    df.groupby(
        "department",          # Groups students based on their department
        observed=False         # Includes all department categories, even if unused
    )["marks"]                 # Selects the marks column
    .mean()                    # Calculates the average marks for each department
    .sort_values()              # Sorts average marks in ascending order (lowest to highest)
)

print(department_means)         # Displays the average marks of each department

# Extract categories in ascending order of mean marks.
 
sorted_departments = department_means.index.tolist()  
# .index → Extracts labels, .tolist() → Converts them into a Python list.
print("\nAVERAGE MARKS BY DEPARTMENT")
print(department_means)

sns.barplot(
    data=df,
    x="department",
    y="marks",
    order=sorted_departments
)

plt.title("Departments Ordered by Average Marks")
plt.show()


# ============================================================
# 10. UNDERSTANDING HUE, DODGE AND JITTER
# ============================================================

# hue:
# Separates observations using colors.

# dodge:
# Separates hue groups instead of placing them at
 the same category position.
#
# jitter:
# Spreads points horizontally to reduce overlap.

# These parameters are especially important when
# comparing two or more categorical groups.


# ------------------------------------------------------------
# 10.1 HUE WITHOUT DODGE
# ------------------------------------------------------------

sns.stripplot(
    data=df,
    x="department",
    y="marks",
    hue="gender",
    dodge=False,
    jitter=True
)

plt.title("Hue Without Dodge")
plt.show()


# ------------------------------------------------------------
# 10.2 HUE WITH DODGE
# ------------------------------------------------------------

# Create a stripplot to compare students' marks across departments and gender
sns.stripplot(
    data=df,               # DataFrame containing student data
    x="department",        # X-axis: Departments
    y="marks",             # Y-axis: Students' marks
    hue="gender",          # Uses different colors to distinguish Male and Female students
    dodge=True,            # Separates Male and Female points side by side within each department
    jitter=True            # Slightly spreads overlapping points horizontally for better visibility
)
plt.title("Marks Distribution by Department and Gender")
plt.xlabel("Department")
plt.ylabel("Marks")
plt.legend(title="Gender")
plt.show()                 # Displays the stripplot



# ------------------------------------------------------------
# 10.3 DODGE IN BOX PLOTS
# ------------------------------------------------------------

sns.boxplot(
    data=df,
    x="department",
    y="marks",
    hue="gender",
    dodge=True
)

plt.title("Grouped Box Plots")
plt.show()


# ============================================================
# 11. UNDERSTANDING NATIVE SCALE
# ============================================================

# Categorical plots normally treat categories as equally spaced positions.
# In some situations, the category variable may contain  numerical or datetime values with meaningful spacing.
# native_scale=True allows Seaborn to preserve the original spacing of such values in supported plots.

# Example:
# Measurements taken at different days: Day 1, Day 2, Day 10.
# With native_scale=False, category positions are equally spaced.
# With native_scale=True, numerical spacing is preserved.


measurement_df = pd.DataFrame({
    "day": [1, 1, 2, 2, 10, 10, 20, 20],
    "measurement": [10, 12, 15, 14, 20, 22, 25, 27]
})

# Preserve numerical spacing where supported.

sns.stripplot(
    data=measurement_df,
    x="day",
    y="measurement",
    native_scale=True, # In Seaborn, native_scale=True tells the plot to preserve the original numerical or datetime values ,
    # on the categorical axis instead of treating them as equally spaced categories.
    # By default, Seaborn treats the X-axis values as categorical positions.
    # For example, days 1, 2, 5 and 10 would be placed at equally spaced positions, even though the actual gaps between days are different.
    # but in case of native_scale=True : 
    # Now, the X-axis preserves the actual numerical spacing.
    # For example:- 
    # Day 1 to Day 2 = 1 unit apart.
    # Day 2 to Day 5 = 3 units apart.
    # Day 5 to Day 10 = 5 units apart.
    jitter=True
)
plt.title("Strip Plot with Native Numerical Scale")
plt.show()


# NOTE:
# native_scale is available in modern Seaborn versions.
# It is not necessary for ordinary categorical EDA.


# ============================================================
# 12. FIGURE-LEVEL VS AXES-LEVEL FUNCTIONS
# ============================================================

# Axes-level categorical functions:

# sns.stripplot()
# sns.swarmplot()
# sns.boxenplot()
# sns.pointplot()

# They draw on a Matplotlib Axes.

# You can pass: ax=ax
# Figure-level function: sns.catplot()
# It creates and manages its own figure and axes.
# Do not pass ax=ax to catplot().


# ------------------------------------------------------------
# 12.1 AXES-LEVEL FUNCTION WITH SUBPLOTS
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9)
)

sns.stripplot(
    data=df,
    x="department",
    y="marks",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Strip Plot")
sns.swarmplot(
    data=df.sample(100, random_state=42),
    x="department",
    y="marks",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Swarm Plot")
sns.boxenplot(
    data=df,
    x="department",
    y="marks",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Boxen Plot")
sns.pointplot(
    data=df,
    x="department",
    y="marks",
    errorbar=None,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Point Plot")
plt.tight_layout()
plt.show()


# ============================================================
# 13. CATEGORICAL VISUALIZATION FOR EDA
# ============================================================

# EDA = Exploratory Data Analysis.
# Categorical visualizations help answer:

# 1. Which category has higher numerical values?
# 2. Which category has greater variability?
# 3. Are there unusual observations?
# 4. Are categories distributed equally?
# 5. Does a numerical feature behave differently  across different groups?

# We will create a small EDA dashboard.


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)
# Distribution of marks by department
sns.boxenplot(
    data=df,
    x="department",
    y="marks",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Marks Distribution by Department")


# Individual observations by study level
sns.stripplot(
    data=df,
    x="study_level",
    y="marks",
    order=study_order,
    jitter=True,
    alpha=0.6,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Marks by Study Level")

# Compare genders
sns.pointplot(
    data=df,
    x="department",
    y="marks",
    hue="gender",
    errorbar=None,
    ax=axes[1, 0]
)
axes[1, 0].set_title("Average Marks by Department and Gender")

# Category counts
sns.countplot(
    data=df,
    x="result",
    hue="gender",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Result Counts by Gender")

plt.tight_layout()
plt.show()


# ============================================================
# 14. ML APPLICATION: FEATURE DISTRIBUTION BY TARGET CLASS
# ============================================================

# This is one of the most important applications
# of categorical visualization for ML preparation.

# Suppose, Features:study_hours, attendance, age
# Target: result

# We want to understand how each numerical feature behaves across the target classes.
# This can help identify whether a feature may be useful for predicting the target.

# IMPORTANT:
# A visible difference does not automatically mean that a feature is useful or causal.
# Actual feature usefulness must be evaluated with appropriate ML validation.


# ------------------------------------------------------------
# 14.1 STUDY HOURS BY RESULT
# ------------------------------------------------------------

sns.boxplot(
    data=df,
    x="result",
    y="study_hours"
)
plt.title("Study Hours by Result Class")
plt.show()


# ------------------------------------------------------------
# 14.2 ATTENDANCE BY RESULT
# ------------------------------------------------------------

sns.violinplot(
    data=df,
    x="result",
    y="attendance",
    inner="quart",
    cut=0
)
plt.title("Attendance Distribution by Result")
plt.show()


# ------------------------------------------------------------
# 14.3 MARKS BY RESULT
# ------------------------------------------------------------

sns.stripplot(
    data=df,
    x="result",
    y="marks",
    jitter=True,
    alpha=0.5
)
plt.title("Marks by Result Class")
plt.show()


# ------------------------------------------------------------
# 14.4 COMPARE MULTIPLE FEATURES BY CLASS
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5)
)
sns.boxplot(
    data=df,
    x="result",
    y="study_hours",
    ax=axes[0]
)
axes[0].set_title("Study Hours by Result")

sns.boxplot(
    data=df,
    x="result",
    y="attendance",
    ax=axes[1]
)
axes[1].set_title("Attendance by Result")

sns.boxplot(
    data=df,
    x="result",
    y="age",
    ax=axes[2]
)
axes[2].set_title("Age by Result")      
plt.tight_layout()
plt.show()


# ============================================================
# 15. ML APPLICATION: CLASS IMBALANCE
# ============================================================

# In classification problems, the target may contain different numbers of observations for each class.

# Example:
# Class 0 -> 900 observations
# Class 1 -> 100 observations
# This is called class imbalance.
# A count plot helps us inspect the class distribution.


# Create an example imbalanced classification dataset.

class_df = pd.DataFrame({
    "target": (
        ["Normal"] * 900 +
        ["Fraud"] * 100
    )
})

sns.countplot(
    data=class_df,
    x="target",
    order=["Normal", "Fraud"]
)
plt.title("Class Distribution")
plt.xlabel("Target Class")
plt.ylabel("Number of Observations")
plt.show()


# Calculate class proportions.

class_distribution = (
    class_df["target"]
    .value_counts(normalize=True)
    * 100
)

print("\nCLASS DISTRIBUTION (%)")
print(class_distribution)


# IMPORTANT ML CONCEPT:

# Class imbalance may affect:
# - Model training.
# - Prediction performance.
# - Recall for minority classes.
# - Precision and F1-score.

# Possible approaches:
# - Stratified train-test split.
# - Class weights.
# - Oversampling.
# - Undersampling.
# - Appropriate evaluation metrics.
#
# Do not automatically balance every dataset.
# First understand the problem and evaluation objective.


# ============================================================
# 16. ML APPLICATION: FEATURE COMPARISON BY CATEGORY
# ============================================================

# Example:
# Compare numerical features across different categories in a dataset.
# This is useful before selecting features for a classification or regression model.


fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5)
)
sns.boxenplot(
    data=df,
    x="department",
    y="attendance",
    ax=axes[0]
)
axes[0].set_title("Attendance by Department")

sns.pointplot(
    data=df,
    x="department",
    y="study_hours",
    errorbar=None,
    ax=axes[1]
)
axes[1].set_title("Average Study Hours by Department")

plt.tight_layout()
plt.show()


# ============================================================
# 17. CUSTOMIZATION AND PRESENTATION
# ============================================================

# Categorical plots can be customized using:
#
# Matplotlib methods:
# set_title()
# set_xlabel()
# set_ylabel()
# tick_params()
# set_ylim()
# legend()

# Seaborn parameters:
# palette
# hue
# dodge
# order
# alpha
# size
# width


fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="department",
    y="marks",
    hue="gender",
    palette="Set2",
    width=0.7,
    ax=ax
)

ax.set_title(
    "Student Marks by Department and Gender",
    fontsize=14
)

ax.set_xlabel(
    "Department",
    fontsize=11
)

ax.set_ylabel(
    "Marks",
    fontsize=11
)

ax.tick_params(
    axis="both",
    labelsize=10
)

ax.set_ylim(0, 105)

ax.legend(
    title="Gender"
)

plt.tight_layout()
plt.show()


# ============================================================
# 18. SAVING CATEGORICAL VISUALIZATIONS
# ============================================================

# Save the chart before or after plt.show().

# Common formats:
# PNG
# JPG
# PDF
# SVG
# dpi controls image resolution.


fig, ax = plt.subplots(figsize=(9, 5))
sns.boxenplot(
    data=df,
    x="department",
    y="marks",
    ax=ax
)

ax.set_title("Marks Distribution by Department")
plt.tight_layout()

# Save the image in the current working directory.

fig.savefig(
    "categorical_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 19. IMPORTANT COMPARISON OF CATEGORICAL PLOTS
# ============================================================

# stripplot():
# Shows individual observations with jitter.

# swarmplot():
# Shows individual observations while trying to avoid overlap.

# boxenplot():
# Shows detailed quantile information.

# pointplot():
# Shows an estimated statistic and uncertainty interval.

# catplot():
# Figure-level interface that can create several
# categorical plot types and faceted layouts.

#
# WHEN TO USE WHICH?

# Inividual data points:
# stripplot or swarmplot.

# Large dataset distribution:
# boxenplot.

# Compare estimated averages:
# pointplot.

# Multiple categorical plot layouts:
# catplot.

# Category frequency:
# countplot (also covered in Phase 1).

# Numerical distribution by category:
# boxplot, violinplot or boxenplot.


# ============================================================
# 20. COMMON MISTAKES TO AVOID
# ============================================================

# 1. Using swarmplot on extremely large datasets.
#    It can become slow and visually crowded.

# 2. Forgetting hue.
#    You may miss important differences between groups.

# 3. Not specifying category order.
#    Ordinal variables should generally follow their
#    meaningful order.

# 4. Treating mean and median as the same.
#    They are different measures of central tendency.

# 5. Assuming different group averages prove causation.
#    Group differences do not establish causal relationships.

# 6. Ignoring sample sizes.
#    A group with very few observations may have an unstable
#    estimated mean or distribution.

# 7. Misinterpreting uncertainty intervals.
#    Understand whether an interval represents CI, SD or SE.

# 8. Using too many categories in one plot.
#    The visualization may become difficult to read.

# 9. Confusing catplot() with Axes-level functions.
#    catplot() manages its own figure and axes.

# 10. Using categorical plots as a substitute for statistical
#     testing or model validation.
#     Visualizations support analysis but do not prove a result.


# ============================================================
# 21. PRACTICE TASKS
# ============================================================

# Complete these exercises independently.

# TASK 1:
# Create a stripplot of attendance by department.

# TASK 2:
# Add gender as hue to the stripplot.

# TASK 3:
# Compare stripplot and swarmplot on the same sample.

# TASK 4:
# Create a boxenplot of marks by study_level.

# TASK 5:
# Use hue to compare marks by gender and department.

# TASK 6:
# Create a pointplot of average attendance by department.

# TASK 7:
# Change the pointplot estimator from mean to median.

# TASK 8:
# Create a catplot with kind="box".

# TASK 9:
# Create separate plots for each gender using col.

# TASK 10:
# Use row and col together in catplot.

# TASK 11:
# Define a custom category order.

# TASK 12:
# Sort departments based on average marks.

# TASK 13:
# Create a class distribution countplot.

# TASK 14:
# Compare study_hours and attendance by result.
#
# TASK 15:
# Build a 2 x 2 categorical EDA dashboard.

# TASK 16:
# Explain which plot is appropriate for:
# - Individual observations.
# - Large distribution comparisons.
# - Average comparisons.
# - Class-wise feature analysis.


# ============================================================
# 22. FINAL PHASE 3 REVISION
# ============================================================

# CONCEPTS YOU SHOULD UNDERSTAND:
# [ ] Categorical variables: nominal and ordinal.
# [ ] stripplot() and its jitter parameter.
# [ ] swarmplot() and point overlap.
# [ ] boxenplot() and quantile distributions.
# [ ] pointplot() and estimators.
# [ ] Confidence intervals and errorbar.
# [ ] catplot() and its supported kinds.
# [ ] Faceting using row, col and col_wrap.
# [ ] Category ordering using order.
# [ ] Grouping using hue.
# [ ] Separating groups using dodge.
# [ ] native_scale.
# [ ] Figure-level vs Axes-level functions.
# [ ] Categorical EDA.
# [ ] Class-wise feature comparisons.
# [ ] Class imbalance
# [ ] Interpretation of group differences.
# [ ] Common mistakes and plotting best practices.

# PHASE 3 COMPLETION GOAL:

# You should be able to take a DataFrame containing
# categorical and numerical variables and choose
# appropriate categorical plots to understand:

# 1. Individual observations.
# 2. Distribution across categories.
# 3. Group differences.
# 4. Outliers and variability.
# 5. Feature behavior across target classes.
# 6. Class imbalance.

# ============================================================
# END OF SEABORN PHASE 3
# ============================================================