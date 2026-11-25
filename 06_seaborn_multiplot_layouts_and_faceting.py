# ============================================================
# PHASE 6: SEABORN MULTI-PLOT LAYOUTS AND FACETING
# ============================================================

# OBJECTIVE:
# Learn how to create multiple related visualizations together using Seaborn's figure-level functions and FacetGrid.

# This phase focuses on:
# 1. Figure-level vs axes-level functions
# 2. FacetGrid
# 3. relplot()
# 4. displot()
# 5. catplot()
# 6. lmplot() overview
# 7. Multi-faceted EDA layouts
# 8. Choosing the right layout for ML and data analysis

# PREVIOUSLY COMPLETED:
# - Seaborn introduction and setup
# - Scatterplot, lineplot, barplot, countplot
# - Histplot, kdeplot, boxplot, violinplot, ecdfplot
# - Stripplot, swarmplot, boxenplot, pointplot, catplot basics
# - Pairplot, jointplot, regplot, lmplot basics, heatmap
# - Seaborn customization, themes, palettes, labels, legends

# We will not repeat those topics in detail.
# This phase focuses on combining visualizations effectively.


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 2. CREATE A SAMPLE DATASET
# ------------------------------------------------------------

# We will use a student performance dataset, Each row represents one student's performance.

# This dataset contains:
# - department
# - gender
# - study_hours
# - attendance
# - assignment_score
# - previous_marks
# - marks
# - result

# We will use the same dataset throughout this phase.

np.random.seed(42)
number_of_students = 300
departments = np.random.choice(
    ["CSE", "ECE", "MECH"],
    number_of_students
)
gender = np.random.choice(
    ["Male", "Female"],
    number_of_students
)
study_hours = np.random.uniform(
    1,
    10,
    number_of_students
)
attendance = np.random.uniform(
    50,
    100,
    number_of_students
)
assignment_score = np.random.uniform(
    40,
    100,
    number_of_students
)
previous_marks = np.random.uniform(
    35,
    95,
    number_of_students
)
marks = (
    15
    + study_hours * 4
    + attendance * 0.15
    + assignment_score * 0.20
    + previous_marks * 0.15
    + np.random.normal(0, 5, number_of_students)
)
marks = np.clip(marks, 0, 100)
result = np.where(
    marks >= 50,
    "Pass",
    "Fail"
)
df = pd.DataFrame({
    "department": departments,
    "gender": gender,
    "study_hours": study_hours,
    "attendance": attendance,
    "assignment_score": assignment_score,
    "previous_marks": previous_marks,
    "marks": marks,
    "result": result
})
print(df.head())


# ============================================================
# 3. FIGURE-LEVEL VS AXES-LEVEL FUNCTIONS
# ============================================================

# This is one of the MOST IMPORTANT concepts in this phase.
# Seaborn plotting functions can broadly be divided into:
# 1. Axes-level functions
# 2. Figure-level functions


# ------------------------------------------------------------
# 3.1 AXES-LEVEL FUNCTIONS
# ------------------------------------------------------------

# Axes-level functions draw a plot inside an existing Matplotlib Axes object.

# Examples: sns.scatterplot(), sns.lineplot(), sns.barplot(), sns.countplot(), sns.histplot(), sns.kdeplot(), sns.boxplot()
#  sns.violinplot(), sns.stripplot(), sns.swarmplot(), sns.heatmap(), sns.regplot()

# These functions commonly accept:
# ax=   => This allows us to decide exactly where each plot appears.

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)
sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="result",
    ax=axes[0]
)
axes[0].set_title("Study Hours vs Marks")
sns.boxplot(
    data=df,
    x="department",
    y="marks",
    ax=axes[1]
)
axes[1].set_title("Marks by Department")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 3.2 FIGURE-LEVEL FUNCTIONS
# ------------------------------------------------------------

# Figure-level functions manage their own complete figure.
# They can automatically create multiple subplots based on categories in your dataset.

# Examples: sns.relplot(), sns.displot(), sns.catplot(), sns.lmplot(), sns.pairplot(), sns.jointplot()
# These functions are especially useful for FACETING.

# Important:
# Figure-level functions generally do not accept ax= in the same way as axes-level functions.
#  They create and manage their own figure and axes.

# Example:

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="result",
    kind="scatter"
)
plt.show()


# ------------------------------------------------------------
# 3.3 IMPORTANT DIFFERENCES
# ------------------------------------------------------------

# AXES-LEVEL:
# - Draws inside an existing Axes.
# - Usually accepts ax=.
# - Useful when manually arranging subplots.
# - Gives direct control over individual Axes.

# FIGURE-LEVEL:
# - Creates and manages its own figure.
# - Can automatically create multiple facets.
# - Usually returns a grid object.
# - Useful for comparing categories across multiple panels.

# RULE:
# If you want to manually arrange plots:
# Use Matplotlib subplots + axes-level Seaborn functions.
# If you want automatic category-based panels:
# Use Seaborn figure-level functions.


# ============================================================
# 4. FACETING: THE CORE CONCEPT
# ============================================================

# Faceting means dividing one dataset into multiple smaller
# subsets and displaying each subset in a separate plot.

# Example:
# Instead of displaying all departments in one scatterplot,
# create separate scatterplots for: CSE, ECE, MECH
# This makes comparisons easier.

# Common faceting dimensions:
# col = create separate plots horizontally
# row = create separate plots vertically
# Example: col="department"
# Creates one column/panel for each department => row="gender" - Creates one row for each gender.


# ------------------------------------------------------------
# 4.1 BASIC FACETING USING RELPLOT
# ------------------------------------------------------------

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    kind="scatter"
)
plt.show()

# Three panels are created:
# CSE | ECE | MECH
# Each panel displays only the observations belonging
# to that department.


# ------------------------------------------------------------
# 4.2 FACETING USING ROW
# ------------------------------------------------------------

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    row="gender",
    kind="scatter"
)
plt.show()

# Creates separate rows:
# Male
# Female


# ------------------------------------------------------------
# 4.3 FACETING USING BOTH ROW AND COL
# ------------------------------------------------------------

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    row="gender",
    col="department",
    kind="scatter"
)
plt.show()

# Creates a grid:
#
#             CSE       ECE       MECH
#
# Male        Plot      Plot      Plot
# Female      Plot      Plot      Plot

# This allows us to compare departments and genders simultaneously.


# ------------------------------------------------------------
# 4.4 ADD HUE TO FACETING
# ------------------------------------------------------------

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    hue="result",
    kind="scatter"
)
plt.show()

# col separates departments into panels, hue distinguishes Pass and Fail using colors.
# Faceting and hue can be used together.
# Use faceting when you want separate panels.
# Use hue when you want multiple groups in the same panel.


# ------------------------------------------------------------
# 4.5 FACET ORDER
# ------------------------------------------------------------

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    col_order=["MECH", "CSE", "ECE"],
    kind="scatter"
)
plt.show()

# col_order controls the order of the panels.
# Similarly:
# row_order controls the order of row categories.
# hue_order controls the order of hue categories.


# ------------------------------------------------------------
# 4.6 COL_WRAP
# ------------------------------------------------------------

# col_wrap controls how many facet columns appear in each row.
# It is useful when there are many categories.

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    col_wrap=2,
    kind="scatter"
)
plt.show()

# col_wrap=2 means: Display at most two facet columns in each row.

# Important:
# col_wrap is used with col faceting.
# It cannot be combined with row faceting.


# ============================================================
# 5. FACETGRID: MANUAL FACETING CONTROL
# ============================================================

# FacetGrid is a Seaborn class used to create a grid of subplots based on categories in a dataset.
# It provides more manual control than relplot(), displot() and catplot().

# Basic syntax:
# grid = sns.FacetGrid(
#     data,
#     col="category",
#     row="another_category"
# )
#
# Then use:
# grid.map(), grid.map_dataframe() to apply plotting functions to each subset.


# ------------------------------------------------------------
# 5.1 BASIC FACETGRID
# ------------------------------------------------------------

grid = sns.FacetGrid(
    data=df,                  # DataFrame containing the data
    col="department"          # Create a separate plot for each department
)
grid.map(
    plt.hist,                 # Use Matplotlib's histogram function
    "marks"                   # Plot the distribution of marks
)
plt.show()                    # Display all histograms

# What happens?

# 1. FacetGrid divides the DataFrame by department.
# 2. It creates one panel for each department.
# 3. grid.map() applies plt.hist() to each subset.

# Each panel displays the marks distribution of one department.


# ------------------------------------------------------------
# 5.2 FACETGRID WITH ROW AND COL
# ------------------------------------------------------------

grid = sns.FacetGrid(
    data=df,                  # DataFrame containing the data
    row="gender",              # Create separate rows for each gender
    col="department"           # Create separate columns for each department
)
grid.map(
    plt.hist,                  # Use histogram to visualize the data
    "marks"                    # Show the distribution of marks
)
plt.show()                     # Display all histograms


# ------------------------------------------------------------
# 5.3 IMPORTANT FACETGRID PARAMETERS
# ------------------------------------------------------------

grid = sns.FacetGrid(
    data=df,
    # Separate panels horizontally.
    col="department",
    # Separate panels vertically.
    row="gender",
    # Control category order.
    col_order=["CSE", "ECE", "MECH"],
    # Control row order.
    row_order=["Male", "Female"],
    # Width and height of each panel.
    height=3,
    # Width / height ratio.
    aspect=1.2,
    # Share x-axis scale across panels.
    sharex=True,
    # Share y-axis scale across panels.
    sharey=True,
    # Position of legend.
    legend_out=True
)
grid.map(
    plt.hist,
    "marks"
)
plt.show()


# PARAMETER EXPLANATIONS:

# data: DataFrame containing the observations.
# col: Creates separate columns for category values.
# row: Creates separate rows for category values.
# col_order: Controls column category order.
# row_order: Controls row category order.
# height: Height of each individual facet in inches.
# aspect: Width divided by height.
# sharex: Shares x-axis limits across panels.
# sharey: Shares y-axis limits across panels.
# legend_out: Places the legend outside the grid when a legend exists.
# hue: Can be used to divide observations by color.
# palette: Controls colors for hue categories.
# margin_titles: Places row titles near the outer margin.


# ------------------------------------------------------------
# 5.4 MAP() VS MAP_DATAFRAME()
# ------------------------------------------------------------

# Both methods apply a plotting function to each facet, But they pass data differently.


# MAP()
# grid.map(function, "column_name")

# It passes the selected column values to the plotting function as positional arguments.

# Example:

grid = sns.FacetGrid(
    data=df,
    col="department"
)
grid.map(
    plt.hist,
    "marks",
    bins=15,
    edgecolor="black"
)
plt.show()


# MAP_DATAFRAME()
# grid.map_dataframe(function, ...)

# It passes the subset DataFrame using the data= argument.
# This is especially useful with Seaborn plotting functions.

# Example:

grid = sns.FacetGrid(
    data=df,                  # DataFrame containing the data
    col="department"          # Create a separate plot for each department
)
grid.map_dataframe(
    sns.histplot,             # Use Seaborn's histogram function
    x="marks",                # Column to display on the X-axis
    bins=15                   # Divide marks into 15 intervals
)
plt.show()                    # Display all histograms

# IMPORTANT DIFFERENCE:

# map(): Passes selected column values to the plotting function.
# map_dataframe(): Passes the current subset using data=.
# Use map_dataframe() when your plotting function expects a DataFrame through its data parameter.


# ------------------------------------------------------------
# 5.5 FACETGRID WITH HUE
# ------------------------------------------------------------

grid = sns.FacetGrid(
    data=df,                  # DataFrame containing the data
    col="department",         # Create a separate plot for each department
    hue="result",             # Assign different colors based on student results
    palette="Set1"            # Choose the color palette
)
grid.map_dataframe(
    sns.scatterplot,          # Use Seaborn's scatter plot
    x="study_hours",          # Study hours on the X-axis
    y="marks"                 # Marks on the Y-axis
)
grid.add_legend()             # Display the legend for different result categories
plt.show()                    # Display all plots

# hue separates Pass and Fail using colors.
# add_legend() displays the legend for the hue categories.


# ------------------------------------------------------------
# 5.6 FACETGRID TITLES AND LABELS
# ------------------------------------------------------------

grid = sns.FacetGrid(
    data=df,
    col="department",
    height=4
)
grid.map_dataframe(
    sns.scatterplot,
    x="study_hours",
    y="marks"
)
# Change x-axis and y-axis labels.
grid.set_axis_labels(
    "Study Hours",
    "Marks"
)

# Change facet titles.
grid.set_titles(
    col_template="Department: {col_name}"
)
# Add an overall title.
grid.figure.suptitle(
    "Student Performance by Department",
    y=1.05
)
plt.show()

# set_axis_labels(): Sets labels for the facets.

# set_titles(): Customizes facet titles.
# grid.figure: Provides access to the underlying Matplotlib Figure.


# ------------------------------------------------------------
# 5.7 FACETGRID WITH MULTIPLE CUSTOMIZATIONS
# ------------------------------------------------------------

grid = sns.FacetGrid(
    data=df,
    col="department",
    hue="result",
    col_wrap=2,
    height=4,
    aspect=1.2,
    palette="Set1",
    sharex=True,
    sharey=True
)
grid.map_dataframe(
    sns.scatterplot,
    x="study_hours",
    y="marks",
    alpha=0.7
)
grid.set_axis_labels(
    "Study Hours",
    "Marks"
)
grid.set_titles(
    "Department: {col_name}"
)
grid.add_legend()
grid.figure.suptitle(
    "Student Performance Comparison",
    y=1.04
)
plt.show()


# ============================================================
# 6. RELPLOT(): FIGURE-LEVEL RELATIONAL PLOTS
# ============================================================

# relplot() is a figure-level function.
# It combines relational plots with automatic faceting.

# It supports: kind="scatter", kind="line"
#
# It can use: hue, style, size, row, col, col_wrap.
# Basic syntax:

sns.relplot(
    data=df,                  # DataFrame containing the data
    x="study_hours",          # Display study hours on the X-axis
    y="marks",                # Display marks on the Y-axis
    kind="scatter"            # Create a scatter plot
)
plt.suptitle("RELPLOT Figure level")
plt.show()                    # Display the plot


# ------------------------------------------------------------
# 6.1 RELPLOT WITH COL
# ------------------------------------------------------------

sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    kind="scatter"
)
plt.suptitle("RELPLOT WITH Col")
plt.show()


# ------------------------------------------------------------
# 6.2 RELPLOT WITH HUE, STYLE AND SIZE
# ------------------------------------------------------------

sns.relplot(
    data=df,                  # DataFrame containing the data
    x="study_hours",          # X-axis: Study hours
    y="marks",                # Y-axis: Marks
    col="department",         # Create a separate plot for each department
    hue="result",             # Different colors for different results
    style="gender",           # Different marker shapes for each gender
    size="attendance",        # Different dot sizes based on attendance
    sizes=(30, 150),          # Minimum and maximum dot sizes
    alpha=0.7,                # Set dot transparency
    kind="scatter"            # Create scatter plots
)
plt.suptitle("RELPLOT WITH HUE, STYLE AND SIZE")
plt.show()                    # Display the plots

# hue: Uses colors to represent result.
# style: Uses different marker styles for gender.
# size: Maps attendance to marker size.
# sizes: Controls the minimum and maximum marker sizes.
# alpha: Controls transparency.


# ------------------------------------------------------------
# 6.3 RELPLOT WITH LINE
# ------------------------------------------------------------

# Create separate line plots for each department and compare study hours with marks by gender.

sns.relplot(
    data=df,                  # Dataset containing student information
    x="study_hours",          # X-axis: Study hours
    y="marks",                 # Y-axis: Marks
    col="department",         # Create a separate graph for each department
    hue="gender",              # Use different line colors for each gender
    kind="line"                # Display the relationship using line plots
)
plt.suptitle("relplot with line")
plt.show()                    # Display all department-wise line plots

# kind="line" creates line plots instead of scatterplots.
# Remember:
# Line plots are useful when x represents an ordered variable such as time, sequence or measurement levels.
# Do not use a line merely because you have two numerical columns. A scatterplot may be more appropriate.


# ------------------------------------------------------------
# 6.4 RELPLOT IMPORTANT PARAMETERS
# ------------------------------------------------------------

# Commonly used parameters:
# data: Input DataFrame.
# x, y: Variables mapped to the axes.
# kind: "scatter" or "line".
# hue: Color grouping.
# style: Marker or line style grouping.
# size: Numerical or categorical marker-size grouping.
# col: Create separate columns.
# row: Create separate rows.
# col_wrap: Maximum number of facet columns per row.
# height: Height of each facet.
# aspect: Width / height ratio of each facet.
# palette: Color palette.
# legend: Controls legend display.
# markers: Controls marker styles in supported relational plots.
# facet_kws: Dictionary of additional FacetGrid parameters.

# Example:

# Create separate scatter plots for each department to compare study hours and marks.

sns.relplot(
    data=df,                  # Dataset containing student information
    x="study_hours",          # X-axis: Study hours
    y="marks",                 # Y-axis: Marks
    col="department",         # Create a separate graph for each department
    hue="result",              # Use different colors to distinguish Pass and Fail
    kind="scatter",            # Display individual students as scatter points
    height=4,                  # Set the height of each individual graph to 4 inches
    aspect=1.1,                 # Set graph width relative to its height (4 × 1.1 = 4.4 inches)
    facet_kws={                 # Customize the settings of individual subplot axes
        "sharex": True,         # Share the same X-axis scale across all department graphs
        "sharey": True          # Share the same Y-axis scale across all department graphs
    }
)
plt.suptitle("using relplot")
plt.show()                     # Display all department-wise scatter plots

# facet_kws passes additional settings to the underlying FacetGrid.


# ============================================================
# 7. DISPLOT(): FIGURE-LEVEL DISTRIBUTION PLOTS
# ============================================================

# displot() is a figure-level distribution plotting function.
# It supports:
# kind="hist"
# kind="kde"
# kind="ecdf"

# It is useful when you want to compare distributions
# across categories using separate panels.


# ------------------------------------------------------------
# 7.1 BASIC DISPLOT
# ------------------------------------------------------------

# Create a histogram to visualize the distribution of student marks.

sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    kind="hist"                # Display the distribution using a histogram
)
plt.suptitle("basic displot")
plt.show()                    # Display the histogram


# ------------------------------------------------------------
# 7.2 DISTRIBUTION BY DEPARTMENT
# ------------------------------------------------------------

sns.displot(
    data=df,
    x="marks",
    col="department",
    kind="hist"
)
plt.suptitle("distribution by department")
plt.show()


# Creates one histogram for each department.


# ------------------------------------------------------------
# 7.3 DISPLOT WITH HUE
# ------------------------------------------------------------

# Create a histogram to compare the distribution of marks between genders.
sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    hue="gender",              # Use different colors to distinguish genders
    kind="hist"                # Display the distribution using overlapping histograms
)
plt.suptitle("Displot with hue")
plt.show()                    # Display the histogram

# hue compares gender distributions in the same panel.


# ------------------------------------------------------------
# 7.4 MULTIPLE PARAMETER
# ------------------------------------------------------------

# multiple controls how multiple distributions are displayed.

# Common options:
# "layer": Draw distributions on top of each other.
# "dodge": Place histogram bars beside each other.
# "stack": Stack histogram counts.
# "fill": Normalize stacked bars so each bin represents proportions.


# Example: Layer
# Create overlapping histograms to compare marks distributions between genders.
sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    hue="gender",              # Use different colors to distinguish genders
    kind="hist",                # Display the data as histograms
    multiple="layer",           # Draw histograms on top of each other for comparison
    alpha=0.5                   # Set 50% opacity so overlapping histograms remain visible
)
plt.suptitle("Layered Histogram of Marks by Gender")
plt.show()                     # Display the overlapping histograms


# Example: Dodge
# Create side-by-side histograms to compare marks distributions between genders.
sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    hue="gender",              # Use different colors to distinguish genders
    kind="hist",                # Display the data as histograms
    multiple="dodge"            # Place histograms side by side instead of overlapping
)
plt.suptitle("Dodge Histogram of Marks by Gender")
plt.show()                     # Display the side-by-side histograms


# Example: Stack
# Create a stacked histogram to compare the combined marks distribution of different genders.
sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    hue="gender",              # Use different colors to distinguish genders
    kind="hist",                # Display the data as histograms
    multiple="stack"            # Stack gender-wise histograms on top of each other
)
plt.suptitle("Stacked Histogram of Marks by Gender")  # Add the overall graph title
plt.show()                     # Display the stacked histogram


# Example: Fill

# Create a 100% stacked histogram to compare the proportion of marks between genders.
sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    hue="gender",              # Use different colors to distinguish genders
    kind="hist",                # Display the data as histograms
    multiple="fill"             # Normalize each bin to 100% and show proportions
)
plt.suptitle("Proportional Distribution of Marks by Gender")  # Add the overall graph title
plt.show()                     # Display the proportional histogram

# Use fill when you want to compare proportions in bins.
# Do not interpret it as the original observation count.


# ------------------------------------------------------------
# 7.5 DISPLOT WITH KDE
# ------------------------------------------------------------

# Create separate KDE plots to compare the distribution of marks across departments.
sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    col="department",         # Create a separate KDE plot for each department
    kind="kde",                # Display the distribution using Kernel Density Estimation
    fill=True                  # Fill the area under each KDE curve with color
)
plt.suptitle("Distribution of Marks Across Departments")  # Add the overall graph title
plt.show()                     # Display all department-wise KDE plots

# Creates a separate KDE distribution for each department.


# ------------------------------------------------------------
# 7.6 DISPLOT WITH ECDF
# ------------------------------------------------------------

# Create separate ECDF plots to compare the cumulative distribution of marks across departments.

sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    col="department",         # Create a separate ECDF plot for each department
    kind="ecdf"                # Display the Empirical Cumulative Distribution Function
)
plt.suptitle("Cumulative Distribution of Marks Across Departments")  # Add the overall graph title
plt.show()                     # Display all department-wise ECDF plots

# ECDF shows the proportion of observations at or below each x-axis value.
# It is useful for comparing distributions without choosing histogram bins or KDE bandwidths.


# ------------------------------------------------------------
# 7.7 DISPLOT WITH ROW AND COL
# ------------------------------------------------------------

# Create a grid of histograms to compare marks distributions by gender and department.
sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                 # X-axis: Student marks
    row="gender",              # Create separate rows for each gender
    col="department",         # Create separate columns for each department
    kind="hist",                # Display the distribution using histograms
    bins=15                     # Divide the marks range into 15 intervals
)
plt.suptitle("Distribution of Marks by Gender and Department")  # Add the overall graph title
plt.show()                     # Display the complete grid of histograms
# Creates separate distributions for each gender and department combination.


# ------------------------------------------------------------
# 7.8 DISPLOT IMPORTANT PARAMETERS
# ------------------------------------------------------------

# data: Input DataFrame.
# x: Variable whose distribution is displayed.
# y: Optional second variable for bivariate distributions.
# kind: "hist", "kde" or "ecdf".
# hue: Separate distributions by category.
# row, col: Create facets.
# col_wrap: Wrap columns into multiple rows.
# height: Height of each facet.
# aspect: Width / height ratio.
# multiple: Controls how hue distributions are displayed.
# common_bins: Uses common histogram bins across groups.
# common_norm: Controls whether normalization is shared across groups.
# facet_kws: Additional FacetGrid settings.

# Example:

# Compare the marks distributions of different genders across departments using histograms.

sns.displot(
    data=df,                    # Dataset containing student information
    x="marks",                  # X-axis: Student marks
    hue="gender",               # Use different colors to distinguish genders
    col="department",           # Create a separate histogram for each department
    kind="hist",                # Display the data using histograms
    bins=15,                    # Divide the marks range into 15 intervals
    multiple="layer",           # Draw gender-wise histograms on top of each other
    common_bins=True,           # Use the same bin intervals across all genders and departments
    common_norm=False,          # Normalize each gender's distribution independently, if normalized
    height=4,                   # Set the height of each individual graph to 4 inches
    aspect=1.1,                 # Set graph width to 1.1 times its height
    facet_kws={                 # Customize the settings of the subplot axes
        "sharex": True,         # Use the same X-axis scale across all department graphs
        "sharey": True          # Use the same Y-axis scale across all department graphs
    }
)
plt.suptitle("Distribution of Marks by Gender Across Departments")  # Add the overall graph title
plt.show()                     # Display all department-wise histograms

# common_bins=True: Makes bin boundaries consistent across groups.
# common_norm=False: Normalizes each group independently when using a normalized statistic.
# Consistent bins and scales are important for fair distribution comparisons.


# ============================================================
# 8. CATPLOT(): FIGURE-LEVEL CATEGORICAL PLOTS
# ============================================================

# catplot() is a figure-level function for categorical data.
# It supports multiple categorical plot types.
# Common kinds: "strip", "swarm", "box", "violin", "boxen", "point", "bar", "count"
# We studied the individual plot types in earlier phases.
# Here we focus on arranging them into multiple panels.


# ------------------------------------------------------------
# 8.1 BASIC CATPLOT
# ------------------------------------------------------------

# Compare the distribution of student marks across different departments using box plots.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Student marks
    kind="box"                # Display the distribution using box plots
)
plt.suptitle("Distribution of Marks Across Departments")  # Add the overall graph title
plt.show()                   # Display the box plot


# ------------------------------------------------------------
# 8.2 CATPLOT WITH COL
# ------------------------------------------------------------

# Compare the distribution of student marks across departments, separately for each gender.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Student marks
    col="gender",              # Create a separate box plot for each gender
    kind="box"                 # Display the distribution using box plots
)
plt.suptitle("Distribution of Marks by Department and Gender")  # Add the overall graph title
plt.show()                     # Display all gender-wise box plots


# Creates separate boxplots for Male and Female students.

# ------------------------------------------------------------
# 8.3 CATPLOT WITH ROW AND COL
# ------------------------------------------------------------

# Compare marks distributions across departments, separated by gender and result.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Student marks
    row="gender",              # Create separate rows for each gender
    col="result",              # Create separate columns for each result (Pass/Fail)
    kind="violin"               # Display distributions using violin plots
)
plt.suptitle("Distribution of Marks by Department, Gender and Result")  # Add the overall graph title
plt.show()                     # Display the complete grid of violin plots

# Compares department-wise marks distributions
# across gender and result categories.


# ------------------------------------------------------------
# 8.4 CATPLOT WITH HUE
# ------------------------------------------------------------

# Compare marks distributions between genders within each department using box plots.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Student marks
    hue="gender",              # Display separate colored box plots for each gender within each department
    kind="box"                 # Display the distribution using box plots
)
plt.suptitle("Comparison of Marks by Department and Gender")  # Add the overall graph title
plt.show()                     # Display the box plots

# hue compares genders inside each department panel.


# ------------------------------------------------------------
# 8.5 CATPLOT WITH DIFFERENT KINDS
# ------------------------------------------------------------

# Boxplot
# Compare marks distributions across departments using separate box plots for each gender.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Student marks
    col="gender",              # Create a separate graph for each gender
    kind="box"                 # Display the distribution using box plots
)
plt.suptitle("Distribution of Marks Across Departments by Gender")  # Add the overall graph title
plt.show()                     # Display all gender-wise box plots



# Violin plot
# Compare the distribution and density of marks across departments for each gender using violin plots.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Student marks
    col="gender",              # Create a separate graph for each gender
    kind="violin"               # Display the distribution using violin plots
)
plt.suptitle("Distribution of Marks Across Departments by Gender")  # Add the overall graph title
plt.show()                     # Display all gender-wise violin plots



# Strip plot
# Compare individual student marks across departments using separate strip plots for each gender.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Individual student marks
    col="gender",              # Create a separate graph for each gender
    kind="strip"               # Display individual observations as scattered dots
)
plt.suptitle("Individual Marks Across Departments by Gender")  # Add the overall graph title
plt.show()                     # Display all gender-wise strip plots



# Swarm plot
# Compare individual student marks across departments using swarm plots for each gender.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Individual student marks
    col="gender",              # Create a separate graph for each gender
    kind="swarm"                # Display individual points without overlapping them as much as possible
)
plt.suptitle("Individual Marks Across Departments by Gender")  # Add the overall graph title
plt.show()                     # Display all gender-wise swarm plots


# Bar plot
# Compare the average marks across departments using separate bar plots for each gender.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Student marks
    col="gender",              # Create a separate graph for each gender
    kind="bar"                 # Display the average marks using bars
)
plt.suptitle("Average Marks Across Departments by Gender")  # Add the overall graph title
plt.show()                     # Display all gender-wise bar plots


# Point plot
# Compare average marks across departments using point plots for each gender.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Student marks
    col="gender",              # Create a separate graph for each gender
    kind="point"               # Display average marks as points with confidence intervals
)
plt.suptitle("Average Marks Across Departments by Gender")  # Add the overall graph title
plt.show()                     # Display all gender-wise point plots


# Count plot :
# Compare the number of students in each department using separate count plots for each gender.
sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    col="gender",              # Create a separate graph for each gender
    kind="count"               # Count and display the number of students in each department
)
plt.suptitle("Number of Students by Department and Gender")  # Add the overall graph title
plt.show()                     # Display all gender-wise count plots


# ------------------------------------------------------------
# 8.6 CATPLOT IMPORTANT PARAMETERS
# ------------------------------------------------------------

# data: Input DataFrame.
# x, y: Categorical and numerical variables.
# kind: Selects the categorical plot type.
# hue: Adds another grouping variable.
# row: Creates separate rows.
# col: Creates separate columns.
# col_wrap: Controls the number of facet columns per row.
# order: Controls the category order on the categorical axis.
# hue_order: Controls the hue category order.
# height: Height of each facet.

# Compare marks distributions across departments and genders, separately for each result.
sns.catplot(
    data=df,                              # Dataset containing student information
    x="department",                       # X-axis: Different departments
    y="marks",                            # Y-axis: Student marks
    hue="gender",                         # Use different colors for each gender
    col="result",                         # Create a separate graph for each result (Pass/Fail)
    kind="box",                           # Display marks distributions using box plots
    order=["CSE", "ECE", "MECH"],         # Display departments in the specified order on the X-axis
    height=4,                             # Set the height of each individual graph to 4 inches
    aspect=1.1,                           # Set graph width to 1.1 times its height
    sharey=True                           # Use the same Y-axis scale across all result-wise graphs
)
plt.suptitle("Marks Distribution by Department, Gender and Result")  # Add the overall graph title
plt.show()                                     # Display all result-wise box plots

# Use sharey=True when you want the same numerical scale
# across panels for fair comparison.


# ============================================================
# 9. LMPLOT(): FIGURE-LEVEL REGRESSION FACETING
# ============================================================

# lmplot() combines regression plotting with faceting.
# It is useful for comparing relationships and fitted regression trends across categories.
# It is figure-level, so it manages its own figure.


# ------------------------------------------------------------
# 9.1 BASIC LMPLOT
# ------------------------------------------------------------

sns.lmplot(
    data=df,
    x="study_hours",
    y="marks"
)
plt.show()


# ------------------------------------------------------------
# 9.2 LMPLOT WITH HUE
# ------------------------------------------------------------

sns.lmplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="gender"
)
plt.show()


# ------------------------------------------------------------
# 9.3 LMPLOT WITH FACETING
# ------------------------------------------------------------

sns.lmplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    hue="gender",
    height=4,
    aspect=1.1
)
plt.show()

# This allows us to compare regression relationships across departments and genders.

# Important:
# A fitted regression line shows an estimated relationship.
# It does not prove that study hours cause higher marks.
# For actual model evaluation, use proper training and validation data rather than relying only on this plot.


# ============================================================
# 10. PAIRGRID AND JOINTGRID: GRID OBJECT AWARENESS
# ============================================================

# We covered pairplot() and jointplot() in Phase 4.
# Here we only need to understand their grid objects.

# pairplot(): Returns a PairGrid-like object.
# jointplot(): Returns a JointGrid object.
# These are useful for more customized multi-variable and joint-distribution layouts.

# Example:
pair_grid = sns.pairplot(
    data=df,
    vars=[
        "study_hours",
        "attendance",
        "assignment_score",
        "marks"
    ],
    hue="result"
)
plt.show()


# Example:
joint_grid = sns.jointplot(
    data=df,
    x="study_hours",
    y="marks",
    kind="reg"
)
plt.show()

# You do not need to master custom PairGrid or JointGrid methods at this stage.
# Understand their purpose and how they differ from
# FacetGrid.


# ============================================================
# 11. RETURN OBJECTS: IMPORTANT TO UNDERSTAND
# ============================================================

# Seaborn figure-level functions return grid objects.

# Examples:
# sns.relplot()  -> FacetGrid
# sns.displot()  -> FacetGrid
# sns.catplot()  -> FacetGrid
# sns.lmplot()   -> FacetGrid
# sns.pairplot() -> PairGrid
# sns.jointplot()-> JointGrid
#
# Axes-level functions generally return an Axes object.
# Example:

grid = sns.relplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    kind="scatter"
)

# Access the underlying Matplotlib Figure.
figure = grid.figure

# Access the individual Axes.
axes = grid.axes

# Change the overall title.
grid.figure.suptitle(
    "Student Performance",
    y=1.05
)
plt.show()

# grid.figure: The complete Matplotlib Figure.
# grid.axes: Array of individual Matplotlib Axes.
# These properties are useful when you need additional Matplotlib customization after creating a Seaborn grid.


# ============================================================
# 12. SHAREX AND SHAREY: COMPARISON OF SCALES
# ============================================================

# When creating multiple panels, you must decide whether their axes should use the same scale.
# sharex=True: All panels share the x-axis scale.
# sharey=True: All panels share the y-axis scale.

# Example:

# Compare the relationship between study hours and marks across different departments.
sns.relplot(
    data=df,                  # Dataset containing student information
    x="study_hours",          # X-axis: Number of hours students study
    y="marks",                 # Y-axis: Marks obtained by students
    col="department",         # Create a separate scatter plot for each department
    kind="scatter",            # Display individual students as scatter points
    facet_kws={                # Customize the subplot axes
        "sharex": True,        # Use the same X-axis scale across all department plots
        "sharey": True         # Use the same Y-axis scale across all department plots
    }
)
plt.suptitle("Study Hours vs Marks Across Departments")  # Add the overall graph title
plt.show()                     # Display all department-wise scatter plots


# ------------------------------------------------------------
# 12.1 WHEN TO USE SHARED AXES
# ------------------------------------------------------------

# Use shared scales when:
# - Comparing numerical values across categories.
# - Comparing distributions between departments.
# - Comparing model results across groups.
# - You want visual comparisons to be fair.

# Use independent scales when:
# - Groups have very different numerical ranges.
# - One group's values would become difficult to inspect.
# - Local patterns are more important than direct comparison.

# WARNING
# Independent scales can make small differences look large.
# Always check the axis limits before interpreting panels.


# ============================================================
# 13. COMPLETE MULTI-PLOT EDA EXAMPLE
# ============================================================

# Now combine the concepts into a small EDA dashboard.
# We will use figure-level Seaborn functions to compare student performance across departments and genders.
# This is a reusable analysis pattern, not a full project.


# ------------------------------------------------------------
# 13.1 RELATIONSHIP ANALYSIS
# ------------------------------------------------------------

# Create separate scatter plots to study the relationship between study hours and marks across departments.

relationship_grid = sns.relplot(
    data=df,                  # Dataset containing student information
    x="study_hours",          # X-axis: Number of hours students study
    y="marks",                # Y-axis: Marks obtained by students
    col="department",         # Create a separate plot for each department
    hue="result",             # Use different colors to distinguish Pass and Fail
    kind="scatter",           # Display individual students as scatter points
    height=4,                 # Set the height of each individual plot to 4 inches
    aspect=1.1,               # Set each plot's width to 1.1 times its height
    facet_kws={               # Customize the settings of individual subplot axes
        "sharex": True,       # Use the same X-axis scale across all department plots
        "sharey": True        # Use the same Y-axis scale across all department plots
    }
)

relationship_grid.set_axis_labels(
    "Study Hours",             # Set the X-axis label for all plots
    "Marks"                    # Set the Y-axis label for all plots
)
relationship_grid.set_titles(
    "Department: {col_name}"  # Set each subplot title using its department name
)
relationship_grid.figure.suptitle(
    "Relationship Between Study Hours and Marks",  # Set the overall figure title
    y=1.05                   # Adjust the vertical position of the overall title
)
plt.show()                    # Display the complete figure


# ------------------------------------------------------------
# 13.2 DISTRIBUTION ANALYSIS
# ------------------------------------------------------------

# Create separate histograms to compare marks distributions by department and gender.
distribution_grid = sns.displot(
    data=df,                  # Dataset containing student information
    x="marks",                # X-axis: Marks obtained by students
    col="department",         # Create a separate histogram for each department
    hue="gender",             # Use different colors to distinguish genders
    kind="hist",              # Display the distribution using histograms
    bins=15,                  # Divide marks into 15 intervals (bins)
    multiple="layer",         # Draw gender-wise histograms on top of each other
    alpha=0.5,                # Set 50% opacity so overlapping histograms remain visible
    height=4,                 # Set the height of each individual plot to 4 inches
    aspect=1.1,               # Set each plot's width to 1.1 times its height
    facet_kws={               # Customize the settings of individual subplot axes
        "sharex": True,       # Use the same X-axis scale across all department plots
        "sharey": True        # Use the same Y-axis scale across all department plots
    }
)
distribution_grid.set_axis_labels(
    "Marks",                    # Set the X-axis label for all plots
    "Number of Students"        # Set the Y-axis label for all plots
)
distribution_grid.figure.suptitle(
    "Marks Distribution by Department and Gender",  # Set the overall figure title
    y=1.05                    # Adjust the vertical position of the overall title
)
plt.show()                     # Display all department-wise histograms


# ------------------------------------------------------------
# 13.3 CATEGORICAL ANALYSIS
# ------------------------------------------------------------

# Compare marks distributions across departments and genders, separately for each result.
categorical_grid = sns.catplot(
    data=df,                  # Dataset containing student information
    x="department",           # X-axis: Different departments
    y="marks",                # Y-axis: Marks obtained by students
    hue="gender",             # Use different colors to distinguish genders
    col="result",             # Create a separate graph for each result (Pass/Fail)
    kind="box",               # Display marks distributions using box plots
    height=4,                 # Set the height of each individual graph to 4 inches
    aspect=1.1,               # Set each graph's width to 1.1 times its height
    sharey=True               # Use the same Y-axis scale across all result-wise graphs
)
categorical_grid.set_axis_labels(
    "Department",              # Set the X-axis label for all graphs
    "Marks"                    # Set the Y-axis label for all graphs
)
categorical_grid.figure.suptitle(
    "Marks Comparison by Department and Result",  # Set the overall figure title
    y=1.05                     # Adjust the vertical position of the overall title
)
plt.show()                     # Display all result-wise box plots


# ------------------------------------------------------------
# 13.4 REGRESSION COMPARISON
# ------------------------------------------------------------

regression_grid = sns.lmplot(
    data=df,
    x="study_hours",
    y="marks",
    col="department",
    hue="gender",
    height=4,
    aspect=1.1
)
regression_grid.set_axis_labels(
    "Study Hours",
    "Marks"
)
regression_grid.figure.suptitle(
    "Regression Trends Across Departments",
    y=1.05
)
plt.show()


# ============================================================
# 14. HOW TO CHOOSE THE RIGHT FUNCTION
# ============================================================

# Use the following decision guide.

# ------------------------------------------------------------
# RELPLOT()
# ------------------------------------------------------------

# Use when:
# - You want to study relationships between variables.
# - You need scatterplots or lineplots.
# - You want separate panels by category.
#
# Example: Study hours vs marks across departments.


# ------------------------------------------------------------
# DISPLOT()
# ------------------------------------------------------------

# Use when:
# - You want to study numerical distributions.
# - You need histograms, KDE or ECDF plots.
# - You want to compare distributions across groups.
#
# Example: Marks distribution across departments.


# ------------------------------------------------------------
# CATPLOT()
# ------------------------------------------------------------

# Use when:
# - You want to compare categorical groups.
# - You need boxplots, violinplots, barplots, etc.
# - You want separate panels by category.

# Example: Marks across departments and genders.


# ------------------------------------------------------------
# FACETGRID()
# ------------------------------------------------------------

# Use when:
# - You need more manual control over faceting.
# - You want to map custom plotting functions.
# - You want to create a custom grid from subsets.

# Example: Custom histogram or scatterplot for each department.


# ------------------------------------------------------------
# LMPLOT()
# ------------------------------------------------------------

# Use when:
# - You want regression plots across groups.
# - You want to compare fitted trends in separate panels.

# Example: Study hours vs marks across departments.


# ============================================================
# 15. IMPORTANT PRACTICAL RULES AND COMMON MISTAKES
# ============================================================

# 1. Do not pass ax= to figure-level functions in the same way you do with axes-level functions.
# 2. Use relplot(), displot() and catplot() for automatic faceting.
# 3. Use FacetGrid when you need more manual control.
# 4. Do not create too many facets, Too many panels make individual plots difficult to read.
# 5. Use sharex/sharey carefully, Different scales can make comparisons misleading.
# 6. Use col_order and row_order when category order matters.
# 7. Use hue for color-based grouping, Use col and row for separate panels.
# 8. Remember that col_wrap works with col faceting and cannot be combined with row faceting.
# 9. Use map_dataframe() when the plotting function expects a DataFrame through data=.
# 10. Keep axis labels and titles clear.
# 11. Use consistent scales when comparing categories.
# 12. Avoid using too many colors, markers and grouping variables in one chart.
# 13. A visual pattern is not proof of causation.
# 14. Figure-level functions manage their own layout, Use their returned grid object for customization.
# 15. Check your Seaborn version when using newer parameters, Some arguments differ across versions.


# ============================================================
# 16. ML ENGINEERING AND EDA APPLICATIONS
# ============================================================

# Multi-plot layouts are useful during exploratory data
# analysis before building machine learning models.

# Common applications:

# 1. Compare feature distributions across target classes.
# 2. Compare numerical relationships across categories.
# 3. Identify groups with different feature ranges.
# 4. Inspect class imbalance across different categories.
# 5. Compare outlier patterns across groups.
# 6. Compare training data segments visually.
# 7. Understand whether relationships differ between groups.
# 8. Examine possible data-quality issues in subsets.

# IMPORTANT:
# Visualization helps you understand data.
# It does not replace:

# - Train/test splitting
# - Statistical validation
# - Feature engineering
# - Model evaluation
# - Cross-validation

# These are separate parts of the ML workflow.


# ============================================================
# 17. PRACTICE TASKS
# ============================================================

# Complete these tasks using the sample DataFrame.

# TASK 1:
# Create a scatterplot using relplot().
# Separate panels by department.

# TASK 2:
# Add result as hue to the above plot.

# TASK 3:
# Add gender as style.

# TASK 4:
# Create separate rows for gender and columns for department.

# TASK 5:
# Use col_wrap=2 and observe the layout.

# TASK 6:
# Create a histogram using displot().
# Compare marks distributions across departments.

# TASK 7:
# Use hue="gender" and multiple="layer".

# TASK 8:
# Create a KDE distribution using displot().

# TASK 9:
# Create an ECDF distribution using displot().

# TASK 10:
# Create a boxplot using catplot().
# Compare marks across departments.

# TASK 11:
# Add gender as hue.

# TASK 12:
# Create separate panels for Pass and Fail.

# TASK 13:
# Create a FacetGrid with department as col.
# Apply histplot() using map_dataframe().

# TASK 14:
# Customize FacetGrid titles and axis labels.

# TASK 15:
# Create a regression comparison using lmplot().

# TASK 16:
# Compare shared and independent y-axis scales.

# TASK 17:
# Create a complete relationship, distribution and categorical analysis using separate figure-level plots.

# TASK 18:
# Explain why relplot() is figure-level but scatterplot()
# is axes-level.

# TASK 19:
# Explain the difference between hue and col.

# TASK 20:
# Explain when you would use FacetGrid instead of relplot().


# ============================================================
# 18. PHASE 6 REVISION CHECKLIST
# ============================================================

# [ ] Understand axes-level functions.
# [ ] Understand figure-level functions.
# [ ] Understand the difference between them.
# [ ] Understand faceting.
# [ ] Understand row and col.
# [ ] Understand col_wrap.
# [ ] Understand col_order and row_order.
# [ ] Understand FacetGrid.
# [ ] Understand map() and map_dataframe().
# [ ] Know how to customize FacetGrid titles and labels.
# [ ] Understand relplot().
# [ ] Know scatter and line kinds in relplot().
# [ ] Understand displot().
# [ ] Know hist, KDE and ECDF kinds in displot().
# [ ] Understand multiple="layer", "dodge", "stack", "fill".
# [ ] Understand catplot().
# [ ] Know the different categorical kinds in catplot().
# [ ] Understand lmplot() for regression faceting.
# [ ] Understand returned grid objects.
# [ ] Understand sharex and sharey.
# [ ] Know how to select the right function.
# [ ] Complete the practice tasks.


# ============================================================
# 19. WHAT YOU SHOULD BE ABLE TO DO AFTER PHASE 6
# ============================================================

# By the end of this phase, you should be able to:

# 1. Create multi-panel visualizations.
# 2. Compare categories using separate panels.
# 3. Use figure-level Seaborn functions confidently.
# 4. Use FacetGrid for custom faceting.
# 5. Combine hue, row and col effectively.
# 6. Control facet dimensions and axis sharing.
# 7. Build multi-panel EDA visualizations.
# 8. Choose the appropriate plotting function for a task.

# You do not need to memorize every parameter.
# Focus on understanding the purpose of each function and knowing when to use it.