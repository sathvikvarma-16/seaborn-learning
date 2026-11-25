"""
============================================================
SEABORN PHASE 0: INTRODUCTION, SETUP AND FUNDAMENTALS
============================================================

Objective:
    Understand Seaborn, its relationship with Matplotlib,
    Pandas integration, data formats, built-in datasets,
    and important plotting parameters.

Libraries:
    NumPy
    Pandas
    Matplotlib
    Seaborn

Installation:
    pip install numpy pandas matplotlib seaborn

============================================================
"""


# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

# NumPy is used for numerical operations and arrays.
import numpy as np

# Pandas is used for creating and manipulating DataFrames.
import pandas as pd

# Matplotlib is used for displaying and customizing figures.
import matplotlib.pyplot as plt

# Seaborn is used for statistical data visualization.
import seaborn as sns


# ============================================================
# 2. CHECK SEABORN VERSION
# ============================================================

# Display the installed Seaborn version.
# This helps verify that Seaborn is installed correctly.
print("Seaborn Version:", sns.__version__)


# ============================================================
# 3. SET A DEFAULT SEABORN THEME
# ============================================================

# set_theme() configures the default appearance of Seaborn plots.
# It affects properties such as background, grid, fonts and colors.
sns.set_theme(
    style="whitegrid"
)


# ============================================================
# 4. CREATE A SAMPLE PANDAS DATAFRAME
# ============================================================

# Create sample student data using a Python dictionary.
# Each dictionary key becomes a DataFrame column.
student_data = {
    "student": ["A", "B", "C", "D", "E", "F", "G", "H"],
    "study_hours": [2, 3, 4, 5, 6, 7, 8, 9],
    "marks": [40, 48, 55, 65, 72, 78, 88, 95],
    "sleep_hours": [8, 7, 7, 6, 7, 6, 8, 7],
    "result": [
        "Fail", "Fail", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Pass"
    ],
    "attendance": [60, 65, 70, 75, 80, 85, 90, 95]
}

# Convert the dictionary into a Pandas DataFrame.
df = pd.DataFrame(student_data)

# Display the complete DataFrame.
print("\nStudent Dataset:")
print(df)

# Display the first five records.
print("\nFirst Five Records:")
print(df.head())

# Display dataset dimensions.
print("\nDataset Shape:")
print(df.shape)

# Display column data types and non-null counts.
print("\nDataset Information:")
df.info()

# Display statistical summaries of numerical columns.
print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 5. BASIC SEABORN SCATTER PLOT
# ============================================================

# Create a scatter plot using Seaborn.
# data specifies the DataFrame.
# x specifies the column for the horizontal axis.
# y specifies the column for the vertical axis.

sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks"
)

# Add a chart title using Matplotlib.
plt.title("Study Hours vs Marks")

# Add labels to the axes.
plt.xlabel("Study Hours")
plt.ylabel("Marks")

# Display the chart.
plt.show()


# ============================================================
# 6. UNDERSTAND THE HUE PARAMETER
# ============================================================

# hue separates observations using different colors.
# Here, students are grouped according to their result.
# Pass and Fail observations receive different colors.

sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="result"
)

plt.title("Study Hours vs Marks by Result")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()


# ============================================================
# 7. UNDERSTAND THE STYLE PARAMETER
# ============================================================

# style uses different marker styles for different categories.
# hue and style can be combined to distinguish groups
# using both color and marker shape.

# Create a scatter plot using Seaborn
sns.scatterplot(
    data=df,             # Dataset to use
    x="study_hours",     # X-axis column
    y="marks",           # Y-axis column
    hue="result",        # Different colors for each result
    style="result"       # Different marker styles for each result
)

# Add title and axis labels
plt.title("Student Performance by Result")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

# Display the plot
plt.show()

# ============================================================
# 8. UNDERSTAND THE SIZE PARAMETER
# ============================================================

# size maps a numerical variable to marker size.
# Students with different sleep-hour values receive
# different marker sizes.

# Create a scatter plot using Seaborn
sns.scatterplot(
    data=df,             # Dataset to use
    x="study_hours",     # X-axis values
    y="marks",           # Y-axis values
    size="sleep_hours",  # Bubble size based on sleep hours
    hue="result",        # Different colors for each result
    sizes=(50, 250)      # Set minimum and maximum bubble sizes
)

plt.title("Study Hours, Marks and Sleep")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.show()


# ============================================================
# 9. UNDERSTAND THE PALETTE PARAMETER
# ============================================================

# palette changes the colors assigned to categorical groups.
# Set2 is one of the available Seaborn color palettes.

# Create a scatter plot using Seaborn
sns.scatterplot(
    data=df,             # Dataset to use
    x="study_hours",     # X-axis values
    y="marks",           # Y-axis values
    hue="result",        # Different colors for each result
    palette="Set2"       # Set color palette
)

# Display the plot
plt.show()

plt.title("Student Performance with Set2 Palette")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.show()


# ============================================================
# 10. SEABORN WITH MATPLOTLIB OBJECT-ORIENTED API
# ============================================================

# Create a Figure and an Axes object.
# figsize controls the figure width and height in inches.

fig, ax = plt.subplots(figsize=(8, 5))

# Draw a Seaborn chart on the existing Matplotlib Axes.
# ax= specifies where the chart should be drawn.

# Create a scatter plot using Seaborn
sns.scatterplot(
    data=df,          # Dataset to use
    x="study_hours",  # X-axis values
    y="marks",        # Y-axis values
    hue="result",     # Different colors for each result
    style="result",   # Different marker styles for each result
    ax=ax             # Display plot on the specified axis
)

# Display the plot
plt.show()
# Customize the existing Axes using Matplotlib.
ax.set_title("Student Performance Dashboard")
ax.set_xlabel("Daily Study Hours")
ax.set_ylabel("Final Marks")

# Add a legend.
ax.legend(title="Student Result")

# Adjust spacing automatically.
plt.tight_layout()

# Display the figure.
plt.show()


# ============================================================
# 11. UNDERSTAND LONG-FORM DATA
# ============================================================

# Long-form data stores different variables in separate columns.
# Each row represents one observation.

long_data = {
    "student": ["A", "A", "B", "B", "C", "C"],

    "subject": [
        "Maths", "Science",
        "Maths", "Science",
        "Maths", "Science"
    ],

    "marks": [85, 90, 75, 80, 95, 88]
}

# Convert the dictionary into a DataFrame.
long_df = pd.DataFrame(long_data)
print("\nLong-Form Data:")
print(long_df)

# Create a bar plot using the long-form DataFrame.
# x represents subject categories.
# y represents numerical marks.
sns.barplot(
    data=long_df,
    x="subject",
    y="marks"
)

plt.title("Average Marks by Subject")
plt.xlabel("Subject")
plt.ylabel("Marks")
plt.show()


# ============================================================
# 12. UNDERSTAND WIDE-FORM DATA
# ============================================================

# Wide-form data stores different groups in separate columns.

wide_data = {
    "Maths": [85, 75, 95, 80, 90],
    "Science": [90, 80, 88, 85, 92],
    "English": [78, 82, 91, 87, 89]
}

# Convert the dictionary into a DataFrame.
wide_df = pd.DataFrame(wide_data)

print("\nWide-Form Data:")
print(wide_df)

# Seaborn interprets each numerical column as a separate group.
# This creates a box plot for comparing subject distributions.

sns.boxplot(
    data=wide_df
)

plt.title("Subject Marks Distribution")
plt.xlabel("Subject")
plt.ylabel("Marks")
plt.show()


# ============================================================
# 13. LOAD A BUILT-IN SEABORN DATASET
# ============================================================

# load_dataset() loads an example dataset provided by Seaborn.
# The tips dataset contains restaurant bill and tip information.
# Internet access may be required the first time it is loaded.

tips = sns.load_dataset("tips")

# Display the first five rows.
print("\nTips Dataset:")
print(tips.head())

# Display dataset dimensions.
print("\nTips Dataset Shape:")
print(tips.shape)

# Display column names.
print("\nTips Dataset Columns:")
print(tips.columns)

# Display information about columns and data types.
print("\nTips Dataset Information:")
tips.info()

# Display statistical summaries.
print("\nTips Dataset Summary:")
print(tips.describe())


# ============================================================
# 14. VISUALIZE THE BUILT-IN TIPS DATASET
# ============================================================

# Compare total bills with tips.
# hue separates observations according to smoker category.

sns.scatterplot(
    data=tips,
    x="total_bill",
    y="tip",
    hue="smoker"
)

plt.title("Total Bill vs Tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")

plt.show()


# ============================================================
# 15. BASIC DATAFRAME COLUMN SELECTION
# ============================================================

# Select one column from the DataFrame.
# The result is a Pandas Series.

study_hours = df["study_hours"]
print("\nStudy Hours Column:")
print(study_hours)

# Select multiple columns from the DataFrame.
# The result is another DataFrame.
selected_data = df[
    ["study_hours", "marks", "attendance"]
]

print("\nSelected Columns:")
print(selected_data)


# ============================================================
# 16. FILTER DATA BEFORE VISUALIZATION
# ============================================================

# Filter students whose marks are greater than or equal to 70.
# This creates a new DataFrame containing only matching rows.

high_performers = df[
    df["marks"] >= 70
]

print("\nHigh Performing Students:")
print(high_performers)

# Visualize the filtered DataFrame.
sns.scatterplot(
    data=high_performers,
    x="study_hours",
    y="marks",
    hue="result"
)

plt.title("Study Hours vs Marks: High Performers")
plt.xlabel("Study Hours")
plt.ylabel("Marks")

plt.show()


# ============================================================
# 17. FINAL UNDERSTANDING
# ============================================================

print("\nSeaborn Phase 0 Completed!")

print("""
You have practiced:

1. Importing Seaborn, Matplotlib, Pandas and NumPy.
2. Checking the Seaborn version.
3. Setting a Seaborn theme.
4. Creating a Pandas DataFrame.
5. Visualizing DataFrames using Seaborn.
6. Understanding data, x and y.
7. Understanding hue, style, size and palette.
8. Combining Seaborn with Matplotlib.
9. Understanding long-form and wide-form data.
10. Loading a built-in Seaborn dataset.
11. Filtering DataFrames before visualization.

Next:
    Phase 1 - Basic Statistical Plots.
""")