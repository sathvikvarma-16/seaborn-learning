# ============================================================
# SEABORN PHASE 1: BASIC STATISTICAL PLOTS
# ============================================================
# Topics:
# 1. Scatter Plot
# 2. Line Plot
# 3. Bar Plot
# 4. Count Plot
#
# Goal:
# Understand relationships, trends, category comparisons,
# and category frequencies using Seaborn.
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. SETUP
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")
# Create a sample student dataset
df = pd.DataFrame({
    "student": ["A", "B", "C", "D", "E", "F",
                "G", "H", "I", "J", "K", "L"],
    "study_hours": [1, 2, 3, 4, 5, 6,
                    2, 3, 5, 7, 8, 6],
    "marks": [35, 42, 50, 55, 65, 75,
              45, 52, 68, 80, 90, 78],
    "attendance": [60, 65, 70, 75, 80, 90,
                   68, 72, 85, 95, 98, 88],
    "gender": ["Male", "Female", "Male", "Female",
               "Male", "Female", "Male", "Female",
               "Male", "Female", "Male", "Female"],
    "result": ["Fail", "Fail", "Pass", "Pass",
               "Pass", "Pass", "Fail", "Pass",
               "Pass", "Pass", "Pass", "Pass"]
})

print("STUDENT DATASET")
print(df)


# ============================================================
# 2. SCATTER PLOT
# ============================================================

# Relationship between study hours and marks

sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks"
)

plt.title("Study Hours vs Marks")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()


# Scatter plot with hue, style, and size

sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="result",          # Color based on result
    style="result",        # Marker shape based on result
    size="attendance",     # Marker size based on attendance
    palette="Set1",
    alpha=0.8
)

plt.title("Student Performance by Result")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.show()


# ============================================================
# 3. LINE PLOT
# ============================================================

# Create monthly sales data

sales_df = pd.DataFrame({
    "month": [1, 2, 3, 4, 5, 6],
    "sales": [100, 150, 130, 180, 220, 250]
})

# Display sales trend

# Create a line plot using Seaborn
sns.lineplot(
    data=sales_df,  # Dataset to use
    x="month",      # X-axis values
    y="sales",      # Y-axis values
    marker="o"      # Display circular markers on data points
)

# Add title and axis labels
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

# Display the plot
plt.show()


# Line plot with multiple categories

region_df = pd.DataFrame({
    "month": [1, 2, 3, 4, 5, 6] * 2,
    "sales": [100, 120, 150, 140, 180, 200,
              80, 100, 110, 130, 150, 170],
    "region": ["North"] * 6 + ["South"] * 6
})


plt.title("Monthly Sales by Region")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()


# ============================================================
# 4. BAR PLOT
# ============================================================

# Average marks for each result category

sns.barplot(
    data=df,
    x="result",
    y="marks"
)

plt.title("Average Marks by Result")
plt.xlabel("Result")
plt.ylabel("Average Marks")
plt.show()


# Compare average marks by gender and resul
# Create a bar plot using Seaborn
sns.barplot(
    data=df,          # Dataset to use
    x="result",       # X-axis categories
    y="marks",        # Y-axis values
    hue="gender",     # Separate bars based on gender
    palette="Set2"    # Set color palette
)

plt.title("Average Marks by Result and Gender")
plt.xlabel("Result")
plt.ylabel("Average Marks")
plt.show()


# Use median instead of mean

# Create a bar plot using Seaborn
sns.barplot(
    data=df,             # Dataset to use
    x="result",          # X-axis categories
    y="marks",            # Y-axis values
    estimator=np.median,  # Calculate median instead of mean
    errorbar=None         # Remove error bars
)

plt.title("Median Marks by Result")
plt.xlabel("Result")
plt.ylabel("Median Marks")
plt.show()


# ============================================================
# 5. COUNT PLOT
# ============================================================

# Count students in each result category

sns.countplot(
    data=df,
    x="result"
)

plt.title("Number of Students by Result")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.show()


# Count students by result and gender

sns.countplot(
    data=df,
    x="result",
    hue="gender",
    palette="Set2"
)

plt.title("Student Count by Result and Gender")
plt.xlabel("Result")
plt.ylabel("Number of Students")
plt.show()


# ============================================================
# 6. COMBINED DASHBOARD
# ============================================================

# Create a 2 x 2 dashboard

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# Scatter plot
sns.scatterplot(
    data=df,
    x="study_hours",
    y="marks",
    hue="result",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Study Hours vs Marks")

# Line plot
sns.lineplot(
    data=sales_df,
    x="month",
    y="sales",
    marker="o",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Monthly Sales Trend")

# Bar plot
sns.barplot(
    data=df,
    x="result",
    y="marks",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Average Marks by Result")

# Count plot
sns.countplot(
    data=df,
    x="result",
    hue="gender",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Student Count by Result")

# Adjust spacing between plots
plt.tight_layout()

plt.show()


# ============================================================
# END OF PHASE 1
# ============================================================