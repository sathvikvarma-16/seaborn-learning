# ============================================================
# SEABORN PHASE 5: CUSTOMIZATION AND VISUAL DESIGN
# ============================================================

# THIS PHASE COVERS:

# 1. Seaborn themes
# 2. set_theme()
# 3. set_style()
# 4. set_context()
# 5. Theme and context differences
# 6. Color palettes
# 7. Qualitative, sequential and diverging palettes
# 8. Choosing colors for categorical and numerical data
# 9. Color palette customization
# 10. Color blindness considerations
# 11. Palette inspection
# 12. Matplotlib and Seaborn customization integration
# 13. Titles and subtitles
# 14. Axis labels and tick formatting
# 15. Tick rotation and formatting
# 16. Legends and their positioning
# 17. Grid and spine customization
# 18. Figure backgrounds and axes backgrounds
# 19. Annotations and reference lines
# 20. Subplot-level customization
# 21. Reusable plotting functions
# 22. Consistent visualization standards
# 23. Saving high-quality visualizations
# 24. Professional EDA dashboard
# 25. ML visualization design
# 26. Common mistakes and best practices


# ============================================================
# 0. IMPORT LIBRARIES AND CREATE SAMPLE DATA
# ============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.ticker import (
    MultipleLocator, # Sets the interval between axis ticks, MultipleLocator(10) means show major ticks at multiples of 10.
    FormatStrFormatter, # Changes how numerical tick labels are displayed using Python's format specifications.
    # '%.0f'	25	No decimal places  || '%.1f'	25.5	One decimal place
    FuncFormatter # Allows you to create your own custom formatting function for tick labels.
)

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)

# Create sample business dataset.

n = 200  # Total number of rows (records) to generate
df = pd.DataFrame({  # Create a DataFrame using a dictionary of columns
    "month": np.tile(  # Repeat the given month names 40 times to generate 200 values
        ["Jan", "Feb", "Mar", "Apr", "May"],
        40
    ),
    "sales": rng.normal(  # Generate normally distributed sales values
        50000,             # Mean (average) sales value
        10000,             # Standard deviation (spread of values)
        n                  # Generate 200 random values
    ).clip(10000, 100000), # Restrict sales values between 10,000 and 100,000

    "expenses": rng.normal(  # Generate normally distributed expenses values
        30000,               # Mean (average) expenses
        7000,                # Standard deviation
        n                    # Generate 200 random values
    ).clip(5000, 70000),    # Restrict expenses between 5,000 and 70,000

    "advertising": rng.normal(  # Generate normally distributed advertising costs
        10000,                  # Mean (average) advertising cost
        3000,                   # Standard deviation
        n                       # Generate 200 random values
    ).clip(1000, 20000),       # Restrict advertising values between 1,000 and 20,000

    "customers": rng.integers(  # Generate random integer values for customers
        100,                    # Minimum value (inclusive)
        1000,                   # Maximum value (exclusive)
        n                       # Generate 200 random integers
    ),

    "region": rng.choice(  # Randomly select a region for each record
        ["North", "South", "East", "West"],  # Possible region names
        n                                      # Generate 200 random selections
    ),

    "product": rng.choice(  # Randomly select a product for each record
        ["Laptop", "Mobile", "Tablet"],  # Possible product names
        n                                # Generate 200 random selections
    )

})

df["profit"] = df["sales"] - df["expenses"]  # Calculate profit by subtracting expenses from sales
print(df.head())  # Display the first 5 rows of the DataFrame


# ============================================================
# 1. UNDERSTANDING SEABORN THEMES
# ============================================================

# A theme controls the overall appearance of visualizations.

# It affects:
# - Background
# - Grid appearance
# - Default colors
# - Text appearance
# - Font sizes
# - Axes appearance

# A consistent theme makes multiple charts look related.
# Seaborn provides five built-in styles:
# darkgrid
# whitegrid
# dark
# white
# ticks
# set_theme() is the main function used to configure the overall visualization appearance.


# ------------------------------------------------------------
# 1.1 SET DEFAULT THEME
# ------------------------------------------------------------
   
sns.set_theme() # Applies Seaborn's default visual theme to your plots.
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Default Seaborn Theme")
plt.show()


# ------------------------------------------------------------
# 1.2 DARKGRID STYLE
# ------------------------------------------------------------

sns.set_theme(style="darkgrid")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Dark Grid Style")
plt.show()


# ------------------------------------------------------------
# 1.3 WHITEGRID STYLE
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("White Grid Style")
plt.show()


# ------------------------------------------------------------
# 1.4 DARK STYLE
# ------------------------------------------------------------

sns.set_theme(style="dark")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Dark Style")
plt.show()


# ------------------------------------------------------------
# 1.5 WHITE STYLE
# ------------------------------------------------------------

sns.set_theme(style="white")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("White Style")
plt.show()


# ------------------------------------------------------------
# 1.6 TICKS STYLE
# ------------------------------------------------------------

sns.set_theme(style="ticks")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Ticks Style")
plt.show()


# ------------------------------------------------------------
# 1.7 COMPARING STYLES
# ------------------------------------------------------------

# darkgrid: Dark background with grid.
# whitegrid: White background with grid.
# Useful for reading numerical values.
# dark: Dark background without prominent grid.
# white: Clean white background.
# ticks: Clean background with axis ticks.

# Practical recommendation:
# whitegrid and white are good general-purpose choice, Choose according to the data and intended presentation.


# ============================================================
# 2. SET_THEME()
# ============================================================

# set_theme() configures multiple visual settings together.
# Main parameters:
# context: Controls scaling of visual elements.
# style: Controls background and grid appearance.
# palette: Default color palette.
# font: Default font family.
# font_scale: Scales font sizes.
# color_codes: Enables shorthand color codes for palette colors.
# rc: Additional Matplotlib configuration dictionary.


# ------------------------------------------------------------
# 2.1 CUSTOM THEME
# ------------------------------------------------------------

sns.set_theme(
    context="notebook",  # Controls the overall scale of plot elements (fonts, labels, lines, etc.)
    style="whitegrid",   # Sets a white background with gridlines
    palette="deep",      # Uses Seaborn's deep color palette for plot elements
    font="sans-serif",   # Sets the font family to sans-serif
    font_scale=1.1,      # Increases the default font sizes by 10%
    color_codes=True     # Allows shorthand color codes (b, g, r, etc.) to use Seaborn palette colors
)
sns.barplot(
    data=df,
    x="product",
    y="sales"
)
plt.title("Sales by Product")
plt.show()


# ------------------------------------------------------------
# 2.2 CUSTOM THEME USING RC
# ------------------------------------------------------------

# rc allows additional Matplotlib configuration.

sns.set_theme(
    style="whitegrid",  # Apply a white background with gridlines
    rc={  # Customize individual Matplotlib visual properties
        "axes.facecolor": "white",       # Set the background color inside the plot area
        "figure.facecolor": "white",     # Set the background color of the entire figure
        "axes.edgecolor": "black",       # Set the color of the plot boundaries (axis edges)
        "axes.labelcolor": "black",      # Set the color of X-axis and Y-axis labels
        "grid.color": "lightgray",       # Set the color of gridlines
        "grid.linestyle": "--",          # Display gridlines using dashed lines
        "grid.linewidth": 0.6            # Set the thickness of gridlines
    }
)
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Theme Customized Using RC")
plt.show()


# ============================================================
# 3. SET_STYLE()
# ============================================================

# set_style() changes the background and grid appearance.
# It does not primarily control the overall font scaling.
# Available styles: darkgrid, whitegrid, dark, white, ticks


# ------------------------------------------------------------
# 3.1 CHANGE STYLE
# ------------------------------------------------------------

sns.set_style("whitegrid")
sns.lineplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("White Grid")
plt.show()


# ------------------------------------------------------------
# 3.2 MODIFY INDIVIDUAL STYLE SETTINGS
# ------------------------------------------------------------

# axes.spines.top: Controls the top border.
# axes.spines.right: Controls the right border.
# axes.grid: Controls grid visibility.

sns.set_style(
    "white",  # Apply a white background style
    rc={  # Customize individual Matplotlib style properties
        "axes.spines.top": False,    # Hide the top border of the plot
        "axes.spines.right": False,  # Hide the right border of the plot
        "axes.grid": True,           # Enable gridlines on the plot
        "grid.linestyle": "--"       # Display gridlines using dashed lines

    }
)
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Customized White Style")
plt.show()


# ============================================================
# 4. SET_CONTEXT()
# ============================================================

# set_context() controls the scale of visual elements.
# Four built-in contexts: paper, notebook, talk, poster

# paper: Smaller elements for reports and documents.
# notebook: Standard interactive notebook size.
# talk: Larger elements for presentations.
# poster: Large elements for posters and displays.
# Context changes the scale of fonts and other visual elements, not the underlying data.


# ------------------------------------------------------------
# 4.1 PAPER CONTEXT
# ------------------------------------------------------------

sns.set_context("paper")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Paper Context")
plt.show()


# ------------------------------------------------------------
# 4.2 NOTEBOOK CONTEXT
# ------------------------------------------------------------

sns.set_context("notebook")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Notebook Context")
plt.show()


# ------------------------------------------------------------
# 4.3 TALK CONTEXT
# ------------------------------------------------------------

sns.set_context("talk")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Talk Context")
plt.show()


# ------------------------------------------------------------
# 4.4 POSTER CONTEXT
# ------------------------------------------------------------

sns.set_context("poster")
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales"
)
plt.title("Poster Context")
plt.show()


# ------------------------------------------------------------
# 4.5 CUSTOM CONTEXT
# ------------------------------------------------------------

# font_scale: Changes font scaling.
# rc: Allows individual context parameters to be adjusted.

sns.set_context(
    "notebook",
    font_scale=1.2,
    rc={
        "lines.linewidth": 2,
        "lines.markersize": 7
    }
)
sns.lineplot(
    data=df,
    x="advertising",
    y="sales"
)   
plt.title("Custom Notebook Context")
plt.show()


# ============================================================
# 5. DIFFERENCE BETWEEN THEME, STYLE AND CONTEXT
# ============================================================

# set_theme(): Controls the overall visualization configuration.
# set_style(): Controls background, grid and spine appearance.
# set_context(): Controls scale of fonts and visual elements.

# REMEMBER:
# Theme = Overall appearance.
# Style = Background and grid.
# Context = Visual scale.

# You should be able to select these independently according to the intended output.


# ============================================================
# 6. UNDERSTANDING COLOR PALETTES
# ============================================================

# A color palette defines the colors used in a chart.

# The correct palette improves: Category distinction, Visual interpretation, Consistency across charts, Accessibility.
# Seaborn provides built-in palettes.
# Important palette categories:
# 1. Qualitative palettes.
# 2. Sequential palettes.
# 3. Diverging palettes.


# ============================================================
# 7. QUALITATIVE PALETTES
# ============================================================

# Qualitative palettes are designed for categorical variables.
# Examples: Department, Region, Product, Gender
# Each category receives a distinct color.
# Common qualitative palettes: deep, muted, pastel, bright, dark, colorblind, Set1, Set2, Set3, tab10


# ------------------------------------------------------------
# 7.1 DEEP PALETTE
# ------------------------------------------------------------

sns.set_palette("deep")  # Set Seaborn's deep color palette for subsequent plots
sns.countplot(
    data=df,          # Specify the DataFrame containing the data
    x="region"        # Count the occurrences of each region and display them on the X-axis
)
plt.title("Deep Palette")  # Set the title of the plot
plt.show()  # Display the visualization


# ------------------------------------------------------------
# 7.2 MUTED PALETTE
# ------------------------------------------------------------

sns.set_palette("muted")
sns.countplot(
    data=df,
    x="region"
)
plt.title("Muted Palette")
plt.show()


# ------------------------------------------------------------
# 7.3 PASTEL PALETTE
# ------------------------------------------------------------

sns.set_palette("pastel")
sns.countplot(
    data=df,
    x="region"
)
plt.title("Pastel Palette")
plt.show()


# ------------------------------------------------------------
# 7.4 BRIGHT PALETTE
# ------------------------------------------------------------

sns.set_palette("bright")
sns.countplot(
    data=df,
    x="region"
)
plt.title("Bright Palette")
plt.show()


# ------------------------------------------------------------
# 7.5 COLORBLIND PALETTE
# ------------------------------------------------------------

# Designed to improve distinction for common forms of color vision deficiency.

sns.set_palette("colorblind")
sns.countplot(
    data=df,
    x="region"
)
plt.title("Colorblind-Friendly Palette")
plt.show()


# ============================================================
# 8. SEQUENTIAL PALETTES
# ============================================================

# Sequential palettes represent values progressing from low to high.
# Useful for: Numerical intensity, Correlation magnitude, Feature value, Heatmaps.

# Examples: blues, greens, reds, purples, viridis, magma, rocket


# ------------------------------------------------------------
# 8.1 BLUES PALETTE
# ------------------------------------------------------------

sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="profit",
    palette="Blues"
)
plt.title("Sequential Blue Palette")
plt.show()


# ------------------------------------------------------------
# 8.2 VIRIDIS PALETTE
# ------------------------------------------------------------

sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="profit",
    palette="viridis" # Purple to green-yellow
)
plt.title("Viridis Palette")
plt.show()


# ------------------------------------------------------------
# 8.3 MAGMA PALETTE
# ------------------------------------------------------------

sns.scatterplot(
    data=df,                 # Specify the DataFrame containing the data
    x="advertising",         # Use advertising values on the X-axis
    y="sales",               # Use sales values on the Y-axis
    hue="profit",            # Map profit values to different point colors
    palette="magma"          # Use the magma color palette to represent profit values
)
plt.title("Magma Palette")   # Set the title of the plot - Dark purple to orange-yellow
plt.show()                   # Display the visualization

# plasma   -	Purple to pink-yellow
# inferno  -	Black to red-yellow
# cividis  -	Blue to yellow, designed for color accessibility


# ============================================================
# 9. DIVERGING PALETTES
# ============================================================

# Diverging palettes emphasize values around a meaningful central point.
# Example: Correlation values from -1 to +1.
# Negative values: One color direction.
# Positive values: Another color direction.
# Center: Often zero.

# Examples: coolwarm, RdBu, vlag, icefire


# ------------------------------------------------------------
# 9.1 DIVERGING HEATMAP
# ------------------------------------------------------------

correlation = df[
    ["sales", "expenses", "advertising", "customers", "profit"]
].corr()
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1
)
plt.title("Correlation Using Diverging Palette")
plt.show()


# ------------------------------------------------------------
# 9.2 OTHER DIVERGING PALETTES
# ------------------------------------------------------------

sns.heatmap(
    correlation,       # Provide the correlation matrix to visualize
    annot=True,        # Display the numerical correlation values inside each cell
    cmap="vlag",       # Use the vlag diverging color palette
    center=0,           # Set zero as the midpoint of the color scale
    vmin=-1,            # Set the minimum color scale value to -1
    vmax=1              # Set the maximum color scale value to +1
)
plt.title("Vlag Diverging Palette")  # Set the title of the heatmap
plt.show()                           # Display the heatmap


# ============================================================
# 10. CUSTOM COLOR PALETTES
# ============================================================

# You can define your own colors.
# This is useful when: Maintaining brand colors, Maintaining consistency, Highlighting specific categories.
# Colors can be supplied as: Named colors, Hexadecimal color codes, RGB or RGBA tuples.


# ------------------------------------------------------------
# 10.1 CUSTOM HEX PALETTE
# ------------------------------------------------------------

custom_palette = [
    "#264653",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261"
]
sns.barplot(
    data=df,
    x="region",
    y="sales",
    hue="region",
    palette=custom_palette,
    legend=False
)
plt.title("Custom Hex Color Palette")
plt.show()


# ------------------------------------------------------------
# 10.2 DICTIONARY PALETTE
# ------------------------------------------------------------

region_palette = {
    "North": "#264653",
    "South": "#2A9D8F",
    "East": "#E9C46A",
    "West": "#E76F51"
}
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    palette=region_palette
)
plt.title("Dictionary-Based Color Mapping")
plt.show()


# ------------------------------------------------------------
# 10.3 CREATE A PALETTE USING COLOR_PALETTE()
# ------------------------------------------------------------

custom_colors = sns.color_palette(
    "Set2",
    n_colors=4
)
print("\nCUSTOM COLORS")
print(custom_colors)
sns.palplot(custom_colors)
plt.title("Generated Color Palette")
plt.show()


# ------------------------------------------------------------
# 10.4 CREATE A CONTINUOUS PALETTE
# ------------------------------------------------------------

continuous_palette = sns.color_palette(
    "viridis",
    as_cmap=True
)
sns.heatmap(
    correlation,
    cmap=continuous_palette,
    annot=True,
    center=0
)
plt.title("Continuous Colormap")
plt.show()


# ============================================================
# 11. COLOR PALETTE SELECTION
# ============================================================

# Choose a palette according to the type of data.
# Categorical values: Qualitative palette
# Ordered numerical values: Sequential palette.
# Values with positive and negative meaning: Diverging palette.

# Examples :
# Region: Qualitative, Temperature: Sequential, Correlation : Diverging, Profit: Sequential if showing magnitude.
# Profit and loss: Diverging if zero separates positive and negative values.
# Avoid using too many similar colors.
# Avoid relying on color alone to communicate meaning.
# Use labels, markers and legends when necessary.


# ============================================================
# 12. MATPLOTLIB AND SEABORN CUSTOMIZATION INTEGRATION
# ============================================================

# Seaborn handles statistical visualization.
# Matplotlib provides detailed control over: Figure size, Axes, Labels, Ticks, Spines, Annotations, Legends, Saving.
# You should be comfortable combining both libraries.


# ------------------------------------------------------------
# 12.1 CUSTOMIZE FIGURE AND AXES
# ------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(10, 6)
)
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    ax=ax
)
ax.set_title(
    "Advertising vs Sales",
    fontsize=16,
    fontweight="bold"
)
ax.set_xlabel(
    "Advertising Budget",
    fontsize=12
)
ax.set_ylabel(
    "Sales",
    fontsize=12
)
plt.tight_layout()
plt.show()


# ============================================================
# 13. TITLE AND LABEL CUSTOMIZATION
# ============================================================

# A good title should explain what the chart represents.
#
# Avoid vague titles such as:
# "Graph 1"
# "Chart"
#
# Prefer:
# "Monthly Sales by Region"
# "Correlation Between Numerical Features"
# "Feature Distribution Across Target Classes"


# ------------------------------------------------------------
# 13.1 TITLE CUSTOMIZATION
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    data=df,
    x="product",
    y="sales",
    ax=ax
)
ax.set_title(
    "Average Sales by Product",
    fontsize=16,
    fontweight="bold",
    pad=15
)
plt.show()


# ------------------------------------------------------------
# 13.2 TITLE WITH FIGURE-LEVEL SUPTITLE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    ax=ax
)
fig.suptitle(
    "Business Performance Analysis",
    fontsize=17,
    fontweight="bold"
)
ax.set_title(
    "Relationship Between Advertising and Sales",
    fontsize=12
)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 13.3 LABEL CUSTOMIZATION
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))  # Create a figure and axes with width 9 inches and height 5 inches
sns.lineplot(
    data=df,                  # Specify the DataFrame containing the data
    x="advertising",          # Use advertising values on the X-axis
    y="sales",                # Use sales values on the Y-axis
    ax=ax                     # Draw the line plot on the specified Axes object
)
ax.set_xlabel(
    "Advertising Spend (₹)",  # Set the X-axis label
    fontsize=12,              # Set the X-axis label font size to 12
    labelpad=10               # Add 10 points of padding between the label and the axis
)
ax.set_ylabel(
    "Sales Revenue (₹)",      # Set the Y-axis label
    fontsize=12,              # Set the Y-axis label font size to 12
    labelpad=10               # Add 10 points of padding between the label and the axis
)
plt.show()                    # Display the final visualization


# ============================================================
# 14. TICK CUSTOMIZATION
# ============================================================

# Ticks are the numerical or categorical labels displayed along the axes.

# You should know how to:
# - Change tick positions.
# - Change tick labels.
# - Rotate labels.
# - Change tick font size.
# - Format numerical values.
# - Set tick intervals.


# ------------------------------------------------------------
# 14.1 CHANGE TICK FONT SIZE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=df,
    x="product",
    y="sales",
    ax=ax
)

ax.tick_params(
    axis="both",
    labelsize=11
)
plt.show()


# ------------------------------------------------------------
# 14.2 ROTATE X-AXIS LABELS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.countplot(
    data=df,
    x="region",
    ax=ax
)
ax.tick_params(
    axis="x",
    labelrotation=30 # rotates 30 degrees on x-axis only
)
plt.show()


# ------------------------------------------------------------
# 14.3 FORMAT NUMERICAL TICKS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    ax=ax
)
ax.yaxis.set_major_formatter(
    FormatStrFormatter("%.0f")
)
plt.title("Sales with Integer Tick Labels")
plt.show()


# ------------------------------------------------------------
# 14.4 CUSTOM TICK INTERVALS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    ax=ax
)
ax.xaxis.set_major_locator(
    MultipleLocator(5000)
)
plt.title("Custom X-Axis Tick Interval")
plt.show()


# ------------------------------------------------------------
# 14.5 FORMAT LARGE NUMBERS
# ------------------------------------------------------------

# Define a custom formatter function to display large values in thousands (K)
# 10000 becomes 10k, 20000 becomes 20k

def thousands_formatter(value, position):
    return f"{value / 1000:.0f}K"  # Divide the value by 1000, round to 0 decimal places and append K
fig, ax = plt.subplots(figsize=(9, 5))  # Create a figure and axes with width 9 inches and height 5 inches
sns.scatterplot(
    data=df,                # Specify the DataFrame containing the data
    x="advertising",        # Use advertising values on the X-axis
    y="sales",              # Use sales values on the Y-axis
    ax=ax                   # Draw the scatter plot on the specified Axes object
)
ax.xaxis.set_major_formatter(
    FuncFormatter(thousands_formatter)  # Apply the custom thousands formatter to X-axis tick labels
)
ax.yaxis.set_major_formatter(
    FuncFormatter(thousands_formatter)  # Apply the custom thousands formatter to Y-axis tick labels
)
ax.set_xlabel("Advertising")  # Set the X-axis label
ax.set_ylabel("Sales")        # Set the Y-axis label
plt.title("Values Displayed in Thousands")  # Set the plot title
plt.show()  # Display the final visualization


# ============================================================
# 15. LEGEND CUSTOMIZATION
# ============================================================

# A legend explains the meaning of colors, markers or line styles.
# Important concepts :
# title: Legend title.
# loc: Legend position.
# bbox_to_anchor: Fine-tunes legend placement.
# frameon: Controls legend frame.
# ncol: Number of legend columns.
# fontsize: Text size.


# ------------------------------------------------------------
# 15.1 BASIC LEGEND CUSTOMIZATION
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    ax=ax
)

ax.legend(
    title="Region",
    fontsize=10,
    title_fontsize=11,
    loc="best",
    frameon=True
)
plt.show()


# ------------------------------------------------------------
# 15.2 MOVE LEGEND OUTSIDE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    ax=ax
)
ax.legend(
    title="Region",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 15.3 MULTIPLE LEGEND COLUMNS
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    ax=ax
)
ax.legend(
    title="Region",
    ncol=2,
    loc="upper left"
)
plt.show()


# ------------------------------------------------------------
# 15.4 REMOVE LEGEND
# ------------------------------------------------------------

sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    legend=False
)
plt.title("Scatter Plot Without Legend")
plt.show()


# ============================================================
# 16. SPINES AND GRID CUSTOMIZATION
# ============================================================

# Spines are the border lines around the plotting area.
# You can: Hide unnecessary spines, Keep only left and bottom spines, Adjust grid appearance.
# Clean charts often use fewer visual distractions.


# ------------------------------------------------------------
# 16.1 REMOVE TOP AND RIGHT SPINES
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    ax=ax
)
sns.despine(
    ax=ax,
    top=True,
    right=True
)
plt.title("Chart Without Top and Right Spines")
plt.show()


# ------------------------------------------------------------
# 16.2 ADJUST SPINE OFFSET
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))  # Create a figure and axes with width 9 inches and height 5 inches
sns.lineplot(
    data=df,             # Specify the DataFrame containing the data
    x="advertising",     # Use advertising values on the X-axis
    y="sales",           # Use sales values on the Y-axis
    ax=ax                # Draw the line plot on the specified Axes object
)
sns.despine(
    ax=ax,               # Apply despine to the specified Axes object
    offset=10            # Move the remaining visible spines 10 points away from the plotting area
)
plt.show()               # Display the final visualization


# ------------------------------------------------------------
# 16.3 GRID CUSTOMIZATION
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    ax=ax
)
ax.grid(
    visible=True,
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.6
)
ax.grid(
    visible=False,
    axis="x"
)
plt.title("Customized Grid")
plt.show()


# ============================================================
# 17. FIGURE AND AXES BACKGROUND
# ============================================================

# Figure background: The area surrounding all axes.
# Axes background: The actual plotting area.

fig, ax = plt.subplots(
    figsize=(9, 5),          # Set the figure width to 9 inches and height to 5 inches
    facecolor="white"        # Set the background color of the entire figure to white
)
ax.set_facecolor("#F5F5F5")  # Set the background color inside the plotting area to light gray using a HEX color code
sns.scatterplot(
    data=df,                 # Specify the DataFrame containing the data
    x="advertising",         # Use advertising values on the X-axis
    y="sales",               # Use sales values on the Y-axis
    hue="region",            # Assign different colors to points based on their region
    ax=ax                    # Draw the scatter plot on the specified Axes object
)
ax.set_title("Custom Figure and Axes Background")  # Set the title of the plot
plt.show()                   # Display the final visualization


# ============================================================
# 18. ANNOTATIONS AND REFERENCE LINES
# ============================================================

# Annotations add explanations to important points or areas in a chart.
# Reference lines can indicate: Target values, Thresholds, Averages, Limits.
# These are useful in EDA and ML evaluation.


# ------------------------------------------------------------
# 18.1 ANNOTATE A POINT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))  # Create figure and axes

sns.scatterplot(
    data=df,                    # DataFrame containing the data
    x="advertising",             # X-axis variable
    y="sales",                   # Y-axis variable
    ax=ax                        # Draw plot on the specified axes
)

# Find the observation with the highest sales
highest_index = df["sales"].idxmax()  # Get index of maximum sales value
highest_row = df.loc[highest_index]   # Retrieve the complete row at that index

# Add an annotation to highlight that observation
ax.annotate(
    "Highest Sales",             # Text displayed near the selected point
    xy=(                         # Actual location of the point to annotate
        highest_row["advertising"],
        highest_row["sales"]
    ),
    xytext=(20, 20),             # Position of annotation text relative to the point
    textcoords="offset points",  # Interpret xytext as an offset in points
    arrowprops={                 # Customize the arrow connecting text and point
        "arrowstyle": "->"        # Draw an arrow pointing towards the observation
    }
)
plt.title("Important Observation Annotation")  # Set plot title
plt.show()                                      # Display the plot


# ------------------------------------------------------------
# 18.2 HORIZONTAL REFERENCE LINE
# ------------------------------------------------------------

average_sales = df["sales"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    ax=ax
)
ax.axhline(
    average_sales,
    color="red",
    linestyle="--",
    label="Average Sales"
)
ax.legend()
plt.title("Average Sales Reference Line")
plt.show()


# ------------------------------------------------------------
# 18.3 VERTICAL REFERENCE LINE
# ------------------------------------------------------------

average_advertising = df["advertising"].mean()
fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    ax=ax

ax.axvline(
    average_advertising,
    color="green",
    linestyle="--",
    label="Average Advertising"

ax.legend()
plt.title("Average Advertising Reference Line")
plt.show()


# ============================================================
# 19. CUSTOMIZE SUBPLOTS
# ============================================================

# Subplots allow multiple charts inside one figure.
# Each Axes can have its own:
# - Title.
# - Labels.
# - Limits.
# - Legend.
# - Grid.
# - Ticks.
# Keep the styling consistent across related plots.


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Advertising vs Sales")


sns.boxplot(
    data=df,
    x="product",
    y="profit",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Profit by Product")


sns.barplot(
    data=df,
    x="region",
    y="sales",
    ax=axes[1, 0]
)
axes[1, 0].set_title("Average Sales by Region")

sns.histplot(
    data=df,
    x="profit",
    kde=True,
    ax=axes[1, 1]
)

axes[1, 1].set_title("Profit Distribution")
fig.suptitle(
    "Business Performance Dashboard",
    fontsize=17,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# ============================================================
# 20. REUSABLE VISUALIZATION FUNCTIONS
# ============================================================

# When building multiple charts, avoid repeating the same customization code.
# Create reusable functions instead.
# This helps maintain consistency and cleaner code.


# ------------------------------------------------------------
# 20.1 REUSABLE SCATTER PLOT FUNCTION
# ------------------------------------------------------------

def create_scatter_plot(
    data,
    x,
    y,
    hue=None,
    title=None,
    xlabel=None,
    ylabel=None,
    palette="deep",
    figsize=(9, 5)
):
    fig, ax = plt.subplots(
        figsize=figsize
    )
ns.scatterplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        ax=ax
    )
    ax.set_title(
        title if title else f"{x} vs {y}",
        fontsize=14,
        fontweight="bold"
    )
    ax.set_xlabel(
        xlabel if xlabel else x
    )
    ax.set_ylabel(
        ylabel if ylabel else y
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig, ax


# Use the reusable function.

fig, ax = create_scatter_plot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    title="Advertising vs Sales by Region",
    xlabel="Advertising Budget",
    ylabel="Sales Revenue"
)
plt.show()


# ------------------------------------------------------------
# 20.2 REUSABLE BAR PLOT FUNCTION
# ------------------------------------------------------------

def create_bar_plot(
    data,
    x,
    y,
    hue=None,
    title=None,
    palette="deep"
):
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        ax=ax
    )
    ax.set_title(
        title if title else f"{y} by {x}",
        fontsize=14,
        fontweight="bold"
    )
    sns.despine(ax=ax)
    plt.tight_layout()
    return fig, ax


fig, ax = create_bar_plot(
    data=df,
    x="region",
    y="sales",
    title="Average Sales by Region"
)
plt.show()


# ============================================================
# 21. COLOR ACCESSIBILITY AND VISUAL CLARITY
# ============================================================

# Good visualization design should be accessible.

# Best practices:
# 1. Use distinguishable colors.
# 2. Avoid relying only on color to identify categories.
# 3. Use markers and line styles when appropriate.
# 4. Use readable text sizes.
# 5. Maintain sufficient contrast.
# 6. Avoid unnecessary decorative effects.
# 7. Use colorblind-friendly palettes when suitable.
# 8. Avoid too many categories in one visualization.
# 9. Use labels and legends that explain the chart.
# 10. Keep colors consistent across related charts.


# Example: use both hue and style.

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    style="region",
    palette="colorblind",
    ax=ax
)
ax.set_title(
    "Accessible Group Comparison"
)
plt.tight_layout()
plt.show()


# ============================================================
# 22. PROFESSIONAL VISUALIZATION STANDARDS
# ============================================================

# Before presenting a visualization, check:

# DATA:
# - Is the correct data displayed?
# - Are missing values handled appropriately?
# - Is the chart appropriate for the variable types?

# DESIGN:
# - Is the title meaningful?
# - Are axis labels clear?
# - Are units included?
# - Are colors appropriate?
# - Is the legend understandable?

# READABILITY:
# - Are tick labels readable?
# - Is the chart overcrowded?
# - Is the figure size appropriate?
# - Is the text large enough?

# INTERPRETATION:
# - Does the visualization communicate the intended point?
# - Are important observations highlighted?
# - Are limitations explained?
#
# ACCURACY:
# - Are axes misleading?
# - Are comparisons fair?
# - Are confidence intervals interpreted correctly?
# - Are conclusions supported by the data?


# ============================================================
# 23. SAVING HIGH-QUALITY VISUALIZATIONS
# ============================================================

# You already learned the basics of savefig().
# This section focuses on selecting the correct output format and quality.

# Common formats:
# PNG: Good for general reports and presentations.
# JPG: Suitable for photographs and smaller file sizes, but it uses lossy compression.
# PDF: Good for documents and vector-based charts.
# SVG: Useful for scalable vector graphics and editing.
# DPI: Controls raster image resolution.
# bbox_inches="tight": Helps remove unnecessary surrounding whitespace.
# transparent=True: Saves with a transparent background when supported.


# ------------------------------------------------------------
# 23.1 SAVE PNG
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    ax=ax
)
ax.set_title("Advertising vs Sales")
plt.tight_layout()
fig.savefig(
    "advertising_sales.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 23.2 SAVE PDF
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5)
sns.boxplot(
    data=df,
    x="product",
    y="profit",
    ax=ax
)
ax.set_title("Profit by Product")
plt.tight_layout()
fig.savefig(
    "profit_by_product.pdf",
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 23.3 SAVE SVG
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.lineplot(
    data=df,
    x="advertising",
    y="sales",
    ax=ax
ax.set_title("Sales Trend")
plt.tight_layout()
fig.savefig(
    "sales_trend.svg",
    bbox_inches="tight"
)
plt.show()


# ------------------------------------------------------------
# 23.4 SAVE TRANSPARENT PNG
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    ax=ax
)
fig.savefig(
    "transparent_chart.png",
    dpi=300,
    transparent=True,
    bbox_inches="tight"
)
plt.show()  


# ============================================================
# 24. PROFESSIONAL EDA DASHBOARD
# ============================================================

# Combine the customization concepts into one dashboard.
# Objective: Create a clean and consistent business EDA dashboard.

# Include:
# 1. Sales distribution.
# 2. Sales by product.
# 3. Sales by region.
# 4. Advertising vs sales.
# 5. Correlation heatmap.


sns.set_theme(
    style="whitegrid",
    context="notebook",
    palette="colorblind",
    font_scale=1.0
)   
fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 11)
)

# 1. Sales distribution
sns.histplot(
    data=df,
    x="sales",
    kde=True,
    color="#264653",
    ax=axes[0, 0]
)
axes[0, 0].set_title(
    "Sales Distribution",
    fontweight="bold"
)

# 2. Sales by product
sns.boxplot(
    data=df,
    x="product",
    y="sales",
    palette="Set2",
    hue="product",
    legend=False,
    ax=axes[0, 1]
)
axes[0, 1].set_title(
    "Sales by Product",
    fontweight="bold"
)

# 3. Sales by region
sns.barplot(
    data=df,
    x="region",
    y="sales",
    hue="region",
    palette="colorblind",
    legend=False,
    ax=axes[1, 0]
)
axes[1, 0].set_title(
    "Average Sales by Region",
    fontweight="bold"
)


# 4. Advertising and sales relationship

sns.scatterplot(
    data=df,
    x="advertising",
    y="sales",
    hue="region",
    palette="colorblind",
    alpha=0.7,
    ax=axes[1, 1]
)
axes[1, 1].set_title(
    "Advertising vs Sales",
    fontweight="bold"
)
fig.suptitle(
    "Business Exploratory Data Analysis",
    fontsize=18,
    fontweight="bold"
)
sns.despine(fig=fig)
plt.tight_layout()
fig.savefig(
    "professional_eda_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ============================================================
# 25. ML ENGINEERING APPLICATIONS
# ============================================================

# Visualization customization is useful when preparing:

# 1. EDA reports.
# 2. Model evaluation reports.
# 3. Feature analysis dashboards.
# 4. Model comparison charts.
# 5. Training and validation plots.
# 6. Classification performance reports.
# 7. Regression diagnostic reports.

# Examples of good visualization choices:

# Class distribution: Use clear categorical colors.
# Correlation matrix: Use a diverging palette centered at zero.
# Feature distribution: Use readable sequential or neutral colors.
# Actual vs predicted: Use clear markers and a reference line.
# Confusion matrix: Use an annotated heatmap with readable labels.
# Learning curves: Use distinguishable lines and a clear legend.
# Residual analysis: Include a zero reference line.

# Do not confuse visual design with model quality.
# A beautiful chart does not make an analysis correct.


# ============================================================
# 26. COMMON MISTAKES TO AVOID
# ============================================================

# 1. Using too many colors.

# 2. Using a qualitative palette for continuous numerical
#    values without considering whether it is appropriate.

# 3. Using a sequential palette for unrelated categories.
# 4. Using a diverging palette without a meaningful center.
# 5. Making fonts too small.
# 6. Using vague chart titles.
# 7. Hiding important legends.
# 8. Adding unnecessary grid lines.
# 9. Using excessive annotations.
# 10. Changing category colors across related charts.
# 11. Using misleading axis limits.
# 12. Saving low-resolution images for print.
# 13. Relying only on color for category identification.
# 14. Applying a dark theme to every chart without considering the purpose and readability.
# 15. Making a dashboard overcrowded.
# 16. Forgetting that set_theme(), set_style() and set_context() modify visualization settings globally.


# ============================================================
# 27. PRACTICE TASKS
# ============================================================

# TASK 1:
# Apply darkgrid, whitegrid, white and ticks styles.

# TASK 2:
# Compare paper, notebook, talk and poster contexts.

# TASK 3:
# Create a custom theme using set_theme().

# TASK 4:
# Experiment with rc parameters.

# TASK 5:
# Use qualitative palettes for regions.

# TASK 6:
# Use sequential palettes for numerical values.

# TASK 7:
# Use diverging palettes for correlations.

# TASK 8:
# Create a custom palette using hexadecimal colors.

# TASK 9:
# Create a dictionary palette for categories.

# TASK 10:
# Customize titles and labels.

# TASK 11:
# Format tick labels in thousands.

# TASK 12:
# Move a legend outside the plot.

# TASK 13:
# Remove unnecessary spines.

# TASK 14:
# Customize grid lines.

# TASK 15:
# Add annotations and reference lines.

# TASK 16:
# Create a consistent 2 x 2 dashboard.

# TASK 17:
# Write a reusable scatter plot function.

# TASK 18:
# Save a chart in PNG, PDF and SVG formats.

# TASK 19:
# Create a professional EDA dashboard.

# TASK 20:
# Review your charts for accessibility and readability.


# ============================================================
# 28. FINAL PHASE 5 REVISION CHECKLIST
# ============================================================

# THEMES:

# [ ] set_theme()
# [ ] set_style()
# [ ] set_context()
# [ ] Difference between theme, style and context.
# [ ] Custom rc parameters.

# PALETTES:

# [ ] Qualitative palettes.
# [ ] Sequential palettes.
# [ ] Diverging palettes.
# [ ] Custom palettes.
# [ ] Hexadecimal colors.
# [ ] Dictionary palettes.
# [ ] Colorblind-friendly palettes.

# CUSTOMIZATION:

# [ ] Figure and Axes customization.
# [ ] Titles and subtitles.
# [ ] Axis labels.
# [ ] Tick positions and formatting.
# [ ] Legend positioning.
# [ ] Spines and grids.
# [ ] Figure and Axes backgrounds.
# [ ] Annotations and reference lines.
# [ ] Subplot consistency.

# REUSABILITY:

# [ ] Reusable visualization functions.
# [ ] Consistent visualization settings.
# [ ] Dashboard design.

# EXPORT:

# [ ] PNG.
# [ ] PDF.
# [ ] SVG.
# [ ] DPI.
# [ ] Transparent background.
# [ ] Tight bounding box.

# PROFESSIONAL PRACTICE:

# [ ] Appropriate color selection.
# [ ] Accessible charts.
# [ ] Readable labels.
# [ ] Clear titles.
# [ ] Correct interpretation.
# [ ] Avoid unnecessary decoration.


# ============================================================
# 29. WHAT YOU SHOULD BE ABLE TO DO AFTER PHASE 5
# ============================================================

# You should now be able to:

# 1. Apply a consistent theme to your visualizations.
# 2. Choose suitable color palettes based on data type.
# 3. Customize titles, labels, ticks and legends.
# 4. Control backgrounds, grids and spines.
# 5. Add annotations and reference lines.
# 6. Create reusable plotting functions.
# 7. Build readable EDA dashboards.
# 8. Export high-quality visualizations.
# 9. Prepare charts suitable for technical reports.
# 10. Maintain consistent visual design across charts.

# ============================================================
# END OF SEABORN PHASE 5
# ============================================================

