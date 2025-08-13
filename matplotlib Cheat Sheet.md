# Matplotlib Comprehensive Cheatsheet for Data Professionals

**Matplotlib** is the foundational plotting library for Python data visualization. It provides low-level control over every aspect of a figure, making it essential for creating publication-quality plots, dashboards, and custom visualizations.

**When to Use Matplotlib:**

- Creating custom, publication-ready visualizations
- Building interactive dashboards and applications
- Fine-grained control over plot appearance
- Base layer for other visualization libraries (Seaborn, Pandas plotting)
- Scientific computing and research presentations
- Web applications requiring embedded plots

## Table of Contents

1. [Essential Imports & Setup](#essential-imports--setup)
2. [Figure & Axes Basics](#figure--axes-basics)
3. [Basic Plot Types](#basic-plot-types)
4. [Customization Essentials](#customization-essentials)
5. [Advanced Plot Types](#advanced-plot-types)
6. [Subplots & Layouts](#subplots--layouts)
7. [Styling & Themes](#styling--themes)
8. [Statistical Plots](#statistical-plots)
9. [Time Series Plotting](#time-series-plotting)
10. [3D Plotting](#3d-plotting)
11. [Interactive Features](#interactive-features)
12. [Saving & Export](#saving--export)
13. [Performance Tips](#performance-tips)
14. [Common Use Cases](#common-use-cases)

---

## Essential Imports & Setup

**Description:** Core imports and configuration needed for matplotlib functionality.
**Usage:** Set up at the beginning of any data analysis script or notebook.

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib.patches import Rectangle, Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns  # Optional but recommended

# Common settings
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
%matplotlib inline  # For Jupyter notebooks

# Backend settings for different environments
# %matplotlib widget    # For interactive plots in JupyterLab
# plt.ioff()           # Turn off interactive mode for scripts
```

---

## Figure & Axes Basics

**Description:** Understanding the matplotlib object hierarchy - Figure (canvas) contains Axes (plot area).
**Usage:** Foundation for all plotting - choose pyplot (simple) vs object-oriented (flexible) approach.
**When to Use:** Object-oriented for complex plots, multiple subplots, or when building reusable functions.

### Creating Figures

```python
# Method 1: pyplot interface (simple, good for quick plots)
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.show()

# Method 2: Object-oriented interface (recommended for complex plots)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y)
plt.show()

# Multiple subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
```

### Figure Properties

```python
fig = plt.figure(
    figsize=(12, 8),           # Width, height in inches
    dpi=100,                   # Dots per inch (resolution)
    facecolor='white',         # Background color
    edgecolor='black',         # Border color
    tight_layout=True          # Automatic layout adjustment
)
```

---

## Basic Plot Types

### Line Plots

**Description:** Shows relationships and trends over continuous data.
**Usage:** Time series, trend analysis, function plotting, connecting data points.
**When to Use:** Continuous data, showing trends over time, comparing multiple series.

```python
# Basic line plot
ax.plot(x, y,
        color='blue',          # Color
        linewidth=2,           # Line thickness
        linestyle='-',         # Line style: '-', '--', '-.', ':'
        marker='o',            # Marker style
        markersize=5,          # Marker size
        alpha=0.7,             # Transparency
        label='Data Series')   # Legend label

# Multiple lines
ax.plot(x, y1, label='Series 1')
ax.plot(x, y2, label='Series 2')
ax.legend()

# Common line styles
styles = ['-', '--', '-.', ':', 'None']
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
```

### Scatter Plots

**Description:** Shows relationships between two continuous variables.
**Usage:** Correlation analysis, outlier detection, pattern recognition.
**When to Use:** Exploring relationships, showing clusters, identifying outliers.

```python
ax.scatter(x, y,
           s=50,               # Size (can be array for variable sizes)
           c=colors,           # Color (can be array for color mapping)
           alpha=0.6,          # Transparency
           cmap='viridis',     # Colormap
           edgecolors='black', # Edge color
           linewidth=0.5)      # Edge width

# Color by value (bubble chart)
scatter = ax.scatter(x, y, c=z, s=sizes, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Color Scale')

# Categories with different colors
for category in categories:
    mask = data['category'] == category
    ax.scatter(data[mask]['x'], data[mask]['y'], label=category)
```

### Bar Plots

**Description:** Compares quantities across categories.
**Usage:** Categorical data comparison, frequency distributions, rankings.
**When to Use:** Comparing categories, showing frequency counts, discrete data.

```python
# Vertical bars
ax.bar(x, height,
       width=0.8,             # Bar width
       color='skyblue',       # Color
       edgecolor='black',     # Edge color
       alpha=0.7,             # Transparency
       label='Data')          # Legend label

# Horizontal bars
ax.barh(y, width, color='lightcoral')

# Grouped bars (comparing multiple series)
x = np.arange(len(labels))
width = 0.35
ax.bar(x - width/2, values1, width, label='Group 1')
ax.bar(x + width/2, values2, width, label='Group 2')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# Stacked bars (showing composition)
ax.bar(x, values1, label='Bottom Layer')
ax.bar(x, values2, bottom=values1, label='Top Layer')

# Error bars
ax.bar(x, height, yerr=errors, capsize=5)
```

### Histograms

**Description:** Shows distribution of continuous data by binning values.
**Usage:** Data distribution analysis, probability density estimation.
**When to Use:** Understanding data distribution, detecting skewness, outliers.

```python
ax.hist(data,
        bins=30,              # Number of bins or bin edges
        density=True,         # Normalize to probability density
        alpha=0.7,            # Transparency
        color='skyblue',      # Color
        edgecolor='black',    # Edge color
        cumulative=False)     # Cumulative histogram

# Multiple histograms
ax.hist([data1, data2], bins=20, alpha=0.7, label=['Data 1', 'Data 2'])

# Custom bin edges
bins = np.linspace(data.min(), data.max(), 25)
ax.hist(data, bins=bins)

# 2D histogram (heatmap)
plt.hist2d(x, y, bins=30, cmap='Blues')
plt.colorbar(label='Frequency')
```

---

## Customization Essentials

**Description:** Core styling and labeling options for professional plots.
**Usage:** Making plots publication-ready, adding context, improving readability.

### Labels and Titles

```python
ax.set_title('Plot Title', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('X-axis Label', fontsize=12)
ax.set_ylabel('Y-axis Label', fontsize=12)

# Title positioning
ax.set_title('Title', loc='left')  # 'left', 'center', 'right'

# Multiple line title
ax.set_title('Main Title\nSubtitle', fontsize=14)
```

### Legends

```python
ax.legend(
    loc='best',              # Location: 'best', 'upper right', etc.
    frameon=True,            # Show frame
    fancybox=True,           # Rounded corners
    shadow=True,             # Drop shadow
    ncol=2,                  # Number of columns
    fontsize=10,             # Font size
    title='Legend Title'     # Legend title
)

# Custom legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='blue', lw=2)]
ax.legend(custom_lines, ['Label 1', 'Label 2'])
```

### Axis Formatting

```python
# Axis limits
ax.set_xlim(0, 100)
ax.set_ylim(-10, 10)

# Axis scales
ax.set_xscale('log')        # 'linear', 'log', 'symlog', 'logit'
ax.set_yscale('log')

# Tick formatting
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.set_xticks(np.arange(0, 101, 10))
ax.set_xticklabels(['Label' + str(i) for i in range(11)])

# Grid
ax.grid(True, alpha=0.3, linestyle='--')
ax.grid(True, which='major', alpha=0.5)
ax.grid(True, which='minor', alpha=0.2)
```

### Colors and Styles

```python
# Color options
colors = ['red', 'blue', 'green', '#FF5733', (0.1, 0.2, 0.5)]

# Colormaps
cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'seismic']

# Style sheets
plt.style.use('seaborn-v0_8')  # Built-in styles
available_styles = plt.style.available
```

---

## Advanced Plot Types

### Box Plots

**Description:** Shows distribution summary with quartiles, median, and outliers.
**Usage:** Comparing distributions across groups, outlier detection.
**When to Use:** Statistical analysis, comparing multiple groups, showing data spread.

```python
# Basic box plot
ax.boxplot(data, labels=['Group 1', 'Group 2'])

# Multiple groups
data_groups = [group1_data, group2_data, group3_data]
ax.boxplot(data_groups, labels=['A', 'B', 'C'])

# Customization
box_plot = ax.boxplot(data,
                      notch=True,          # Notched boxes
                      patch_artist=True,   # Fill boxes
                      showmeans=True,      # Show means
                      meanline=True)       # Mean as line
```

### Heatmaps

**Description:** 2D representation of data through colors.
**Usage:** Correlation matrices, pivot tables, geographic data.
**When to Use:** Showing relationships in 2D data, correlation analysis.

```python
# Basic heatmap
im = ax.imshow(data, cmap='viridis', aspect='auto')
plt.colorbar(im)

# With annotations
for i in range(len(data)):
    for j in range(len(data[0])):
        ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center')

# Seaborn-style heatmap in matplotlib
import numpy as np
correlation_matrix = np.corrcoef(data.T)
im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
```

### Violin Plots

**Description:** Combines box plot with kernel density estimation.
**Usage:** Detailed distribution visualization, comparing groups.
**When to Use:** When you need more detail than box plots provide.

```python
violin_parts = ax.violinplot(data_groups, positions=range(1, len(data_groups)+1))

# Customization
for pc in violin_parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)
```

---

## Subplots & Layouts

**Description:** Creating multiple plots in a single figure.
**Usage:** Comparing related data, dashboard creation, multi-faceted analysis.
**When to Use:** Multiple related visualizations, before/after comparisons, different views of same data.

### Basic Subplots

```python
# Grid layout
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Different subplot sizes
fig = plt.figure(figsize=(12, 8))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 1, 2)  # Spans bottom row

# GridSpec for complex layouts
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])    # Top row, all columns
ax2 = fig.add_subplot(gs[1, :-1])  # Middle row, first two columns
ax3 = fig.add_subplot(gs[1:, -1])  # Right column, bottom two rows
```

### Sharing Axes

```python
# Shared x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Shared y-axis
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
```

---

## Styling & Themes

**Description:** Consistent visual appearance across plots.
**Usage:** Brand consistency, publication standards, aesthetic improvement.

### Built-in Styles

```python
# Available styles
print(plt.style.available)

# Apply style
plt.style.use('seaborn-v0_8-darkgrid')
plt.style.use(['seaborn-v0_8-paper', 'seaborn-v0_8-whitegrid'])

# Reset to default
plt.style.use('default')
```

### Custom Styling

```python
# Custom rcParams
custom_params = {
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3
}
plt.rcParams.update(custom_params)
```

---

## Statistical Plots

**Description:** Specialized plots for statistical analysis.
**Usage:** Hypothesis testing, regression analysis, confidence intervals.

### Error Bars

```python
# Symmetric error bars
ax.errorbar(x, y, yerr=errors, fmt='o-', capsize=5, capthick=2)

# Asymmetric error bars
ax.errorbar(x, y, yerr=[lower_errors, upper_errors], fmt='s-')

# Error bars on bar plots
ax.bar(x, height, yerr=errors, capsize=5)
```

### Regression Lines

```python
# Linear regression line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
ax.plot(x, p(x), 'r--', alpha=0.8, label='Trend Line')

# Confidence intervals
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
line = slope * x + intercept
ax.plot(x, line, 'r-', label=f'R² = {r_value**2:.3f}')
```

---

## Time Series Plotting

**Description:** Specialized formatting for temporal data.
**Usage:** Financial data, sensor data, trend analysis over time.
**When to Use:** Any data with datetime index, trend analysis, seasonality detection.

```python
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Date formatting
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_minor_locator(mdates.WeekdayLocator())

# Rotate date labels
plt.xticks(rotation=45)

# Time series with pandas
df.plot(x='date', y='value', ax=ax)

# Multiple time series
for column in df.columns[1:]:
    ax.plot(df['date'], df[column], label=column)
```

---

## 3D Plotting

**Description:** Three-dimensional visualizations.
**Usage:** Surface plots, 3D scatter plots, scientific visualization.
**When to Use:** 3D data, mathematical functions, scientific applications.

```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D scatter
ax.scatter(x, y, z, c=colors, cmap='viridis')

# 3D surface
X, Y = np.meshgrid(x, y)
Z = function(X, Y)
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

# 3D line
ax.plot(x, y, z, '-o')

# Labels
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
```

---

## Interactive Features

**Description:** Adding interactivity to plots.
**Usage:** Exploratory data analysis, presentations, web applications.

```python
# Zoom and pan (automatic in most backends)
# For Jupyter notebooks
%matplotlib widget

# Click events
def on_click(event):
    if event.inaxes:
        print(f'Clicked at x={event.xdata:.2f}, y={event.ydata:.2f}')

fig.canvas.mpl_connect('button_press_event', on_click)

# Animations
from matplotlib.animation import FuncAnimation

def animate(frame):
    ax.clear()
    ax.plot(x[:frame], y[:frame])

ani = FuncAnimation(fig, animate, frames=len(x), interval=50, repeat=True)
```

---

## Saving & Export

**Description:** Exporting plots for various uses.
**Usage:** Reports, presentations, web publishing, print media.

```python
# Save figure
plt.savefig('plot.png',
            dpi=300,              # High resolution
            bbox_inches='tight',   # Tight bounding box
            facecolor='white',     # Background color
            edgecolor='none',      # No edge
            transparent=False,     # Transparent background
            format='png')          # Format: png, pdf, svg, eps

# Multiple formats
formats = ['png', 'pdf', 'svg']
for fmt in formats:
    plt.savefig(f'plot.{fmt}', dpi=300, bbox_inches='tight')

# Size control
plt.savefig('plot.png', figsize=(10, 6), dpi=150)
```

---

## Performance Tips

**Description:** Optimizing matplotlib for large datasets and better performance.
**Usage:** Large datasets, real-time plotting, memory optimization.

```python
# Use collections for many similar objects
from matplotlib.collections import LineCollection
lines = LineCollection(segments, colors=colors, linewidths=2)
ax.add_collection(lines)

# Reduce marker complexity
ax.plot(x, y, marker='.', markersize=1)  # Small dots instead of circles

# Turn off anti-aliasing for speed
ax.plot(x, y, rasterized=True)

# Use appropriate backends
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for scripts

# Batch updates
with plt.ioff():  # Turn off interactive mode
    # Multiple plot operations
    pass
plt.show()
```

---

## Common Use Cases

### Data Analysis Dashboard

```python
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Data Analysis Dashboard', fontsize=16)

# Time series
axes[0,0].plot(dates, values)
axes[0,0].set_title('Trend Over Time')

# Distribution
axes[0,1].hist(data, bins=30, alpha=0.7)
axes[0,1].set_title('Distribution')

# Correlation
axes[1,0].scatter(x, y, alpha=0.6)
axes[1,0].set_title('Correlation Analysis')

# Categories
axes[1,1].bar(categories, counts)
axes[1,1].set_title('Category Comparison')

plt.tight_layout()
```

### Statistical Report

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Box plots for group comparison
axes[0,0].boxplot([group1, group2, group3], labels=['A', 'B', 'C'])
axes[0,0].set_title('Group Comparison')

# Regression analysis
axes[0,1].scatter(x, y, alpha=0.6)
axes[0,1].plot(x, regression_line, 'r-', alpha=0.8)
axes[0,1].set_title('Regression Analysis')

# Error bars
axes[0,2].errorbar(x, means, yerr=std_errors, fmt='o-', capsize=5)
axes[0,2].set_title('Means with Error Bars')

# Continue with additional statistical plots...
```

### Publication Quality Figure

```python
# Set publication style
plt.rcParams.update({
    'font.size': 12,
    'axes.linewidth': 1.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.major.size': 5,
    'ytick.minor.size': 3
})

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, 'o-', linewidth=2, markersize=6, color='#2E8B57')
ax.set_xlabel('X Variable (units)', fontsize=14)
ax.set_ylabel('Y Variable (units)', fontsize=14)
ax.set_title('Publication Title', fontsize=16, pad=20)
ax.grid(True, alpha=0.3)

plt.savefig('publication_figure.pdf', dpi=300, bbox_inches='tight')
```

---

## Quick Reference

### Essential Functions

- `plt.plot()` - Line plots
- `plt.scatter()` - Scatter plots
- `plt.bar()` - Bar charts
- `plt.hist()` - Histograms
- `plt.boxplot()` - Box plots
- `plt.subplots()` - Multiple plots
- `plt.savefig()` - Save plots

### Key Parameters

- `figsize=(width, height)` - Figure size in inches
- `alpha=0.7` - Transparency (0-1)
- `color='blue'` - Color specification
- `label='Legend'` - Legend labels
- `linewidth=2` - Line thickness
- `marker='o'` - Data point markers
- `dpi=300` - Resolution for saving

### Best Practices

1. Use object-oriented interface for complex plots
2. Always label axes and add titles
3. Include legends for multiple series
4. Use appropriate plot types for data
5. Consider colorblind-friendly palettes
6. Save in vector formats (PDF, SVG) for publications
7. Use consistent styling across related plots
8. Optimize performance for large datasets
