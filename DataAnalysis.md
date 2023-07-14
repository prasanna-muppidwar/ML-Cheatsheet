**Data Analytics Cheatsheet: Matplotlib and Seaborn**

This cheatsheet provides an overview of Matplotlib and Seaborn, two popular data visualization libraries in Python. It includes important syntax, code examples, and key information to help you create visually appealing and informative plots.

## Matplotlib:

### Line Plot:
- **Use Cases:**
  - Visualizing trends and changes over continuous data.

**Code:**
```python
import matplotlib.pyplot as plt

# Create a line plot
plt.plot(x, y)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True)
plt.legend(["Legend"])
plt.show()
```

### Bar Plot:
- **Use Cases:**
  - Comparing categories or discrete data.

**Code:**
```python
import matplotlib.pyplot as plt

# Create a bar plot
plt.bar(x, y)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.xticks(rotation=90)
plt.show()
```

### Histogram:
- **Use Cases:**
  - Analyzing the distribution of continuous data.

**Code:**
```python
import matplotlib.pyplot as plt

# Create a histogram
plt.hist(data, bins=10)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

### Scatter Plot:
- **Use Cases:**
  - Visualizing the relationship between two continuous variables.

**Code:**
```python
import matplotlib.pyplot as plt

# Create a scatter plot
plt.scatter(x, y)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

### Box Plot:
- **Use Cases:**
  - Displaying the distribution of data across categories.

**Code:**
```python
import matplotlib.pyplot as plt

# Create a box plot
plt.boxplot(data)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

## Seaborn:

### Line Plot:
- **Use Cases:**
  - Visualizing trends and changes over continuous data.
  - Comparing multiple groups or categories.

**Code:**
```python
import seaborn as sns

# Create a line plot
sns.lineplot(x=x, y=y, hue="category", data=df)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

### Bar Plot:
- **Use Cases:**
  - Comparing categories or discrete data.
  - Showing aggregated values.

**Code:**
```python
import seaborn as sns

# Create a bar plot
sns.barplot(x=x, y=y, hue="category", data=df)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

### Histogram:
- **Use Cases:**
  - Analyzing the distribution of continuous data.

**Code:**
```python
import seaborn as sns

# Create a histogram
sns.histplot(data, bins=10)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

### Scatter Plot:
- **Use Cases:**
  - Visualizing the relationship between two continuous variables.
  - Highlighting patterns or clusters.

**Code:**
```python
import seaborn as sns

# Create a scatter plot
sns.scatterplot(x=x, y=y, hue="category", data=df)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

### Box Plot:
- **Use Cases:**
  - Displaying the distribution of data across categories.
  - Comparing groups or categories.

**Code:**
```python
import seaborn as sns

# Create a box plot
sns.boxplot(x=x, y=y, hue="category", data=df)

# Customize the plot
plt.title("Title")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
```

This cheatsheet provides a quick reference for creating various types of plots using Matplotlib and Seaborn. Remember to import the necessary libraries, customize the plots according to your requirements, and explore the additional functionalities and options available in these libraries. Happy visualizing!