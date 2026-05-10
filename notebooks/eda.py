import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/heart.csv")

# Basic info
print(df.head())
print(df.info())

# Missing values
print(df.isnull().sum())

# Histogram
df.hist(figsize=(12,10))
plt.suptitle("Feature Distributions")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Target Distribution
sns.countplot(x='num', data=df)
plt.title("Target Distribution")
plt.show()

# Boxplot for outliers
plt.figure(figsize=(12,6))
sns.boxplot(data=df)
plt.title("Outlier Detection")
plt.show()