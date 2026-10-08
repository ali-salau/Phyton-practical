# ==========================================
# WEEK 11: DATA ANALYSIS
# EXPLORATORY DATA ANALYSIS ON IRIS DATASET
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------

df = pd.read_csv("Iris.csv")

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nDATASET INFORMATION")
df.info()


# ------------------------------------------
# 2. CHECK FOR MISSING VALUES
# ------------------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())


# ------------------------------------------
# 3. CHECK FOR DUPLICATE ROWS
# ------------------------------------------

print("\nNUMBER OF DUPLICATE ROWS")
print(df.duplicated().sum())


# ------------------------------------------
# 4. SUMMARY STATISTICS
# ------------------------------------------

print("\nSUMMARY STATISTICS")
print(df.describe())


# ------------------------------------------
# 5. SPECIES COUNT
# ------------------------------------------

print("\nNUMBER OF FLOWERS IN EACH SPECIES")
print(df["Species"].value_counts())


# ------------------------------------------
# 6. AVERAGE MEASUREMENTS BY SPECIES
# ------------------------------------------

print("\nAVERAGE MEASUREMENTS BY SPECIES")

print(
    df.groupby("Species")[
        [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    ].mean()
)


# ------------------------------------------
# 7. CORRELATION MATRIX
# ------------------------------------------

print("\nCORRELATION MATRIX")

correlation = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
].corr()

print(correlation)


# ------------------------------------------
# 8. HISTOGRAMS
# ------------------------------------------

df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
].hist(figsize=(10, 8))

plt.suptitle("Distribution of Iris Measurements")
plt.tight_layout()
plt.show()


# ------------------------------------------
# 9. BOXPLOT
# ------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df[
        [
            "SepalLengthCm",
            "SepalWidthCm",
            "PetalLengthCm",
            "PetalWidthCm"
        ]
    ]
)

plt.title("Boxplot of Iris Measurements")
plt.ylabel("Measurement (cm)")
plt.show()


# ------------------------------------------
# 10. SCATTER PLOT
# ------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="PetalLengthCm",
    hue="Species"
)

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()


# ------------------------------------------
# 11. PETAL LENGTH VS PETAL WIDTH
# ------------------------------------------

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species"
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()


# ------------------------------------------
# 12. CORRELATION HEATMAP
# ------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.show()