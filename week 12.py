# ==========================================
# WEEK 12: DECISION TREE CLASSIFICATION
# IRIS DATASET
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ------------------------------------------
# 1. LOAD THE DATASET
# ------------------------------------------

df = pd.read_csv("Iris.csv")


# ------------------------------------------
# 2. SELECT FEATURES
# ------------------------------------------

X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]


# ------------------------------------------
# 3. SELECT TARGET
# ------------------------------------------

y = df["Species"]


# ------------------------------------------
# 4. SPLIT DATA INTO TRAINING AND TESTING
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ------------------------------------------
# 5. CREATE THE DECISION TREE
# ------------------------------------------

model = DecisionTreeClassifier(
    random_state=42
)


# ------------------------------------------
# 6. TRAIN THE MODEL
# ------------------------------------------

model.fit(X_train, y_train)


# ------------------------------------------
# 7. MAKE PREDICTIONS
# ------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------
# 8. CALCULATE ACCURACY
# ------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nACCURACY")
print(accuracy)


# ------------------------------------------
# 9. CALCULATE PRECISION
# ------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nPRECISION")
print(precision)


# ------------------------------------------
# 10. CALCULATE RECALL
# ------------------------------------------

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nRECALL")
print(recall)


# ------------------------------------------
# 11. CLASSIFICATION REPORT
# ------------------------------------------

print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ------------------------------------------
# 12. CONFUSION MATRIX
# ------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nCONFUSION MATRIX")
print(cm)


# ------------------------------------------
# 13. DISPLAY CONFUSION MATRIX
# ------------------------------------------

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

display.plot()

plt.title("Decision Tree Confusion Matrix")
plt.show()


# ------------------------------------------
# 14. VISUALIZE THE DECISION TREE
# ------------------------------------------

plt.figure(figsize=(16, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)

plt.title("Decision Tree Classifier for Iris Dataset")
plt.show()