# milk_quality.py
# -----------------------
# Milk Quality Prediction Project
# -----------------------

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =============================
# 1. Load Dataset
# =============================
DATA_PATH = os.path.join("data", "milknew.csv")
RESULTS_PATH = os.path.join("results", "figures")

os.makedirs(RESULTS_PATH, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("✅ Dataset loaded successfully!")
print(df.head())

# =============================
# 2. Data Cleaning & Exploration
# =============================
if 'Fat ' in df.columns:  # Fix column name if needed
    df.rename(columns={'Fat ': 'Fat'}, inplace=True)

# Encode Grade column
df['Grade'] = df['Grade'].map({'high': 2, 'medium': 1, 'low': 0})

print("\n📊 Dataset Info:")
print(df.info())
print("\nUnique Grades:", df['Grade'].unique())
print("\nMissing values:\n", df.isnull().sum())

# =============================
# 3. Data Visualization
# =============================
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(RESULTS_PATH, "correlation_heatmap.png"))
plt.close()

df.hist(bins=10, figsize=(20, 20), color='blue')
plt.suptitle("Feature Distributions")
plt.savefig(os.path.join(RESULTS_PATH, "histograms.png"))
plt.close()

sns.countplot(data=df, x='Grade')
plt.title("Distribution of Milk Quality Grades")
plt.savefig(os.path.join(RESULTS_PATH, "grade_distribution.png"))
plt.close()

sns.regplot(data=df, x="Temprature", y="pH")
plt.title("Temperature vs pH")
plt.savefig(os.path.join(RESULTS_PATH, "temp_vs_ph.png"))
plt.close()

sns.scatterplot(x="Temprature", y="Colour", data=df)
plt.title("Temperature vs Colour")
plt.savefig(os.path.join(RESULTS_PATH, "temp_vs_colour.png"))
plt.close()

print("📊 Plots saved in 'results/figures/'")

# =============================
# 4. Dataset Preparation
# =============================
X = df.drop(columns=['Grade'])
y = df['Grade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# 5. Logistic Regression Model
# =============================
acc_vec = []
c_vec = np.arange(0.1, 10, 0.5)

for c in c_vec:
    model = LogisticRegression(C=c, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_vec.append(accuracy_score(y_test, y_pred))

best_log_acc = max(acc_vec)
print(f"📈 Logistic Regression Accuracy: {best_log_acc:.2f}")

plt.plot(c_vec, acc_vec)
plt.xlabel("C (Regularization parameter)")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Accuracy vs C")
plt.savefig(os.path.join(RESULTS_PATH, "logistic_regression_accuracy.png"))
plt.close()

# =============================
# 6. Random Forest Model
# =============================
acc_vec_RF = []
depth_vec = np.arange(1, 20, 1)

for d in depth_vec:
    clf = RandomForestClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_RF = clf.predict(X_test)
    acc_vec_RF.append(accuracy_score(y_test, y_pred_RF))

best_rf_acc = max(acc_vec_RF)
print(f"🌲 Random Forest Accuracy: {best_rf_acc:.2f}")

plt.plot(depth_vec, acc_vec_RF)
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Random Forest Accuracy vs Tree Depth")
plt.savefig(os.path.join(RESULTS_PATH, "random_forest_accuracy.png"))
plt.close()

# =============================
# 7. Final Results
# =============================
print("\n✅ Training complete!")
print(f"Best Logistic Regression Accuracy: {best_log_acc:.2f}")
print(f"Best Random Forest Accuracy: {best_rf_acc:.2f}")
print("All results saved in 'results/figures/' 🎉")

