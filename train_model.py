import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# STEP 1: Load dataset
df = pd.read_excel("placement_data.xlsx")

print("Data Loaded Successfully\n")
print(df.head())

# STEP 2: Preprocessing
if "Student_ID" in df.columns:
    df = df.drop("Student_ID", axis=1)

df["Placed"] = df["Placed"].astype(int)

# STEP 3: Check Data
print("\nMissing Values:\n", df.isnull().sum())

# STEP 4: Visualization
sns.countplot(x="Placed_Label", data=df)
plt.title("Placement Distribution")
plt.show()

# STEP 5: Features & Target
if "Placed_Label" in df.columns:
    X = df.drop(["Placed", "Placed_Label"], axis=1)
else:
    X = df.drop(["Placed"], axis=1)

y = df["Placed"]

# STEP 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 7: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# STEP 8: Random Forest (Better Model)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# STEP 9: Save Best Model
pickle.dump(rf, open("model.pkl", "wb"))

print("\nModel saved as model.pkl ✅")