import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Customer_Data.csv")

# 1. Drop ID-like columns automatically
for col in df.columns:
    if df[col].dtype == "object" and df[col].nunique() == len(df):
        print(f"Dropping ID column: {col}")
        df = df.drop(columns=[col])

# 2. Create target variable (1 = Churned, 0 = Active)
y = (df["Customer_Status"] == "Churned").astype(int)

# 3. Drop columns not useful for training
X = df.drop(columns=["Customer_Status", "Churn_Category", "Churn_Reason"], errors="ignore")

# 4. Convert categorical variables to numeric
X = pd.get_dummies(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train RandomForest model
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# 7. Save trained model and feature columns
joblib.dump(rf, "churn_model.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("✅ Model training completed successfully!")
print(f"Saved model with {len(X.columns)} features.")
