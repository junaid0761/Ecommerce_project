import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import joblib

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("amazon.csv")

# -----------------------------
# CLEAN CATEGORY
# -----------------------------
df["category"] = df["category"].astype(str).str.split("|").str[0]
df["category"] = df["category"].str.replace("&", " & ").str.strip()

# -----------------------------
# CLEAN NUMERIC COLUMNS
# -----------------------------
df["actual_price"] = (
    df["actual_price"]
    .astype(str)
    .str.replace("₹", "", regex=False)
    .str.replace(",", "", regex=False)
)

df["discount_percentage"] = (
    df["discount_percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["rating_count"] = (
    df["rating_count"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["actual_price"] = pd.to_numeric(df["actual_price"], errors="coerce")
df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["rating_count"] = pd.to_numeric(df["rating_count"], errors="coerce")

# -----------------------------
# DROP INVALID ROWS
# -----------------------------
df = df.dropna(subset=[
    "category",
    "rating",
    "rating_count",
    "discount_percentage",
    "actual_price"
])

# -----------------------------
# ENCODE CATEGORY
# -----------------------------
le = LabelEncoder()
df["category_encoded"] = le.fit_transform(df["category"])

# -----------------------------
# SELECT FEATURES
# -----------------------------
X = df[["category_encoded", "rating", "rating_count", "discount_percentage"]]

# 🔥 LOG TRANSFORMATION (BALANCE PRICE)
y = np.log1p(df["actual_price"])

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔥 BALANCED RANDOM FOREST
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,          # prevent extreme overfitting
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred_log = model.predict(X_test)

# Convert back to real price
y_pred = np.expm1(y_pred_log)
y_test_real = np.expm1(y_test)

mae = mean_absolute_error(y_test_real, y_pred)
rmse = np.sqrt(((y_test_real - y_pred) ** 2).mean())
r2 = r2_score(y_test_real, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

joblib.dump({"mae": mae, "rmse": rmse, "r2": r2}, "model_metrics.pkl")
print("✅ Model metrics saved.")

# -----------------------------
# SAVE MODEL + ENCODER
# -----------------------------
joblib.dump(model, "price_model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Balanced Model trained and saved successfully.")