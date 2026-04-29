import joblib

model = joblib.load("price_model.pkl")

print("Model Type:", type(model))
print("Number of Trees:", model.n_estimators)
print("Features Used:", model.feature_names_in_)