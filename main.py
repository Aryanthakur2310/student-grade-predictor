import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ---- Load Dataset ----
data = pd.read_csv("data.csv")

# Input features and labels
X = data[["study_hours", "attendance", "previous_score"]]
y = data["final_grade"]

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Train Model ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Predictions ----
predictions = model.predict(X_test)

print("Model training complete!")
print("\nSample Predictions:\n")

for i in range(len(predictions)):
    print(f"Actual: {y_test.iloc[i]} | Predicted: {round(predictions[i], 2)}")

