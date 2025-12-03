import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data.csv")

# Features and labels
X = data[["study_hours", "attendance", "previous_score"]]
y = data["final_grade"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction example
hours = float(input("Enter study hours: "))
att = float(input("Enter attendance (%): "))
prev = float(input("Enter previous score: "))

prediction = model.predict([[hours, att, prev]])
print(f"\nPredicted Final Grade: {prediction[0]:.2f}")

# Visualization
plt.scatter(data["study_hours"], data["final_grade"])
plt.xlabel("Study Hours")
plt.ylabel("Final Grade")
plt.title("Study Hours vs Final Grade")
plt.show()
