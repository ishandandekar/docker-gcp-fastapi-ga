import os

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create artifacts directory if it doesn't exist
if not os.path.exists("artifacts"):
    os.makedirs("artifacts")

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Calculate accuracy
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# Save the model
model_path = os.path.join("artifacts", "random_forest_model.joblib")
joblib.dump(rf_model, model_path)
print(f"Model saved to {model_path}")

# Save feature names for reference
feature_names = iris.feature_names
joblib.dump(feature_names, os.path.join("artifacts", "feature_names.joblib"))
