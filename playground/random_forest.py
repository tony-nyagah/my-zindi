# Random Forest Prediction Example

# %%
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create sample data (house prices)
np.random.seed(42)
n_samples = 1000

# Features
size = np.random.randint(500, 5000, n_samples)  # Square feet
age = np.random.randint(0, 50, n_samples)  # House age in years
bedrooms = np.random.randint(1, 6, n_samples)  # Number of bedrooms
distance_to_city = np.random.uniform(0.5, 30, n_samples)  # Miles from city center

# Price calculation with some randomness
price = (
    size * 100
    + bedrooms * 20000
    - age * 3000
    - distance_to_city * 15000
    + np.random.normal(0, 50000, n_samples)
)

# Create DataFrame
data = pd.DataFrame(
    {
        "size": size,
        "age": age,
        "bedrooms": bedrooms,
        "distance_to_city": distance_to_city,
        "price": price,
    }
)

# Split data into features (X) and target (y)
X = data.drop("price", axis=1)
y = data["price"]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf_model.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Predict for a new house
new_house = pd.DataFrame(
    {"size": [2500], "age": [5], "bedrooms": [3], "distance_to_city": [10]}
)

predicted_price = rf_model.predict(new_house)[0]
print(f"\nPredicted price for new house: ${predicted_price:.2f}")

# Plot predicted vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Random Forest: Actual vs Predicted Prices")
plt.tight_layout()
plt.show()

# %%
