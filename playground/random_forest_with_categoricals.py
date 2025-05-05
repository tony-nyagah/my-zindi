# Random Forest with Mixed Data Types (Numerical and Categorical)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Create sample data with both numerical and categorical features
np.random.seed(42)
n_samples = 1000

# Numerical features
size = np.random.randint(500, 5000, n_samples)
age = np.random.randint(0, 50, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
distance_to_city = np.random.uniform(0.5, 30, n_samples)

# Categorical features
neighborhood = np.random.choice(["Downtown", "Suburb", "Rural"], n_samples)
house_type = np.random.choice(["Apartment", "Townhouse", "Single Family"], n_samples)
renovation = np.random.choice(["Yes", "No"], n_samples)

# Price calculation with some randomness
# Adding effects from categorical variables
neighborhood_effect = {"Downtown": 50000, "Suburb": 20000, "Rural": -30000}
house_type_effect = {"Apartment": -20000, "Townhouse": 10000, "Single Family": 40000}
renovation_effect = {"Yes": 25000, "No": 0}

price = (
    size * 100
    + bedrooms * 20000
    - age * 3000
    - distance_to_city * 15000
    + np.array([neighborhood_effect[n] for n in neighborhood])
    + np.array([house_type_effect[h] for h in house_type])
    + np.array([renovation_effect[r] for r in renovation])
    + np.random.normal(0, 50000, n_samples)
)

# Create DataFrame
data = pd.DataFrame(
    {
        "size": size,
        "age": age,
        "bedrooms": bedrooms,
        "distance_to_city": distance_to_city,
        "neighborhood": neighborhood,
        "house_type": house_type,
        "renovation": renovation,
        "price": price,
    }
)

# Display sample of the data
print("Sample data:")
print(data.head())
print("\nData types:")
print(data.dtypes)

# Split data into features and target
X = data.drop("price", axis=1)
y = data["price"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Method 1: Using scikit-learn's Pipeline with ColumnTransformer
# Identify numerical and categorical columns
numerical_cols = ["size", "age", "bedrooms", "distance_to_city"]
categorical_cols = ["neighborhood", "house_type", "renovation"]

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Create and train the model with preprocessing pipeline
rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)

rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = rf_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Method 2: Manual transformation with pandas (alternative approach)
print("\nAlternative approach with pandas get_dummies:")

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

# Split again with encoded data
X_train_encoded, X_test_encoded, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Create and train the model directly on encoded data
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_encoded, y_train)

# Make predictions
y_pred_alt = rf_model.predict(X_test_encoded)

# Evaluate the alternative model
mse_alt = mean_squared_error(y_test, y_pred_alt)
r2_alt = r2_score(y_test, y_pred_alt)
print(f"Mean Squared Error: {mse_alt:.2f}")
print(f"R² Score: {r2_alt:.4f}")

# Predict for a new house with mixed data types
new_house = pd.DataFrame(
    {
        "size": [2500],
        "age": [5],
        "bedrooms": [3],
        "distance_to_city": [10],
        "neighborhood": ["Suburb"],
        "house_type": ["Single Family"],
        "renovation": ["Yes"],
    }
)

predicted_price = rf_pipeline.predict(new_house)[0]
print(f"\nPredicted price for new house: ${predicted_price:.2f}")

# Feature importance (note: only works with the second approach)
if hasattr(rf_model, "feature_importances_"):
    feature_importance = pd.DataFrame(
        {
            "Feature": X_train_encoded.columns,
            "Importance": rf_model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)

    print("\nTop 10 Feature Importance:")
    print(feature_importance.head(10))
