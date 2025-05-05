# Random Forest with Missing Values

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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
neighborhood = np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples)
house_type = np.random.choice(['Apartment', 'Townhouse', 'Single Family'], n_samples)
renovation = np.random.choice(['Yes', 'No'], n_samples)

# Create price as before - using a base price to avoid negative values
base_price = 300000  # Base price to ensure positive values
neighborhood_effect = {'Downtown': 50000, 'Suburb': 20000, 'Rural': 0}  # Rural no longer negative
house_type_effect = {'Apartment': 0, 'Townhouse': 30000, 'Single Family': 60000}  # All non-negative
renovation_effect = {'Yes': 25000, 'No': 0}

price = (
    base_price +
    size * 100 + 
    bedrooms * 20000 - 
    age * 2000 -  # Reduced age penalty
    distance_to_city * 5000 +  # Reduced distance penalty
    np.array([neighborhood_effect[n] for n in neighborhood]) +
    np.array([house_type_effect[h] for h in house_type]) +
    np.array([renovation_effect[r] for r in renovation]) +
    np.random.normal(0, 40000, n_samples)
)

# Ensure all prices are positive
price = np.maximum(price, 50000)  # Minimum house price of $50,000

# Create DataFrame
data = pd.DataFrame({
    'size': size,
    'age': age,
    'bedrooms': bedrooms,
    'distance_to_city': distance_to_city,
    'neighborhood': neighborhood,
    'house_type': house_type,
    'renovation': renovation,
    'price': price
})

# Introduce missing values (approximately 10% of data)
# Create a mask of random positions to replace with NaN
def introduce_missing(series, missing_rate=0.1):
    mask = np.random.random(len(series)) < missing_rate
    series_copy = series.copy()
    series_copy[mask] = np.nan
    return series_copy

# Add missing values to numerical columns
data['size'] = introduce_missing(data['size'])
data['age'] = introduce_missing(data['age'])
data['bedrooms'] = introduce_missing(data['bedrooms'])
data['distance_to_city'] = introduce_missing(data['distance_to_city'])

# Add missing values to categorical columns
for col in ['neighborhood', 'house_type', 'renovation']:
    mask = np.random.random(len(data)) < 0.1
    data.loc[mask, col] = None

# Display information about missing values
print("Missing values per column:")
print(data.isna().sum())
print(f"\nPercentage of rows with at least one missing value: {(data.isna().any(axis=1).mean() * 100):.2f}%")
print("\nSample rows with missing values:")
print(data[data.isna().any(axis=1)].head())

# Split data into features and target
X = data.drop('price', axis=1)
y = data['price']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Approach 1: Using scikit-learn's Pipeline with imputation
numerical_cols = ['size', 'age', 'bedrooms', 'distance_to_city']
categorical_cols = ['neighborhood', 'house_type', 'renovation']

# Create a preprocessing pipeline with imputation
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('passthrough', 'passthrough')
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# Create and train the model with preprocessing pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = rf_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance with Imputation:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# Approach 2: Using Random Forest's built-in missing value handling
# Note: RandomForest can handle missing values in two ways:
# 1. During training, it can split around NaN values (not available in scikit-learn)
# 2. We can set missing_values parameter in SimpleImputer for handling null values

print("\nApproach 2: Manual handling and using Random Forest directly")

# For numerical features: Using SimpleImputer
num_imputer = SimpleImputer(strategy='median')
X_train_num = pd.DataFrame(
    num_imputer.fit_transform(X_train[numerical_cols]),
    columns=numerical_cols,
    index=X_train.index
)
X_test_num = pd.DataFrame(
    num_imputer.transform(X_test[numerical_cols]),
    columns=numerical_cols,
    index=X_test.index
)

# For categorical features: First impute, then one-hot encode
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train_cat_imputed = pd.DataFrame(
    cat_imputer.fit_transform(X_train[categorical_cols]),
    columns=categorical_cols,
    index=X_train.index
)
X_test_cat_imputed = pd.DataFrame(
    cat_imputer.transform(X_test[categorical_cols]),
    columns=categorical_cols,
    index=X_test.index
)

# One-hot encode the imputed categorical data
X_train_cat = pd.get_dummies(X_train_cat_imputed, columns=categorical_cols)
X_test_cat = pd.get_dummies(X_test_cat_imputed, columns=categorical_cols)

# Make sure both train and test have the same columns after one-hot encoding
missing_cols = set(X_train_cat.columns) - set(X_test_cat.columns)
for col in missing_cols:
    X_test_cat[col] = 0
X_test_cat = X_test_cat[X_train_cat.columns]

# Combine numerical and categorical features
X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)

# Create and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train)

# Make predictions
y_pred_alt = rf_model.predict(X_test_processed)

# Evaluate the model
mse_alt = mean_squared_error(y_test, y_pred_alt)
r2_alt = r2_score(y_test, y_pred_alt)
print(f"Mean Squared Error: {mse_alt:.2f}")
print(f"R² Score: {r2_alt:.4f}")

# Approach 3: Exploring other imputation strategies

print("\nApproach 3: Using different imputation strategies")

# Different imputation strategies for numerical data
imputation_strategies = ['mean', 'median', 'most_frequent']

for strategy in imputation_strategies:
    # Create a preprocessing pipeline with specific imputation strategy
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=strategy)),
                ('passthrough', 'passthrough')
            ]), numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])
    
    # Create and train the model
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Numerical imputation strategy '{strategy}':")
    print(f"  R² Score: {r2:.4f}")

# Predict for a new house with some missing values
new_house = pd.DataFrame({
    'size': [2500],
    'age': [np.nan],  # Missing age
    'bedrooms': [3],
    'distance_to_city': [10],
    'neighborhood': ['Suburb'],
    'house_type': [None],  # Missing house type
    'renovation': ['Yes']
})

predicted_price = rf_pipeline.predict(new_house)[0]
print(f"\nPredicted price for new house with missing values: ${predicted_price:.2f}")

# Add visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Visualize prediction accuracy
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest: Actual vs Predicted Prices')
plt.tight_layout()
plt.show()

# 2. Visualize feature importance (from the manually processed model)
if hasattr(rf_model, 'feature_importances_'):
    # Get feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train_processed.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Plot top 10 features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance')
    plt.tight_layout()
    plt.show()

# 3. Visualize missing data patterns
plt.figure(figsize=(10, 6))
sns.heatmap(data.isna(), cbar=False, yticklabels=False, cmap='viridis')
plt.title('Missing Value Patterns')
plt.xlabel('Features')
plt.tight_layout()
plt.show()

# 4. Compare imputation strategies
strategies = ['mean', 'median', 'most_frequent']
r2_scores = []

for strategy in strategies:
    # Create a preprocessing pipeline with specific imputation strategy
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=strategy)),
                ('passthrough', 'passthrough')
            ]), numerical_cols),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])
    
    # Create and train the model
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    rf_pipeline.fit(X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# Plot comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=strategies, y=r2_scores)
plt.title('R² Score by Imputation Strategy')
plt.ylim(0.9, 1.0)  # Adjust as needed
plt.tight_layout()
plt.show()

# 5. Visualize error distribution
plt.figure(figsize=(10, 6))
errors = y_test - y_pred
sns.histplot(errors, kde=True)
plt.title('Error Distribution')
plt.xlabel('Prediction Error')
plt.axvline(x=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# 6. Visualize impact of missing values on prediction error
# Create a new column that counts missing values per row
X_test_missing = X_test.copy()
X_test_missing['missing_count'] = X_test.isna().sum(axis=1)
X_test_missing['abs_error'] = np.abs(y_test.values - y_pred)

# Plot error vs missing value count
plt.figure(figsize=(10, 6))
sns.boxplot(x='missing_count', y='abs_error', data=X_test_missing)
plt.title('Impact of Missing Values on Prediction Error')
plt.xlabel('Number of Missing Values in Row')
plt.ylabel('Absolute Error')
plt.tight_layout()
plt.show()
# %%
