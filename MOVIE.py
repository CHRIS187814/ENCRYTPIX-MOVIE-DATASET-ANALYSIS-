import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define paths
zip_path = "IMDb Movies India.csv.zip"  # ZIP file
extract_path = "movies_data"
csv_filename = "IMDb Movies India.csv"  # Correct CSV file name

# Ensure the ZIP file exists
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"Error: {zip_path} not found! Please check the file location.")

# Extract ZIP file if not already extracted
if not os.path.exists(extract_path):
    os.makedirs(extract_path, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        print("Extracted files:", zip_ref.namelist())  # Debugging

# Check if CSV file exists
csv_path = os.path.join(extract_path, csv_filename)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Error: {csv_filename} not found inside {extract_path}!")

print(f"Using dataset: {csv_filename}")  # Debugging

# Load Dataset
data = pd.read_csv(csv_path, low_memory=False, encoding="ISO-8859-1")
print(data.columns)

# Display first few rows
print(data.head())

# Selecting relevant features
features = ['Genre', 'Director', 'Duration', 'Year', 'Votes']
target = 'Rating'

# Ensure required columns exist before selection
required_columns = ['Genre', 'Director', 'Duration', 'Rating', 'Year', 'Votes']

data = data[required_columns].dropna()

# Convert 'Duration' to numeric (extract numbers only)
data['Duration'] = data['Duration'].str.extract('(\\d+)').astype(float)

# Convert 'Year' to numeric (extract numbers only)
data['Year'] = data['Year'].str.extract('(\\d+)').astype(float)

# Remove commas and convert 'Votes' to numeric
data['Votes'] = data['Votes'].str.replace(',', '', regex=True).astype(float)

# One-hot encode categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
categorical_features = ['Genre', 'Director']
encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
encoded_data.columns = encoder.get_feature_names_out(categorical_features)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = ['Duration', 'Year', 'Votes']
scaled_data = pd.DataFrame(scaler.fit_transform(data[numerical_features]), columns=numerical_features)

# Merge encoded and scaled data
X = pd.concat([encoded_data, scaled_data], axis=1)
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")

# Plot Actual vs Predicted Ratings
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()
