import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# --- 1. SETUP OUTPUT DIRECTORY ---
OUTPUT_DIR = 'presentation_output'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. LOAD DATASET ---
print("\n[STEP 1] LOADING DATASET FOR PANEL PRESENTATION...")
try:
    df = pd.read_csv('soil_plant_data.csv')
    print(f"Dataset Shape: {df.shape[0]} samples, {df.shape[1]} features")
except FileNotFoundError:
    print("ERROR: 'soil_plant_data.csv' not found. Please run 'train_model.py' first.")
    exit()

# --- 3. EXPLORATORY DATA ANALYSIS (EDA) ---
print("\n[STEP 2] PERFORMING DATA PREPROCESSING & EDA...")
print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Checking for Null Values ---")
print(df.isnull().sum())

# Plotting Feature Distributions
plt.figure(figsize=(15, 10))
df.hist(bins=20, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions (Sensor Data Coverage)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{OUTPUT_DIR}/feature_distributions.png')
plt.close()

# --- 4. PREPROCESSING (Feature-Target Split & Scaling) ---
X = df[['temperature', 'humidity', 'soilMoisture', 'soilPH', 'chlorophyll', 'turbidity']]
y = df['yieldLoss']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 5. MODEL TRAINING ---
print("\n[STEP 3] TRAINING RANDOM FOREST REGRESSOR...")
model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# --- 6. MODEL METRICS & EVALUATION ---
predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n--- PERFORMANCE METRICS FOR PANEL ---")
print(f"✅ Mean Absolute Error (MAE): {mae:.4f}")
print(f"✅ Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"✅ R-squared Score (Accuracy): {r2*100:.2f}%")
print("--------------------------------------")

# --- 7. FINAL VISUALIZATIONS ---
print(f"\n[STEP 4] GENERATING PRESENTATION VISUALIZATIONS IN '{OUTPUT_DIR}'...")

# 7.1 Correlation Heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
sns.heatmap(df.corr(), mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature-Yield Correlation Matrix', fontsize=14)
plt.savefig(f'{OUTPUT_DIR}/correlation_matrix.png')
plt.close()

# 7.2 Feature Importance (Requested Style)
plt.figure(figsize=(10, 6))
importances_series = pd.Series(model.feature_importances_, index=X.columns).sort_values()
importances_series.plot(kind='barh', color='darkcyan')
plt.title('Sensor Importance in Yield Prediction', fontsize=14)
plt.xlabel('Importance Weight')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ranked_feature_importance.png')
plt.close()

# 7.3 Reliability Plot (Jointplot as requested)
plt.figure(figsize=(10, 10))
g = sns.jointplot(x=y_test, y=predictions, kind='reg', color='indigo', height=8)
g.fig.suptitle('Prediction Reliability: Actual vs. Predicted', y=1.02, fontsize=14)
plt.xlabel('Actual Yield Loss (%)')
plt.ylabel('Predicted Yield Loss (%)')
plt.savefig(f'{OUTPUT_DIR}/model_reliability.png')
plt.close()

print(f"\nSUCCESS: All files and plots are ready in the '{OUTPUT_DIR}' folder.")
