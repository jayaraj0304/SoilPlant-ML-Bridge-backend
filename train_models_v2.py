import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

# 1. Load Datasets
print("Loading datasets...")
df_agri = pd.read_csv('datasets/Agri_yield_prediction.csv')
df_health = pd.read_csv('datasets/plant_health_data.csv')
df_30cm = pd.read_excel('datasets/Sensor data for 30 cm.xlsx')
df_60cm = pd.read_excel('datasets/Sensor data for 60 cm.xlsx')

# 2. Data Preparation - Derived Targets
print("Preparing target variables...")

# Yield Loss target for Agri dataset
# Formula: (Max Yield for Crop - Current Yield) / Max Yield * 100
df_agri['Max_Yield'] = df_agri.groupby('Crop_Type')['Yield'].transform('max')
df_agri['Yield_Loss_Pct'] = ((df_agri['Max_Yield'] - df_agri['Yield']) / df_agri['Max_Yield']) * 100

# Mapping features to a common schema
# Schema: [temp, humidity, moisture, pH, chlorophyll, turbidity, N, P, K]

# Mapping Health dataset to schema
df_health_clean = pd.DataFrame({
    'temperature': df_health['Ambient_Temperature'],
    'humidity': df_health['Humidity'],
    'soilMoisture': df_health['Soil_Moisture'],
    'soilPH': df_health['Soil_pH'],
    'chlorophyll': df_health['Chlorophyll_Content'],
    'turbidity': np.random.uniform(0, 40, len(df_health)), # Simulating turbidity for this dataset
    'N': df_health['Nitrogen_Level'],
    'P': df_health['Phosphorus_Level'],
    'K': df_health['Potassium_Level'],
    'health_status': df_health['Plant_Health_Status'],
    'yield_loss': np.nan, # To be predicted later or inferred
    'yield_val': np.nan
})

# Mapping Agri dataset to schema (Mapping NDVI/Chlorophyll)
df_agri_clean = pd.DataFrame({
    'temperature': df_agri['Temperature'],
    'humidity': df_agri['Humidity'],
    'soilMoisture': np.random.uniform(30, 90, len(df_agri)), # Proxy for moisture
    'soilPH': df_agri['pH'],
    'chlorophyll': df_agri['Chlorophyll'],
    'turbidity': np.random.uniform(0, 50, len(df_agri)), # Proxy for microplastic
    'N': df_agri['N'],
    'P': df_agri['P'],
    'K': df_agri['K'],
    'health_status': np.nan,
    'yield_loss': df_agri['Yield_Loss_Pct'],
    'yield_val': df_agri['Yield']
})

# Combine for broader training context
df_combined = pd.concat([df_agri_clean, df_health_clean], ignore_index=True)

# Fill some NaNs with sensible defaults for training
df_combined['yield_loss'] = df_combined['yield_loss'].fillna(df_combined['yield_loss'].mean())
df_combined['yield_val'] = df_combined['yield_val'].fillna(df_combined['yield_val'].mean())

# Prepare features for all models
features = ['temperature', 'humidity', 'soilMoisture', 'soilPH', 'chlorophyll', 'turbidity', 'N', 'P', 'K']
X = df_combined[features].fillna(df_combined[features].mean())

# --- MODEL 1: YIELD LOSS ENSEMBLE (RF + GBR + MLP) ---
print("Training Model 1: Yield Loss Ensemble...")
y_loss = df_combined['yield_loss']
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X, y_loss, test_size=0.2, random_state=42)

estimators = [
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('gbr', GradientBoostingRegressor(random_state=42)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)) # Epochs!
]
loss_ensemble = StackingRegressor(estimators=estimators, final_estimator=RandomForestRegressor(n_estimators=10, random_state=42))
loss_ensemble.fit(X_train_l, y_train_l)
print(f"Yield Loss Model MAE: {mean_absolute_error(y_test_l, loss_ensemble.predict(X_test_l)):.2f}%")

# --- MODEL 2: PLANT HEALTH CLASSIFIER ---
print("Training Model 2: Plant Health Classifier...")
# Filter for rows that had health labels originally
df_h = df_combined.dropna(subset=['health_status'])
X_h = df_h[features]
y_h = df_h['health_status']

le = LabelEncoder()
y_h_encoded = le.fit_transform(y_h)

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h_encoded, test_size=0.2, random_state=42)
health_model = RandomForestClassifier(n_estimators=100, random_state=42)
health_model.fit(X_train_h, y_train_h)
print(f"Health Model Accuracy: {accuracy_score(y_test_h, health_model.predict(X_test_h)):.2f}")

# --- MODEL 3: HARVEST PREDICTOR ---
print("Training Model 3: Harvest Predictor...")
y_yield = df_combined['yield_val']
X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X, y_yield, test_size=0.2, random_state=42)
harvest_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
harvest_model.fit(X_train_y, y_train_y)
print(f"Harvest Model MAE: {mean_absolute_error(y_test_y, harvest_model.predict(X_test_y)):.2f}")

# 3. Save Models
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/yield_loss_model.pkl', 'wb') as f:
    pickle.dump(loss_ensemble, f)
with open('models/health_model.pkl', 'wb') as f:
    pickle.dump({'model': health_model, 'encoder': le}, f)
with open('models/harvest_model.pkl', 'wb') as f:
    pickle.dump(harvest_model, f)

print("\nDONE: All 3 models saved successfully in /models directory.")
