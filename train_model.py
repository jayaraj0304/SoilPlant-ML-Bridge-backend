import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import os

# 1. Generate Synthetic Agricultural Data
def generate_synthetic_data(samples=10000):
    np.random.seed(42)
    
    # Randomly generate sensor features
    temp = np.random.uniform(15, 45, samples)
    humidity = np.random.uniform(30, 95, samples)
    moisture = np.random.uniform(10, 90, samples)
    ph = np.random.uniform(4.0, 9.0, samples)
    chlorophyll = np.random.uniform(10, 80, samples)
    turbidity = np.random.uniform(0, 100, samples) # Proxy for microplastics
    
    # yield_loss formula
    yield_loss = (
        (abs(temp - 28) * 0.5) +          # Temp deviation from optimal 28C
        (abs(ph - 6.5) * 1.5) +           # pH deviation from optimal 6.5
        ((100 - moisture) * 0.2) +        # Low moisture adds to loss
        (turbidity * 0.3) +               # Microplastics (high turbidity) = direct loss
        ((80 - chlorophyll) * 0.4) +      # Low chlorophyll = yield reduction
        np.random.normal(0, 2, samples)   # Random noise
    )
    
    yield_loss = np.clip(yield_loss, 0, 100)
    
    df = pd.DataFrame({
        'temperature': temp,
        'humidity': humidity,
        'soilMoisture': moisture,
        'soilPH': ph,
        'chlorophyll': chlorophyll,
        'turbidity': turbidity,
        'yieldLoss': yield_loss
    })
    
    return df

print("Starting: Generating synthetic dataset...")
df = generate_synthetic_data()

X = df[['temperature', 'humidity', 'soilMoisture', 'soilPH', 'chlorophyll', 'turbidity']]
y = df['yieldLoss']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Starting: Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"REPORT: Model Trained! Mean Absolute Error: {mae:.2f}%")

if not os.path.exists('models'):
    os.makedirs('models')

with open('models/yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("DONE: Model saved to models/yield_model.pkl")

importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)
print("\nFeature Importance:")
print(importance)
