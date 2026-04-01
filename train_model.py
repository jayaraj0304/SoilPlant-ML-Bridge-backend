import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# 1. Generate Synthetic Agricultural Data
def generate_synthetic_data(samples=1000): # Reduced samples for faster local execution
    np.random.seed(42)
    
    # Randomly generate sensor features - EXPANDED RANGES
    temp = np.random.uniform(0, 50, samples)            # 0 to 50C
    humidity = np.random.uniform(0, 100, samples)       # 0 to 100%
    moisture = np.random.uniform(0, 100, samples)       # 0 to 100%
    ph = np.random.uniform(0, 14.0, samples)            # 0 to 14 pH
    chlorophyll = np.random.uniform(0, 100, samples)    # 0 to 100
    turbidity = np.random.uniform(0, 100, samples)      # 0 to 100
    
    # yield_loss formula (physiological coupling logic)
    yield_loss = (
        (abs(temp - 25) * 0.8) +          # Higher penalty for temp dev
        (abs(ph - 7.0) * 2.0) +           # Higher penalty for pH dev
        ((100 - moisture) * 0.3) +        # Moisture impact
        (turbidity * 0.4) +               # Microplastics impact
        ((100 - chlorophyll) * 0.5) +     # Chlorophyll impact
        np.random.normal(0, 1, samples)   # Lower noise for cleaner demo
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

# Save dataset to CSV as requested
csv_file = 'soil_plant_data.csv'
df.to_csv(csv_file, index=False)
print(f"DONE: Dataset saved to {csv_file}")

# 2. Prepare Data
X = df[['temperature', 'humidity', 'soilMoisture', 'soilPH', 'chlorophyll', 'turbidity']]
y = df['yieldLoss']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train Model
print("Starting: Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate Metrics
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- MODEL PERFORMANCE REPORT ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}%")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")
print("--------------------------------")

# 5. Save Model
if not os.path.exists('models'):
    os.makedirs('models')

with open('models/yield_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("DONE: Model saved to models/yield_model.pkl")

# 6. Visualizations
print("\nGenerating visualizations...")
sns.set_theme(style="whitegrid")

# Plot 1: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Yield Loss (%)')
plt.ylabel('Predicted Yield Loss (%)')
plt.title('Actual vs. Predicted Yield Loss')
plt.savefig('actual_vs_predicted.png')
plt.close()

# Plot 2: Feature Importance
importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
importance = importance.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=importance, palette='viridis')
plt.title('Feature Importance for Yield Loss Prediction')
plt.savefig('feature_importance.png')
plt.close()

# Plot 3: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Sensor-Yield Correlation Heatmap')
plt.savefig('correlation_heatmap.png')
plt.close()

print("DONE: Visualizations saved as PNG files (actual_vs_predicted.png, feature_importance.png, correlation_heatmap.png)")
