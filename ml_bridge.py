import firebase_admin
from firebase_admin import credentials, db
import pickle
import pandas as pd
import numpy as np
import os
import json
import time
import threading
from flask import Flask

# --- 1. SET UP MINI-SERVER FOR RENDER ---
app = Flask(__name__)

@app.route('/')
def health_check():
    return "ML Yield Prediction Bridge is Online!", 200

# --- 2. LOAD TRAINED ML MODEL ---
MODEL_PATH = 'models/yield_model.pkl'
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: ML Model not found at {MODEL_PATH}")
    exit()

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print("INFO: ML Model loaded successfully!")

# --- 3. INITIALIZE FIREBASE (Securely) ---
DATABASE_URL = "https://soilplant-fe521-default-rtdb.asia-southeast1.firebasedatabase.app/"

service_key_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT')

if service_key_json:
    print("INFO: Using Firebase credentials from Environment Variable.")
    cred_dict = json.loads(service_key_json)
    cred = credentials.Certificate(cred_dict)
else:
    cred_path = 'serviceAccountKey.json'
    if os.path.exists(cred_path):
        print(f"INFO: Using Firebase credentials from file: {cred_path}")
        cred = credentials.Certificate(cred_path)
    else:
        print("ERROR: No Firebase service account credentials found!")
        exit()

firebase_admin.initialize_app(cred, {
    'databaseURL': DATABASE_URL
})

# --- THRESHOLDS FOR ALERTS ---
THRESHOLDS = {
    'temperature': {'min': 20, 'max': 35, 'unit': '°C'},
    'humidity': {'min': 60, 'max': 90, 'unit': '%'},
    'soilMoisture': {'min': 50, 'max': 90, 'unit': '%'},
    'soilPH': {'min': 5.5, 'max': 7.0, 'unit': 'pH'},
    'chlorophyll': {'min': 30, 'max': 70, 'unit': 'SPAD'},
    'turbidity': {'min': 0, 'max': 30, 'unit': 'NTU'},
}

# --- 4. GENERATE ALERTS FROM SENSOR DATA ---
def generate_alerts(data):
    alerts = {}
    now = int(time.time() * 1000)
    alert_idx = 0

    checks = [
        ('temperature', 'Temperature', 'thermometer-outline'),
        ('humidity', 'Humidity', 'water-outline'),
        ('soilMoisture', 'Soil Moisture', 'leaf-outline'),
        ('soilPH', 'Soil pH', 'flask-outline'),
        ('chlorophyll', 'Chlorophyll', 'sunny-outline'),
        ('turbidity', 'Turbidity', 'analytics-outline'),
    ]

    for key, name, icon in checks:
        val = data.get(key)
        if val is None:
            continue
        th = THRESHOLDS[key]
        if val < th['min']:
            alert_idx += 1
            alerts[f'alert_{now}_{alert_idx}'] = {
                'title': f'Low {name} Detected',
                'message': f'{name} dropped to {val:.1f}{th["unit"]}. Recommended range: {th["min"]}-{th["max"]}{th["unit"]}.',
                'type': 'warning',
                'sensor': key,
                'value': val,
                'threshold': th['min'],
                'timestamp': now,
                'read': False,
            }
        elif val > th['max']:
            alert_idx += 1
            severity = 'danger' if val > th['max'] * 1.2 else 'warning'
            alerts[f'alert_{now}_{alert_idx}'] = {
                'title': f'High {name} Detected',
                'message': f'{name} rose to {val:.1f}{th["unit"]}. Safe range: {th["min"]}-{th["max"]}{th["unit"]}.',
                'type': severity,
                'sensor': key,
                'value': val,
                'threshold': th['max'],
                'timestamp': now,
                'read': False,
            }

    return alerts

# --- 5. GENERATE RECOMMENDATIONS ---
def generate_recommendations(data, yield_loss, risk_level):
    recs = {}
    idx = 0

    moisture = data.get('soilMoisture', 50)
    ph = data.get('soilPH', 6.5)
    turbidity = data.get('turbidity', 0)
    temp = data.get('temperature', 28)
    chlorophyll = data.get('chlorophyll', 42)

    if moisture < 50:
        idx += 1
        recs[f'rec{idx}'] = {
            'title': 'Increase Irrigation',
            'description': f'Soil moisture is at {moisture:.1f}%. Increase watering to reach 50-90% for optimal crop growth.',
            'priority': 'high',
            'icon': 'water-outline',
            'category': 'irrigation',
        }

    if ph < 5.5:
        idx += 1
        recs[f'rec{idx}'] = {
            'title': 'Adjust Soil pH (Too Acidic)',
            'description': f'Soil pH is {ph:.1f}. Apply lime to raise pH to 5.5-7.0 range.',
            'priority': 'medium',
            'icon': 'flask-outline',
            'category': 'soil',
        }
    elif ph > 7.0:
        idx += 1
        recs[f'rec{idx}'] = {
            'title': 'Adjust Soil pH (Too Alkaline)',
            'description': f'Soil pH is {ph:.1f}. Apply sulfur or compost to lower pH to 5.5-7.0.',
            'priority': 'medium',
            'icon': 'flask-outline',
            'category': 'soil',
        }

    if turbidity > 30:
        idx += 1
        recs[f'rec{idx}'] = {
            'title': 'Filter Irrigation Water',
            'description': f'Turbidity at {turbidity:.1f} NTU indicates possible microplastic contamination. Filter water before use.',
            'priority': 'high',
            'icon': 'alert-circle-outline',
            'category': 'water',
        }

    if chlorophyll < 30:
        idx += 1
        recs[f'rec{idx}'] = {
            'title': 'Apply Nitrogen Fertilizer',
            'description': f'Chlorophyll at {chlorophyll:.1f} SPAD is low. Apply nitrogen-rich fertilizer to boost crop health.',
            'priority': 'medium',
            'icon': 'leaf-outline',
            'category': 'crop',
        }

    if risk_level == 'High':
        idx += 1
        recs[f'rec{idx}'] = {
            'title': 'Urgent: High Yield Loss Risk',
            'description': f'Predicted yield loss is {yield_loss:.1f}%. Take immediate corrective action on flagged sensors.',
            'priority': 'high',
            'icon': 'warning-outline',
            'category': 'yield',
        }

    if idx == 0:
        recs['rec1'] = {
            'title': 'All Systems Normal',
            'description': 'All sensor readings are within optimal range. Continue current practices.',
            'priority': 'low',
            'icon': 'checkmark-circle-outline',
            'category': 'general',
        }

    return recs

# --- 6. COMPUTE FARM HEALTH SCORE ---
def compute_farm_health(data, yield_loss):
    scores = []
    for key, th in THRESHOLDS.items():
        val = data.get(key)
        if val is None:
            continue
        mid = (th['min'] + th['max']) / 2
        rng = (th['max'] - th['min']) / 2
        deviation = abs(val - mid) / rng if rng > 0 else 0
        score = max(0, 100 - (deviation * 50))
        scores.append(score)

    avg_sensor = sum(scores) / len(scores) if scores else 50
    yield_penalty = min(yield_loss * 1.5, 40)
    final_score = max(0, min(100, avg_sensor - yield_penalty))

    if final_score >= 80:
        label = 'Excellent'
    elif final_score >= 60:
        label = 'Good'
    elif final_score >= 40:
        label = 'Fair'
    elif final_score >= 20:
        label = 'Poor'
    else:
        label = 'Critical'

    return {
        'score': round(final_score, 1),
        'label': label,
        'timestamp': int(time.time() * 1000),
    }

# --- 7. ML PREDICTION + FULL SYNC LOGIC ---
def predict_and_sync(event):
    """
    Enhanced sync logic: Uses event.data directly for speed and cleans up old alerts.
    """
    # Use the event data if it is the full object, otherwise fetch it
    data = event.data
    if not data or not isinstance(data, dict) or 'temperature' not in data:
        data = db.reference('sensorData').get()
        
    if not data:
        return

    try:
        # 1. Extract and Log Features (Critical for debugging the 55.17% issue)
        current_features = {
            'temperature': float(data.get('temperature', 28.0)),
            'humidity': float(data.get('humidity', 60.0)),
            'soilMoisture': float(data.get('soilMoisture', 50.0)),
            'soilPH': float(data.get('soilPH', 6.5)),
            'chlorophyll': float(data.get('chlorophyll', 42.0)),
            'turbidity': float(data.get('turbidity', 0.0))
        }
        
        # Log the actual values being fed to the model
        print(f"\n--- INFERENCE STEP ---")
        print(f"Sensors: T:{current_features['temperature']} | H:{current_features['humidity']} | PH:{current_features['soilPH']} | SM:{current_features['soilMoisture']}")
        
        features_df = pd.DataFrame([current_features])

        # 2. Predict Yield Loss
        prediction = float(model.predict(features_df)[0])
        
        risk_level = "Low"
        if prediction > 30: risk_level = "High"
        elif prediction > 15: risk_level = "Medium"

        print(f"RESULT: Prediction = {prediction:.2f}% | Risk = {risk_level}")

        # 3. Sync Yield Prediction
        db.reference('yieldPrediction').set({
            'yieldLoss': prediction,
            'riskLevel': risk_level,
            'timestamp': int(time.time() * 1000)
        })

        # 4. Generate & Sync Alerts (Using .set() to avoid infinite growth)
        alerts = generate_alerts(data)
        if alerts:
            # We keep only the most recent alerts for the UI
            db.reference('alerts').set(alerts) 
            print(f"ALERTS: {len(alerts)} active alerts updated.")
        else:
            db.reference('alerts').set({}) # Clear alerts if all is safe

        # 5. Compute & Sync Farm Health
        health = compute_farm_health(data, prediction)
        db.reference('farmHealth').set(health)

        # 6. Store Sensor History (Limit to last 50 for performance)
        history_ref = db.reference('sensorHistory')
        history_entry = {**current_features, 'yieldLoss': prediction, 'timestamp': int(time.time() * 1000)}
        history_ref.push(history_entry)
        
        # Housekeeping: Prevent history node from growing too large
        # (Optional: implement a trimmer function if needed)

    except Exception as e:
        print(f"ERROR: Sync failed - {e}")

# Start the Firebase listener in the background
def start_firebase_listener():
    print("INFO: ML Bridge started listening for live sensor data...")
    db.reference('sensorData').listen(predict_and_sync)

# --- 8. MAIN EXECUTION ---
sync_thread = threading.Thread(target=start_firebase_listener, daemon=True)
sync_thread.start()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"INFO: Starting Health Check server on port {port}...")
    app.run(host='0.0.0.0', port=port)
