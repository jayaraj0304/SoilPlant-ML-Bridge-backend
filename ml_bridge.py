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
# --- 7. ML PREDICTION + FULL SYNC LOGIC ---
def predict_and_sync(event):
    """
    Triggered by Firebase updates. Processes data and syncs results.
    """
    try:
        # 1. Fetch Latest Data
        data = db.reference('sensorData').get()
        if not data or 'temperature' not in data:
            return

        # 2. Strict Feature Ordering (Matches train_model.py exactly)
        feature_order = ['temperature', 'humidity', 'soilMoisture', 'soilPH', 'chlorophyll', 'turbidity']
        
        input_data = {}
        for feat in feature_order:
            val = data.get(feat)
            input_data[feat] = float(val) if val is not None else 0.0

        # Create DataFrame and enforce column order
        features_df = pd.DataFrame([input_data])[feature_order]

        # 3. Inference
        prediction = float(model.predict(features_df)[0])
        
        risk_level = "Low"
        if prediction > 30: risk_level = "High"
        elif prediction > 15: risk_level = "Medium"

        # 4. Sync Everything
        timestamp = int(time.time() * 1000)
        
        # Yield Prediction
        db.reference('yieldPrediction').set({
            'yieldLoss': prediction,
            'riskLevel': risk_level,
            'timestamp': timestamp
        })

        # Farm Health
        health = compute_farm_health(input_data, prediction)
        db.reference('farmHealth').set(health)

        # Alerts
        alerts = generate_alerts(input_data)
        db.reference('alerts').set(alerts if alerts else {})

        # Recommendations
        recs = generate_recommendations(input_data, prediction, risk_level)
        db.reference('recommendations').set(recs)

        # History Management (Limit to latest 50 logs)
        history_ref = db.reference('sensorHistory')
        history_ref.push({**input_data, 'yieldLoss': prediction, 'timestamp': timestamp})
        
        # Periodic cleanup of history (every 10 updates)
        if timestamp % 10 == 0:
            all_history = history_ref.get()
            if all_history and len(all_history) > 50:
                # Keep only most recent
                sorted_keys = sorted(all_history.keys(), key=lambda x: all_history[x].get('timestamp', 0))
                for old_key in sorted_keys[:-50]:
                    history_ref.child(old_key).delete()

        print(f"[{time.strftime('%H:%M:%S')}] SYNCED: Loss {prediction:.2f}% | Health {health['score']}")

    except Exception as e:
        print(f"ERROR in predict_and_sync: {e}")

# --- 8. MAIN EXECUTION ---
def run_bridge():
    print("INFO: ML Bridge is active. Listening for changes...")
    db.reference('sensorData').listen(predict_and_sync)
    
    # Render keep-alive / backup sync (checks every 30s in case stream stalls)
    while True:
        time.sleep(30)
        # Heartbeat check
        predict_and_sync(None)

if __name__ == "__main__":
    # Start the bridge in a dedicated thread
    bridge_thread = threading.Thread(target=run_bridge, daemon=True)
    bridge_thread.start()

    # Flask server for Render port binding
    port = int(os.environ.get("PORT", 5000))
    print(f"INFO: Starting Health Check server on port {port}...")
    app.run(host='0.0.0.0', port=port)
