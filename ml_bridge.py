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
# Render's free tier requires a web server to stay alive.
app = Flask(__name__)

@app.route('/')
def health_check():
    return "ML Yield Prediction Bridge is Online!", 200

# --- 2. LOAD TRAINED ML MODEL ---
MODEL_PATH = 'models/yield_model.pkl'
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: ML Model not found at {MODEL_PATH}")
    # Don't exit here if we're in a cloud environment where it might be built later,
    # but for now, we need it.
    exit()

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print("INFO: ML Model loaded successfully!")

# --- 3. INITIALIZE FIREBASE (Securely) ---
DATABASE_URL = "https://soilplant-fe521-default-rtdb.asia-southeast1.firebasedatabase.app/"

# ☁️ CLOUD DEPLOYMENT TIP: 
# On Render, add an environment variable called 'FIREBASE_SERVICE_ACCOUNT' 
# and paste the entire content of your serviceAccountKey.json file into it.
service_key_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT')

if service_key_json:
    print("INFO: Using Firebase credentials from Environment Variable.")
    cred_dict = json.loads(service_key_json)
    cred = credentials.Certificate(cred_dict)
else:
    # Fallback to local file for testing
    cred_path = 'serviceAccountKey.json'
    if os.path.exists(cred_path):
        print(f"INFO: Using Firebase credentials from file: {cred_path}")
        cred = credentials.Certificate(cred_path)
    else:
        print("ERROR: No Firebase service account credentials found! (Check env var or file)")
        exit()

firebase_admin.initialize_app(cred, {
    'databaseURL': DATABASE_URL
})

# --- 4. ML PREDICTION LOGIC ---
def predict_and_sync(event):
    """Callback function triggered when sensor data changes in Firebase"""
    if event.data:
        data = event.data
        print(f"INPUT: New sensor data received: {data}")
        
        try:
            # Extract features (adjust defaults if necessary)
            features = pd.DataFrame([{
                'temperature': data.get('temperature', 28.0),
                'humidity': data.get('humidity', 60.0),
                'soilMoisture': data.get('soilMoisture', 50.0),
                'soilPH': data.get('soilPH', 6.5),
                'chlorophyll': data.get('chlorophyll', 42.0),
                'turbidity': data.get('turbidity', 0.0)
            }])
            
            # Predict Yield Loss
            prediction = model.predict(features)[0]
            
            # Determine Risk Level
            risk_level = "Low"
            if prediction > 30: risk_level = "High"
            elif prediction > 15: risk_level = "Medium"
            
            print(f"PREDICTION: {prediction:.2f}% Yield Loss ({risk_level})")
            
            # Sync back to Firebase dashboard
            db.reference('yieldPrediction').update({
                'yieldLoss': float(prediction),
                'riskLevel': risk_level,
                'timestamp': int(time.time() * 1000)
            })
            print("SYNC: Prediction synced to Firebase Realtime Database.")
            
        except Exception as e:
            print(f"ERROR during prediction: {e}")

# Start the Firebase listener in the background so it doesn't block Flask
def start_firebase_listener():
    print("INFO: ML Bridge started listening for live sensor data...")
    db.reference('sensorData').listen(predict_and_sync)

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    # Start the Firebase sync thread
    sync_thread = threading.Thread(target=start_firebase_listener, daemon=True)
    sync_thread.start()
    
    # Start the Flask Health Check server (Render will send requests here)
    port = int(os.environ.get("PORT", 5000))
    print(f"INFO: Starting Health Check server on port {port}...")
    app.run(host='0.0.0.0', port=port)

