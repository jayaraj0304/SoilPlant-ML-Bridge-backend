# System Architecture: Soil-Plant Physiological Coupling

This diagram illustrates the flow of data from physical sensor simulation (ESP32) through our real-time ML inference bridge to the final analytical presentation layer.

```mermaid
graph TD
    subgraph IoT_Hardware_Simulation ["1. Sensor Layer (ESP32 Simulation)"]
        ESP32["ESP32 Microcontroller"]
        S1["DHT22 (Temp/Hum)"] --> ESP32
        S2["Soil Moisture (LDR Proxy)"] --> ESP32
        S3["Soil pH (Potentiometer)"] --> ESP32
        S4["Chlorophyll (Potentiometer)"] --> ESP32
        S5["Turbidity/Microplastics (Potentiometer)"] --> ESP32
    end

    subgraph Data_Orchestration ["2. Cloud & Data Layer"]
        Firebase["Firebase Realtime Database"]
        CSV["soil_plant_data.csv (Historical Data)"]
    end

    subgraph Intelligence_Layer ["3. ML Inference & Training"]
        ML_Bridge["Flask ML Bridge (ml_bridge.py)"]
        RF_Model["Trained RF/Ensemble Model"]
        Trainer["Training Script (train_model.py)"]
    end

    subgraph Presentation_Layer ["4. Analytical & Presentation Layer"]
        Notebook["soil_plant_ensemble_presentation.ipynb"]
        Visuals["Reliability & Stressor Plots"]
    end

    %% Data Flow
    ESP32 -- "HTTPS POST (JSON)" --> Firebase
    Firebase -- "Real-time Stream" --> ML_Bridge
    CSV -- "Offline Training" --> Trainer
    Trainer -- "Joblib Export" --> RF_Model
    RF_Model -- "Predictive Logic" --> ML_Bridge
    ML_Bridge -- "Predicted Yield Loss" --> Notebook
    Notebook -- "Ensemble Comparison" --> Visuals
```

### Component Overview:
- **Sensor Layer**: Simulates agricultural conditions using ESP32 and electronic components.
- **Cloud Layer**: Centralized data sync via Firebase ensures low-latency access for the ML bridge.
- **Intelligence Layer**: The heart of the system, transforming raw sensor values into actionable agricultural insights (Yield Loss %).
- **Presentation Layer**: A high-impact visualization suite for stakeholders to evaluate model reliability and sensor sensitivity.
