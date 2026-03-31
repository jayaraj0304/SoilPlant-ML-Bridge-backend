#include <WiFi.h>
#include <Firebase_ESP_Client.h>
#include <addons/RTDBHelper.h>

// ─── CREDENTIALS (FROM YOUR PROJECT) ──────────────────────────────────────────
#define WIFI_SSID "Wokwi-GUEST"
#define WIFI_PASSWORD ""
#define API_KEY "AIzaSyB203V6rxUVAD_NH8KsN_eMUdp2JrcEcj8"
#define DATABASE_URL "https://soilplant-fe521-default-rtdb.asia-southeast1.firebasedatabase.app/"

// ─── DEVICE CONFIG ────────────────────────────────────────────────────────────
FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

void setup() {
  Serial.begin(115200);

  // 1. WiFi Connection
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\n✅ WiFi Connected!");

  // 2. Firebase Configuration
  config.api_key = API_KEY;
  config.database_url = DATABASE_URL;

  // Sign up as anonymous user (Simplest for simulation)
  if (Firebase.signUp(&config, &auth, "", "")) {
    Serial.println("✅ Firebase Authenticated!");
  } else {
    Serial.printf("❌ Auth Error: %s\n", config.signer.signupError.message.c_str());
  }

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);
}

void loop() {
  if (Firebase.ready()) {
    // 3. Simulating Sensor Readings from Analog Pins
    float moisture = (analogRead(34) / 4095.0) * 100.0;
    float ph = (analogRead(35) / 4095.0) * 14.0;
    float chlorophyll = (analogRead(33) / 4095.0) * 100.0;
    float turbidity = (analogRead(32) / 4095.0) * 100.0;
    
    // Simulating Temp & Humidity (since they aren't on analog pins)
    float temp = 25.0 + (random(0, 100) / 10.0);
    float hum = 60.0 + (random(0, 100) / 10.0);

    // 4. Update the "sensorData" path in Firebase
    FirebaseJson data;
    data.set("temperature", temp);
    data.set("humidity", hum);
    data.set("soilMoisture", moisture);
    data.set("soilPH", ph);
    data.set("chlorophyll", chlorophyll);
    data.set("turbidity", turbidity);
    data.set("timestamp", millis()); // Or server timestamp

    Serial.println("\n📡 Sending data to Firebase...");
    if (Firebase.RTDB.setJSON(&fbdo, "/sensorData", &data)) {
      Serial.println("✅ Data sent successfully!");
    } else {
      Serial.printf("❌ Failed to send: %s\n", fbdo.errorReason().c_str());
    }

    Serial.println("---- SENSOR DATA ----");
    Serial.printf("Temp: %.1fC | Hum: %.1f%%\n", temp, hum);
    Serial.printf("Moisture: %.1f%% | pH: %.1f\n", moisture, ph);
    Serial.printf("Chlorophyll: %.1f | Turbidity: %.1f\n", chlorophyll, turbidity);
    Serial.println("---------------------");
  }

  delay(40000); // Send data every 40 seconds
}