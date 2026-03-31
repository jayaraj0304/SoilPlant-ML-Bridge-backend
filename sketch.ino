#include <WiFi.h>
#include <HTTPClient.h>
#include <DHTesp.h>

// ─── SETTINGS ────────────────────────────────────────────────────────────────
#define WIFI_SSID "Wokwi-GUEST"
#define WIFI_PASSWORD ""
#define DATABASE_URL "https://soilplant-fe521-default-rtdb.asia-southeast1.firebasedatabase.app"

// ─── PINS ───────────────────────────────────────────────────────────────────
#define DHT_PIN         15   // DHT22 (Temp/Hum)
#define MOISTURE_PIN    32   // LDR (Moisture Proxy)
#define CHLOROPHYLL_PIN 33   // Pot 3 (Bio Mass/Chlorophyll)
#define TURBIDITY_PIN   34   // Pot 1 (Microplast Conc/Turbidity)
#define PH_PIN          35   // Pot 2 (Soil pH)

DHTesp dht;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n--- ESP32 Soil-Plant Monitor Starting ---");

  dht.setup(DHT_PIN, DHTesp::DHT22);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // 1. Read real values from dedicated pins
    float moisture = (analogRead(MOISTURE_PIN) / 4095.0) * 100.0;
    float ph = (analogRead(PH_PIN) / 4095.0) * 14.0;
    float chlorophyll = (analogRead(CHLOROPHYLL_PIN) / 4095.0) * 100.0;
    float turbidity = (analogRead(TURBIDITY_PIN) / 4095.0) * 100.0;
    
    // 2. Read Temp/Hum from DHT22
    TempAndHumidity tah = dht.getTempAndHumidity();
    float temp = !isnan(tah.temperature) ? tah.temperature : 28.5;
    float hum = !isnan(tah.humidity) ? tah.humidity : 65.0;

    // 3. Build JSON string manually
    String json = "{";
    json += "\"temperature\":" + String(temp, 1) + ",";
    json += "\"humidity\":" + String(hum, 1) + ",";
    json += "\"soilMoisture\":" + String(moisture, 1) + ",";
    json += "\"soilPH\":" + String(ph, 1) + ",";
    json += "\"chlorophyll\":" + String(chlorophyll, 1) + ",";
    json += "\"turbidity\":" + String(turbidity, 1) + ",";
    json += "\"timestamp\":" + String(millis());
    json += "}";

    // 4. Send to Firebase via REST API (PUT = overwrite sensorData)
    HTTPClient http;
    String url = String(DATABASE_URL) + "/sensorData.json";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");

    Serial.println("\nSending data to Firebase...");
    int httpCode = http.PUT(json);

    if (httpCode == 200) {
      Serial.println("Data sent successfully!");
    } else {
      Serial.print("Failed! HTTP Code: ");
      Serial.println(httpCode);
    }
    http.end();

    Serial.println("---- SENSOR DATA ----");
    Serial.print("Temp: "); Serial.print(temp, 1); Serial.print("C | Hum: "); Serial.print(hum, 1); Serial.println("%");
    Serial.print("Moisture: "); Serial.print(moisture, 1); Serial.print("% | pH: "); Serial.println(ph, 1);
    Serial.print("Chlorophyll: "); Serial.print(chlorophyll, 1); Serial.print(" | Turbidity: "); Serial.println(turbidity, 1);
    Serial.println("---------------------");
  }

  // Rapid update for demo (5 seconds)
  delay(5000);
}