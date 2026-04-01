#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHTesp.h>
#include <HTTPClient.h>
#include <WiFi.h>

// ─── SETTINGS ────────────────────────────────────────────────────────────────
#define WIFI_SSID "Wokwi-GUEST"
#define WIFI_PASSWORD ""
#define DATABASE_URL                                                           \
  "https://soilplant-fe521-default-rtdb.asia-southeast1.firebasedatabase.app"

// ─── PINS ───────────────────────────────────────────────────────────────────
#define DHT_PIN 15         // DHT22 (Temp/Hum)
#define MOISTURE_PIN 32    // LDR (Moisture Proxy)
#define CHLOROPHYLL_PIN 33 // Pot 3 (Bio Mass/Chlorophyll)
#define TURBIDITY_PIN 34   // Pot 1 (Microplast Conc/Turbidity)
#define PH_PIN 35          // Pot 2 (Soil pH)

// ─── DISPLAY ────────────────────────────────────────────────────────────────
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

DHTesp dht;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\n--- ESP32 Soil-Plant Monitor Starting ---");

  // OLED Initialization
  Wire.begin(21, 22); // Explicitly start I2C on ESP32 default pins
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("OLED (0x3C) failed, trying 0x3D..."));
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3D)) {
      Serial.println(F("SSD1306 allocation failed. Check wiring!"));
      for(;;); // Don't proceed if display fails
    }
  }
  
  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Farming Monitor");
  display.println("Connecting WiFi...");
  display.display();

  dht.setup(DHT_PIN, DHTesp::DHT22);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
    display.print(".");
    display.display();
  }
  Serial.println("\nWiFi Connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());

  display.clearDisplay();
  display.setCursor(0, 0);
  display.println("Farming Monitor");
  display.println("WiFi: Connected!");
  display.print("IP: "); display.println(WiFi.localIP());
  display.display();
  delay(1000);
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

    // 5. Fetch Predicted Yield Loss from DB
    float yieldLossVal = -1.0;
    String urlYield = String(DATABASE_URL) + "/yieldPrediction/yieldLoss.json";
    
    http.begin(urlYield);
    // Important: Some ESP32 boards need this for HTTPS to work without a certificate
    // Adding this to ensure the GET request doesn't fail silently
    #ifdef ESP32
    // If using an older core or specific setup, this might be needed
    // http.setInsecure(); 
    #endif

    Serial.print("Fetching Yield Loss... ");
    int codeYield = http.GET();
    if (codeYield == 200) {
      String payload = http.getString();
      payload.trim(); // Remove any extra whitespace or newlines
      Serial.print("Payload: "); Serial.println(payload);
      
      if (payload != "null" && payload != "" && payload.length() > 0) {
        yieldLossVal = payload.toFloat();
      } else {
        Serial.println("Data not ready yet (null)");
      }
    } else {
      Serial.print("Failed! Code: ");
      Serial.println(codeYield);
    }
    http.end();

    // 6. Update OLED Display
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Farm Diagnostics:");
    
    display.print("PH: "); display.print(ph, 1);
    display.print("  SM: "); display.print(moisture, 0); display.println("%");
    
    display.print("Temp: "); display.print(temp, 1);
    display.print(" Hum: "); display.print(hum, 0); display.println("%");
    
    display.println("--------------------");
    display.setTextSize(2);
    display.print("Loss:");
    if (yieldLossVal >= 0) {
        display.print(yieldLossVal, 1);
        display.println("%");
    } else {
        display.setCursor(65, 45); // Adjust for alignment
        display.println("WAIT");
    }
    display.display();

    Serial.println("---- SENSOR DATA ----");
    Serial.print("Temp: "); Serial.print(temp, 1); Serial.print("C | Hum: "); Serial.print(hum, 1); Serial.println("%");
    Serial.print("Moisture: "); Serial.print(moisture, 1); Serial.print("% | pH: "); Serial.println(ph, 1);
    Serial.print("Chlorophyll: "); Serial.print(chlorophyll, 1); Serial.print(" | Turbidity: "); Serial.println(turbidity, 1);
    Serial.println("Yield Loss: " + (yieldLossVal >= 0 ? String(yieldLossVal, 2) + "%" : "Pending..."));
    Serial.println("---------------------");
  }

  // Rapid update for demo (5 seconds)
  delay(5000);
}