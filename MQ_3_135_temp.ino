#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <DHT.h>

// === OLED Setup ===
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET     -1
#define SCREEN_ADDRESS 0x3C
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// === DHT Sensor Setup ===
#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// === MQ Sensor Pins ===
#define MQ3_PIN 32       // Acetone - Diabetes
#define MQ135_PIN 39     // Ammonia - GI

// === Prediction ===
String prediction = "Waiting...";

void setup() {
  Serial.begin(9600);
  dht.begin();

  // Init OLED
  if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
    Serial.println(F("OLED not found"));
    while (true);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Initializing...");
  display.display();
  delay(2000);
}

void loop() {
  // === Read sensors ===
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();

  float adc_mq3 = analogRead(MQ3_PIN);        // Raw MQ3 reading
  float adc_mq135 = analogRead(MQ135_PIN);    // Raw MQ135 reading
  float mq3= (adc_mq3 * 5)/4095;
  float mq135= (adc_mq135*3.3)/4095;


  // === Check DHT ===
  if (isnan(temp) || isnan(hum)) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("DHT Error!");
    display.display();
    delay(2000);
    return;
  }

  // === Send to Python for prediction ===
  Serial.print(mq3,2);
  Serial.print(",");
  Serial.print(mq135,3);
  Serial.print(",");
  Serial.print(temp, 2);
  Serial.print(",");
  Serial.println(hum, 2);

  // === Read back prediction ===
  if (Serial.available()) {
    prediction = Serial.readStringUntil('\n');
    prediction.trim();
  }

  // === Display on OLED ===
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(1);
  display.println("Breath Analysis");

  display.setCursor(0, 12);
  display.print("Temp: ");
  display.print(temp, 1);
  display.print(" C");

  display.setCursor(0, 22);
  display.print("Humidity: ");
  display.print(hum, 1);
  display.print(" %");

  display.setCursor(0, 32);
  display.print("MQ3 Raw: ");
  display.print(mq3,2);

  display.setCursor(0, 42);
  display.print("MQ135 Raw: ");
  display.print(mq135,2);

  display.setCursor(0, 52);
  display.print("Prediction:");
  display.print(prediction);

  display.display();
  delay(2000);
}
