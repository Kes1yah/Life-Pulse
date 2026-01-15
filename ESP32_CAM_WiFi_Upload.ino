#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// ========== NETWORK CONFIG ==========
const char* ssid = "Redmi Note 13 Pro 5G";
const char* password = "12345678";
const char* serverUrl = "http://192.168.202.227:5000/api/image-upload";  // Backend IP

// ========== AI-THINKER PINOUT ==========
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void setup() {
  Serial.begin(115200);
  Serial.println("\n\n=== ESP32-CAM WITH WiFi IMAGE UPLOAD ===");
  delay(2000);

  // ===== CAMERA INIT =====
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[ERROR] Camera init failed: 0x%x\n", err);
    return;
  }
  Serial.println("[OK] Camera initialized!");

  // ===== WiFi INIT =====
  Serial.printf("[WiFi] Connecting to %s...\n", ssid);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[WiFi] ✅ Connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[WiFi] ❌ Failed to connect!");
  }

  Serial.println("\n=== READY ===");
  Serial.println("Press 'C' to capture and upload image");
  Serial.println("Press 'T' to toggle between capture modes");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    
    if (c == 'C' || c == 'c') {
      captureAndUpload();
    }
    else if (c == 'T' || c == 't') {
      toggleCameraSettings();
    }
  }
  delay(100);
}

void captureAndUpload() {
  Serial.println("\n[CAM] Capturing image...");
  
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[ERROR] Capture failed!");
    return;
  }

  Serial.printf("[CAM] ✅ Captured! Size: %zu bytes\n", fb->len);
  Serial.println("[WiFi] Uploading to backend...");

  // ===== SEND TO BACKEND =====
  uploadImageToBackend(fb->buf, fb->len);
  
  esp_camera_fb_return(fb);
}

void uploadImageToBackend(uint8_t* jpg_data, size_t jpg_len) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[ERROR] WiFi not connected!");
    return;
  }

  HTTPClient http;
  
  // Prepare multipart form data
  String boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW";
  
  // Build the request
  String head = "--" + boundary + "\r\n"
                "Content-Disposition: form-data; name=\"image\"; filename=\"capture.jpg\"\r\n"
                "Content-Type: image/jpeg\r\n\r\n";
  
  String tail = "\r\n--" + boundary + "--\r\n";
  
  uint32_t contentLength = head.length() + jpg_len + tail.length();

  // Start HTTP connection
  http.begin(serverUrl);
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);
  http.addHeader("Content-Length", String(contentLength));

  // Create buffer with complete body
  uint8_t* body = (uint8_t*)malloc(contentLength);
  uint32_t pos = 0;

  // Copy head
  memcpy(body + pos, head.c_str(), head.length());
  pos += head.length();

  // Copy image data
  memcpy(body + pos, jpg_data, jpg_len);
  pos += jpg_len;

  // Copy tail
  memcpy(body + pos, tail.c_str(), tail.length());
  pos += tail.length();

  // Send
  int httpResponseCode = http.sendRequest("POST", body, contentLength);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("[HTTP] Response Code: %d\n", httpResponseCode);
    
    // Parse JSON response
    if (response.indexOf("\"human_presence\":true") > 0) {
      Serial.println("[RESULT] ✅ HUMAN DETECTED!");
      
      if (response.indexOf("\"matched_person\":null") > 0) {
        Serial.println("[RESULT] ⚠️  Not in database");
      } else {
        // Extract person name (simple parsing)
        int nameStart = response.indexOf("\"matched_person\":\"") + 18;
        int nameEnd = response.indexOf("\"", nameStart);
        String personName = response.substring(nameStart, nameEnd);
        Serial.printf("[RESULT] ✅ MATCH: %s\n", personName.c_str());
      }
    } else if (response.indexOf("\"human_presence\":false") > 0) {
      Serial.println("[RESULT] ❌ No human detected");
    }
    
    Serial.println("[Response Body:]");
    Serial.println(response);
  } else {
    Serial.printf("[ERROR] HTTP Error: %d\n", httpResponseCode);
  }

  free(body);
  http.end();
}

void toggleCameraSettings() {
  sensor_t * s = esp_camera_sensor_get();
  if (!s) return;

  // Toggle between different frame sizes
  static int frameSize = 0;
  int sizes[] = {FRAMESIZE_QVGA, FRAMESIZE_VGA, FRAMESIZE_SVGA};
  
  frameSize = (frameSize + 1) % 3;
  s->set_framesize(s, sizes[frameSize]);
  
  const char* sizeNames[] = {"QVGA (320x240)", "VGA (640x480)", "SVGA (800x600)"};
  Serial.printf("[CAM] Frame size changed to: %s\n", sizeNames[frameSize]);
}
