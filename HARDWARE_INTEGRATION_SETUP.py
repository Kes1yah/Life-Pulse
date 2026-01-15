#!/usr/bin/env python3
"""
HARDWARE INTEGRATION SETUP - QUICK START
=========================================

This guide helps you integrate the ESP32-CAM hardware with Life-Pulse backend.
"""

print("""
╔══════════════════════════════════════════════════════════════════════╗
║         LIFE-PULSE v2.0 - HARDWARE INTEGRATION SETUP                ║
╚══════════════════════════════════════════════════════════════════════╝

┌─ WHAT'S NEW ──────────────────────────────────────────────────────┐
│                                                                      │
│  1. backend_server.py - NEW                                        │
│     - Flask server to receive images from ESP32-CAM               │
│     - Performs face recognition                                   │
│     - Returns: human_presence, matched_person, confidence         │
│                                                                      │
│  2. life_pulse_v2.py - MODIFIED                                    │
│     - Sends captured images to backend via WiFi                   │
│     - Graph shows NOISE or HEART RATE based on backend response   │
│     - Added SENTRY mode stability status                          │
│     - Hardware images displayed in center panel                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ REQUIRED DEPENDENCIES ───────────────────────────────────────────┐
│                                                                      │
│  pip install flask                                                │
│  pip install requests                                             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ STEP 1: START BACKEND SERVER ────────────────────────────────────┐
│                                                                      │
│  Terminal 1 (On your PC/Server):                                 │
│  $ python backend_server.py                                       │
│                                                                      │
│  Output will show:                                               │
│  ✅ Local IP: 192.168.x.x                                         │
│  ✅ Port: 5000                                                     │
│  ✅ Endpoints:                                                     │
│     - http://192.168.x.x:5000/api/image-upload (from ESP32-CAM)  │
│     - http://192.168.x.x:5000/api/alert (from Master Device)    │
│                                                                      │
│  NOTE: Copy the LOCAL IP address!                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ STEP 2: UPDATE ESP32 CODE ───────────────────────────────────────┐
│                                                                      │
│  In MASTER CODE:                                                 │
│                                                                      │
│  Change this line:                                               │
│  const char* serverUrl = "http://your-website.com/api/alert";    │
│                                                                      │
│  To:                                                              │
│  const char* serverUrl = "http://192.168.x.x:5000/api/alert";   │
│  (Use the IP from Step 1)                                         │
│                                                                      │
│  Upload both CAM CODE and MASTER CODE to ESP32 boards            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ STEP 3: UPDATE LIFE-PULSE CLIENT ────────────────────────────────┐
│                                                                      │
│  In life_pulse_v2.py, find line in _send_image_to_backend():    │
│                                                                      │
│  backend_url = "http://192.168.1.100:5000/api/image-upload"      │
│                                                                      │
│  Change to your backend IP:                                      │
│  backend_url = "http://192.168.x.x:5000/api/image-upload"        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ STEP 4: START LIFE-PULSE APPLICATION ────────────────────────────┐
│                                                                      │
│  Terminal 2 (On Raspberry Pi or your PC):                        │
│  $ python launcher.py                                             │
│                                                                      │
│  The application will:                                           │
│  ✅ Show device selection screen                                 │
│  ✅ Select device → Start monitoring                             │
│  ✅ When survival detected → Send image to backend              │
│  ✅ Backend returns → Graph switches to HEART RATE              │
│  ✅ Status updates with matched person name                     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ HOW IT WORKS ────────────────────────────────────────────────────┐
│                                                                      │
│  SEARCH MODE (Person Identification):                             │
│  ─────────────────────────────────────                             │
│  1. Sensor detects survivor (heartbeat/breathing)                │
│  2. Image captured from ESP32-CAM                                │
│  3. Sent to backend for face recognition                         │
│  4. Backend response:                                             │
│     ✅ Human found → Graph shows HEART RATE                      │
│     ✅ Matched in DB → Show person name                         │
│     ❌ No human → Reset and keep searching                      │
│                                                                      │
│  SENTRY MODE (Structural Monitoring):                             │
│  ────────────────────────────────────────                          │
│  1. Master device detects structural failure                     │
│  2. Sends "STRUCTURAL_FAILURE_WARNING" alert                     │
│  3. Backend receives alert                                        │
│  4. Life-Pulse shows: "STRUCTURAL INTEGRITY COMPROMISED"         │
│  5. Update stability_display with warning                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ TROUBLESHOOTING ────────────────────────────────────────────────┐
│                                                                      │
│  ❌ "Cannot connect to backend"                                   │
│  → Check IP address is correct                                  │
│  → Ensure backend_server.py is running                          │
│  → Check firewall allows port 5000                              │
│                                                                      │
│  ❌ "No face detected in image"                                   │
│  → Ensure good lighting for face detection                      │
│  → Face must be clearly visible                                 │
│                                                                      │
│  ❌ "No match found in database"                                  │
│  → Person must be enrolled first                                │
│  → Run manage_database.py to add people                         │
│  → Ensure photo_path is correct                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌─ DATA FLOW ──────────────────────────────────────────────────────┐
│                                                                      │
│  ESP32-CAM                                                         │
│      ↓ (JPEG image via WiFi)                                     │
│  Backend Server (5000)                                            │
│      ↓ (Face recognition)                                        │
│  Life-Pulse Client                                               │
│      ↓ (Updates UI/Graph)                                        │
│  Display (Heart Rate or No Detection)                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

""")
