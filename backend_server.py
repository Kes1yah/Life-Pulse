"""
Life-Pulse Backend Server
==========================
Receives images from ESP32-CAM via WiFi
Performs face recognition against database
Returns human presence confirmation
"""

from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from database import DisasterDatabase
import io
from PIL import Image
import socket

app = Flask(__name__)
db = DisasterDatabase()

# Get local IP for ESP32 to connect
def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for ESP32"""
    return jsonify({
        "status": "online",
        "server": "Life-Pulse Backend",
        "ready_for_images": True
    })

@app.route('/api/image-upload', methods=['POST'])
def receive_image():
    """
    Receive image from ESP32-CAM
    Perform face recognition
    Return: human_presence, matched_person, confidence
    """
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image in request",
                "human_presence": False
            }), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({
                "success": False,
                "error": "Empty filename",
                "human_presence": False
            }), 400
        
        # Convert image to OpenCV format
        img = Image.open(image_file.stream)
        captured_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        print(f"[BACKEND] Image received: {image_file.filename} ({captured_img.shape})")
        
        # Try to match face in database
        matched_person, confidence = perform_face_recognition(captured_img)
        
        if matched_person:
            print(f"[BACKEND] ✅ MATCH FOUND: {matched_person.name} (confidence: {confidence:.2f})")
            return jsonify({
                "success": True,
                "human_presence": True,
                "matched_person": matched_person.name,
                "person_id": matched_person.id,
                "confidence": float(confidence),
                "aadhar": matched_person.aadhar,
                "phone": matched_person.phone_number
            }), 200
        else:
            print(f"[BACKEND] ⚠️  No match found in database")
            return jsonify({
                "success": True,
                "human_presence": True,  # Human detected but not in database
                "matched_person": None,
                "confidence": 0.0,
                "message": "Human detected but no match in database"
            }), 200
        
    except Exception as e:
        print(f"[BACKEND ERROR] Image processing failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "human_presence": False
        }), 500

@app.route('/api/alert', methods=['POST'])
def receive_alert():
    """
    Receive alerts from Master Device
    - LIFE_FOUND_ID_REQ: Person identified in LIFE mode
    - STRUCTURAL_FAILURE_WARNING: Structural integrity compromised in SENTRY mode
    """
    try:
        data = request.get_json()
        device = data.get('device', 'Unknown')
        alert = data.get('alert', 'Unknown')
        
        print(f"[BACKEND] 📢 ALERT from {device}: {alert}")
        
        return jsonify({
            "status": "received",
            "device": device,
            "alert": alert,
            "server_timestamp": str(__import__('datetime').datetime.now())
        }), 200
        
    except Exception as e:
        print(f"[BACKEND ERROR] Alert processing failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

def perform_face_recognition(captured_img):
    """
    Perform face recognition using LBPH
    Returns: (matched_person, confidence_score)
    """
    try:
        # Create recognizer
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Get all persons with encodings
        entries = db.get_all_with_encodings()
        if not entries:
            return None, 0.0
        
        faces = []
        labels = []
        id_map = {}
        
        label_idx = 0
        for entry in entries:
            person = entry['person']
            encodings = entry['encodings']
            
            for encoding_blob in encodings:
                nparr = np.frombuffer(encoding_blob, np.uint8)
                face_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
                if face_img is not None:
                    faces.append(face_img)
                    labels.append(label_idx)
                    id_map[label_idx] = person
            
            label_idx += 1
        
        if not faces:
            return None, 0.0
        
        recognizer.train(faces, np.array(labels))
        
        # Detect face in captured image
        gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 6)
        
        if len(detected_faces) == 0:
            print("[BACKEND] No face detected in image")
            return None, 0.0
        
        # Use largest face
        x, y, w, h = max(detected_faces, key=lambda f: f[2] * f[3])
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (200, 200))
        
        # Predict
        label, confidence = recognizer.predict(roi_gray)
        
        # Confidence threshold: lower is better (0-100, typically <50 is good match)
        if confidence < 70:
            matched_person = id_map[label]
            return matched_person, 100 - confidence  # Convert to 0-100 where 100 is perfect match
        else:
            return None, 0.0
        
    except Exception as e:
        print(f"[BACKEND] Face recognition error: {e}")
        return None, 0.0

if __name__ == '__main__':
    local_ip = get_local_ip()
    print("\n" + "="*60)
    print("  LIFE-PULSE BACKEND SERVER")
    print("="*60)
    print(f"  Local IP: {local_ip}")
    print(f"  Port: 5000")
    print(f"\n  📱 ESP32 should connect to: http://{local_ip}:5000")
    print(f"  Image upload endpoint: http://{local_ip}:5000/api/image-upload")
    print(f"  Alert endpoint: http://{local_ip}:5000/api/alert")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
