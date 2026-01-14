import cv2
import numpy as np
import os
import sqlite3
import pickle
from database import DisasterDatabase

def enroll_person(name, image_filename):
    assets_dir = "assets"
    img_path = os.path.join(assets_dir, image_filename)
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not decode image.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogram equalization for better contrast
    gray = cv2.equalizeHist(gray)
    
    # Detect face (stricter parameters)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    detected_faces = face_cascade.detectMultiScale(gray, 1.1, 6) # Adjusted scaleFactor and minNeighbors
    
    if len(detected_faces) == 0:
        print(f"Error: No face detected for {name}.")
        return
        
    # Take the largest face
    faces = sorted(detected_faces, key=lambda f: f[2]*f[3], reverse=True)
    x, y, w, h = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (200, 200)) # Standardize
    
    # Encode to bytes
    _, buffer = cv2.imencode('.png', roi_gray)
    encoding_blob = buffer.tobytes()
    
    # Update Database
    db = DisasterDatabase()
    
    # Find person's ID (case-insensitive check or direct match)
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM missing_persons WHERE UPPER(name) = UPPER(?)", (name,))
    row = cursor.fetchone()
    person_id = row['id'] if row else None
    conn.close() # Close connection before doing update
    
    if person_id:
        db.update_face_encoding(person_id, encoding_blob, img_path)
        print(f"Successfully registered face for {name} (ID: {person_id})")
    else:
        # If not found, add
        person_id = db.add_person(name.upper(), img_path)
        db.update_face_encoding(person_id, encoding_blob, img_path)
        print(f"Added and registered face for {name} (New ID: {person_id})")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python enroll_survivor.py <NAME> <IMAGE_FILENAME>")
    else:
        enroll_person(sys.argv[1], sys.argv[2])
