import sqlite3
import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Dict
import time
import cv2
import numpy as np

@dataclass
class MissingPerson:
    """Model for a missing person with full details"""
    id: int
    full_name: str
    aadhar: Optional[str] = None
    phone_number: Optional[str] = None
    photo_path: Optional[str] = None
    status: str = ""  # Empty by default, updated only when found by hardware device
    found_timestamp: Optional[float] = None

class DisasterDatabase:
    """
    Manages the SQLite database for identifying found survivors.
    Simplified version: only Name and Photo.
    """
    
    def __init__(self, db_path: str = "life_pulse.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
        
    def _get_connection(self):
        """Get a database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
        
    def _init_database(self):
        """Initialize the database schema"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create missing_persons table with new schema
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS missing_persons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT NOT NULL,
                    aadhar TEXT,
                    phone_number TEXT,
                    photo_path TEXT,
                    status TEXT DEFAULT '',
                    found_timestamp REAL
                )
            ''')
            
            # Create rescue_log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rescue_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    timestamp REAL,
                    notes TEXT,
                    FOREIGN KEY (person_id) REFERENCES missing_persons(id)
                )
            ''')

            # Create face_encodings table for multi-pattern support
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_encodings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    encoding BLOB,
                    photo_path TEXT,
                    FOREIGN KEY (person_id) REFERENCES missing_persons(id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
    def add_person(self, full_name: str, aadhar: str = None, phone_number: str = None, photo_path: str = None) -> int:
        """Add a new person to the missing list"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO missing_persons (full_name, aadhar, phone_number, photo_path) VALUES (?, ?, ?, ?)",
                (full_name, aadhar, phone_number, photo_path)
            )
            new_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return new_id
            
    def get_all_missing(self) -> List[MissingPerson]:
        """Retrieve all persons currently marked as MISSING"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM missing_persons WHERE status = 'MISSING'")
            rows = cursor.fetchall()
            conn.close()
            
            return [MissingPerson(
                id=r['id'], 
                name=r['name'], 
                photo_path=r['photo_path'],
                status=r['status'],
                found_timestamp=r['found_timestamp'],
                face_encoding=r['face_encoding']
            ) for r in rows]
            
    def get_next_missing(self, current_id: int = 0) -> Optional[MissingPerson]:
        """Get the next missing person after specific ID (for cycling display)"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM missing_persons WHERE status = 'MISSING' AND id > ? ORDER BY id LIMIT 1",
                (current_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                # Wrap around
                cursor.execute("SELECT * FROM missing_persons WHERE status = 'MISSING' ORDER BY id LIMIT 1")
                row = cursor.fetchone()
                
            conn.close()
            
            if row:
                return MissingPerson(
                    id=row['id'], 
                    name=row['name'], 
                    photo_path=row['photo_path'],
                    status=row['status'],
                    found_timestamp=row['found_timestamp'],
                    face_encoding=row['face_encoding']
                )
            return None

    def mark_as_found(self, person_id: int, rescuer_notes: str = "") -> bool:
        """Update person status to FOUND and log rescue"""
        with self._lock:
            ts = time.time()
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Update status
            cursor.execute(
                "UPDATE missing_persons SET status = 'FOUND', found_timestamp = ? WHERE id = ?",
                (ts, person_id)
            )
            success = cursor.rowcount > 0
            
            # Log rescue
            if success:
                cursor.execute(
                    "INSERT INTO rescue_log (person_id, timestamp, notes) VALUES (?, ?, ?)",
                    (person_id, ts, rescuer_notes)
                )
            
            conn.commit()
            conn.close()
            return success
            
    def get_person_by_photo(self, photo_path: str) -> Optional[MissingPerson]:
        """Match captured photo to database record (simulated matching)"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            target_filename = os.path.basename(photo_path)
            cursor.execute("SELECT * FROM missing_persons WHERE photo_path LIKE ?", (f"%{target_filename}",))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return MissingPerson(
                    id=row['id'],
                    name=row['name'],
                    photo_path=row['photo_path'],
                    status=row['status'],
                    found_timestamp=row['found_timestamp'],
                    face_encoding=row['face_encoding']
                )
            return None

    def update_face_encoding(self, person_id: int, encoding: bytes, photo_path: str):
        """Add a new face encoding to a person's profile"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 1. Add to new multi-encoding table
            cursor.execute(
                "INSERT INTO face_encodings (person_id, encoding, photo_path) VALUES (?, ?, ?)",
                (person_id, encoding, photo_path)
            )
            
            # 2. Also keep the 'primary' encoding and photo in the main table for legacy UI
            cursor.execute(
                "UPDATE missing_persons SET face_encoding = ?, photo_path = ? WHERE id = ?",
                (encoding, photo_path, person_id)
            )
            conn.commit()
            conn.close()

    def get_all_with_encodings(self) -> List[Dict]:
        """Get all persons and their multiple face encodings for training"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get all persons
            cursor.execute("SELECT * FROM missing_persons")
            persons = cursor.fetchall()
            
            result = []
            for p in persons:
                # Get all encodings for this person
                cursor.execute("SELECT encoding FROM face_encodings WHERE person_id = ?", (p['id'],))
                encodings = [row['encoding'] for row in cursor.fetchall()]
                
                # Fallback to legacy encoding if table migration not done
                if p['face_encoding'] and not encodings:
                    encodings = [p['face_encoding']]
                
                if encodings:
                    result.append({
                        "person": MissingPerson(
                            id=p['id'],
                            name=p['name'],
                            photo_path=p['photo_path'],
                            status=p['status'],
                            found_timestamp=p['found_timestamp']
                        ),
                        "encodings": encodings
                    })
            
            conn.close()
            return result

    def get_statistics(self) -> Dict[str, int]:
        """Get summary stats"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM missing_persons WHERE status = 'MISSING'")
            missing = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM missing_persons WHERE status = 'FOUND'")
            found = cursor.fetchone()[0]
            conn.close()
            return {"missing": missing, "found": found}

    def reset_all_to_missing(self):
        """Wipe FOUND statuses for re-testing"""
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE missing_persons SET status = 'MISSING', found_timestamp = NULL")
            cursor.execute("DELETE FROM rescue_log")
            conn.commit()
            conn.close()

class WebcamCamera:
    """Real webcam camera using OpenCV"""
    def __init__(self, assets_dir: str = "assets"):
        self.assets_dir = assets_dir
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
            
    def capture(self, save_path: Optional[str] = None, save_to_disk: bool = True) -> Optional[str]:
        """Capture a frame from the webcam
        
        Args:
            save_path: Path to save the image (optional)
            save_to_disk: If False, returns image in memory without saving to disk
        
        Returns:
            Path to saved image if save_to_disk=True, or "memory://captured_image" if save_to_disk=False
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return None
            
        # Allow camera to warm up
        for _ in range(5):
            cap.read()
            
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Store frame in memory for comparison
            self._last_captured_frame = frame
            
            if save_to_disk:
                if save_path is None:
                    save_path = os.path.join(self.assets_dir, f"capture_{int(time.time())}.png")
                cv2.imwrite(save_path, frame)
                return save_path
            else:
                # Return special marker indicating image is in memory
                return "memory://captured_image"
            
        return None
    
    def get_last_captured_frame(self):
        """Get the last captured frame from memory"""
        return getattr(self, '_last_captured_frame', None)

    def get_preview_frame(self):
        """Get a single frame for GUI preview"""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame
        return None

    def is_available(self) -> bool:
        cap = cv2.VideoCapture(0)
        available = cap.isOpened()
        cap.release()
        return available
