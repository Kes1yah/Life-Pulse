from database import DisasterDatabase
import os

def setup_survivor_names():
    db = DisasterDatabase()
    names = ["ANAND", "EDWIN", "KESSIYA", "SREYA"]
    photo_path = os.path.join("assets", "placeholder.png")
    
    print("Registering survivor names in database...")
    for name in names:
        db.add_person(name, photo_path)
        print(f"  - Added: {name}")

if __name__ == "__main__":
    setup_survivor_names()
