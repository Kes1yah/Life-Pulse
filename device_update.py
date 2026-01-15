import sqlite3
import time

def update_status_from_device(person_id, device_number):
    """
    Update person status when hardware device detects them.
    
    Args:
        person_id: ID of the person in database
        device_number: Device number that detected them
    """
    conn = sqlite3.connect('life_pulse.db')
    cursor = conn.cursor()
    
    new_status = f"found using device {device_number}"
    
    cursor.execute('''
        UPDATE missing_persons
        SET status = ?, found_timestamp = ?
        WHERE id = ?
    ''', (new_status, time.time(), person_id))
    
    conn.commit()
    conn.close()
    
    print(f"✓ Updated person ID {person_id}: '{new_status}'")
    return new_status


# EXAMPLE: Hardware device detected someone
if __name__ == "__main__":
    print("=" * 60)
    print("HARDWARE DEVICE STATUS UPDATE EXAMPLE")
    print("=" * 60)
    
    # Simulate device detecting person with ID 1
    update_status_from_device(person_id=1, device_number=5)
    
    # Show updated record
    conn = sqlite3.connect('life_pulse.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name, status FROM missing_persons WHERE id = 1;")
    row = cursor.fetchone()
    if row:
        print(f"\nUpdated Record:")
        print(f"ID: {row[0]}")
        print(f"Name: {row[1]}")
        print(f"Status: '{row[2]}'")
    conn.close()
    
    print("=" * 60)
