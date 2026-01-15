import sqlite3

def display_menu():
    print("\n" + "=" * 60)
    print("DATABASE MANAGEMENT TOOL")
    print("=" * 60)
    print("1. Add a new person")
    print("2. Update person details")
    print("3. Mark person as found (with device number)")
    print("4. View all records")
    print("5. Search person by name")
    print("6. Exit")
    print("=" * 60)

def add_person():
    full_name = input("Enter full name: ").strip()
    aadhar = input("Enter Aadhar number (optional): ").strip() or None
    phone = input("Enter phone number (optional): ").strip() or None
    photo_path = input("Enter photo path (optional, e.g., assets/person.jpg): ").strip() or None
    
    conn = sqlite3.connect('life_pulse.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO missing_persons (full_name, aadhar, phone_number, photo_path, status)
        VALUES (?, ?, ?, ?, ?)
    ''', (full_name, aadhar, phone, photo_path, ''))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    print(f"✓ Person added with ID: {new_id}")

def update_person():
    person_id = input("Enter person ID to update: ").strip()
    full_name = input("Enter new full name: ").strip()
    aadhar = input("Enter new Aadhar number: ").strip() or None
    phone = input("Enter new phone number: ").strip() or None
    
    conn = sqlite3.connect('life_pulse.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE missing_persons
        SET full_name = ?, aadhar = ?, phone_number = ?
        WHERE id = ?
    ''', (full_name, aadhar, phone, person_id))
    conn.commit()
    conn.close()
    print(f"✓ Person ID {person_id} updated!")

def mark_found():
    person_id = input("Enter person ID: ").strip()
    device_number = input("Enter device number that found them: ").strip()
    
    status = f"found using device {device_number}"
    
    conn = sqlite3.connect('life_pulse.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE missing_persons
        SET status = ?, found_timestamp = ?
        WHERE id = ?
    ''', (status, time.time(), person_id))
    conn.commit()
    conn.close()
    print(f"✓ Person ID {person_id} marked as: '{status}'")

def view_all():
    conn = sqlite3.connect('life_pulse.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, full_name, aadhar, phone_number, photo_path, status FROM missing_persons;")
    rows = cursor.fetchall()
    
    if not rows:
        print("No records found!")
    else:
        print("\n" + "=" * 100)
        for row in rows:
            print(f"\nID: {row[0]}")
            print(f"  Full Name: {row[1]}")
            print(f"  Aadhar: {row[2] if row[2] else 'N/A'}")
            print(f"  Phone: {row[3] if row[3] else 'N/A'}")
            print(f"  Photo: {row[4] if row[4] else 'No photo'}")
            print(f"  Status: {row[5] if row[5] else '(empty)'}")
        print("=" * 100)
    conn.close()

def search_person():
    search_name = input("Enter name to search: ").strip()
    
    conn = sqlite3.connect('life_pulse.db')
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, full_name, aadhar, phone_number, photo_path, status FROM missing_persons WHERE full_name LIKE ?;",
        (f'%{search_name}%',)
    )
    rows = cursor.fetchall()
    
    if not rows:
        print(f"No records found with name containing '{search_name}'")
    else:
        print("\n" + "=" * 100)
        for row in rows:
            print(f"\nID: {row[0]}")
            print(f"  Full Name: {row[1]}")
            print(f"  Aadhar: {row[2] if row[2] else 'N/A'}")
            print(f"  Phone: {row[3] if row[3] else 'N/A'}")
            print(f"  Photo: {row[4] if row[4] else 'No photo'}")
            print(f"  Status: {row[5] if row[5] else '(empty)'}")
        print("=" * 100)
    conn.close()

import time

while True:
    display_menu()
    choice = input("Select an option (1-6): ").strip()
    
    if choice == '1':
        add_person()
    elif choice == '2':
        update_person()
    elif choice == '3':
        mark_found()
    elif choice == '4':
        view_all()
    elif choice == '5':
        search_person()
    elif choice == '6':
        print("Goodbye!")
        break
    else:
        print("Invalid option. Please try again.")
