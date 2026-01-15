import sqlite3

conn = sqlite3.connect('life_pulse.db')
cursor = conn.cursor()

# See all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# View missing_persons table
print("\nMissing Persons:")
cursor.execute("SELECT id, full_name, aadhar, phone_number, photo_path, status FROM missing_persons;")
rows = cursor.fetchall()
for row in rows:
    print(f"\nID: {row[0]}")
    print(f"  Name: {row[1]}")
    print(f"  Aadhar: {row[2]}")
    print(f"  Phone: {row[3]}")
    print(f"  Photo: {row[4] if row[4] else 'No photo'}")
    print(f"  Status: '{row[5] if row[5] else '(empty)'}'")

conn.close()