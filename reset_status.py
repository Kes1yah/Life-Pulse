import sqlite3

conn = sqlite3.connect('life_pulse.db')
cursor = conn.cursor()

print("Updating database schema and records...")
print("=" * 60)

# Update all existing records to have empty status (instead of MISSING)
print("1. Setting all status fields to empty...")
cursor.execute("UPDATE missing_persons SET status = '' WHERE status IS NOT NULL;")
conn.commit()
print("   ✓ All status fields are now empty")

# Show current data
print("\n2. Current database records:")
print("-" * 60)
cursor.execute("SELECT id, full_name, aadhar, phone_number, status FROM missing_persons;")
rows = cursor.fetchall()
for row in rows:
    status_display = f"'{row[4]}'" if row[4] else "'(empty)'"
    print(f"ID: {row[0]}, Name: {row[1]}, Aadhar: {row[2]}, Phone: {row[3]}, Status: {status_display}")

print("\n" + "=" * 60)
print("✓ Database updated successfully!")
print("\nNote: Status will only be updated when hardware device sends data")
print("Format: 'found using device [number]'")
print("=" * 60)

conn.close()
