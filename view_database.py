import sqlite3

conn = sqlite3.connect('life_pulse.db')
cursor = conn.cursor()

# See all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables:", cursor.fetchall())

# View missing_persons table
cursor.execute("SELECT * FROM missing_persons;")
rows = cursor.fetchall()
for row in rows:
    print(row)

conn.close()