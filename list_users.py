import sqlite3

conn = sqlite3.connect('data/users.db')
cur = conn.cursor()
cur.execute('SELECT username, verification_status, face_encoding IS NOT NULL as has_face FROM users')

print("Existing Accounts in Database:")
print("-" * 50)
for row in cur.fetchall():
    face_status = "Yes" if row[2] else "No"
    print(f"Username: {row[0]:<20} Status: {row[1]:<12} Face: {face_status}")
print("-" * 50)
conn.close()
