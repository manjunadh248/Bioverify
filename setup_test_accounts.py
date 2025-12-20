import sqlite3
import hashlib
from datetime import datetime

conn = sqlite3.connect('data/users.db')
cur = conn.cursor()

# Delete all verified accounts
print("Deleting verified accounts...")
cur.execute("DELETE FROM users WHERE verification_status = 'verified'")
deleted = cur.rowcount
print(f"Deleted {deleted} verified accounts")

# Test accounts to create (without face encodings for testing)
test_accounts = [
    # Unverified accounts
    {"username": "test_unverified1", "status": "unverified", "aadhaar": "111122223333"},
    {"username": "test_unverified2", "status": "unverified", "aadhaar": "111122224444"},
    {"username": "new_user_pending", "status": "unverified", "aadhaar": "111122225555"},
    
    # Blocked accounts
    {"username": "blocked_bot1", "status": "blocked", "aadhaar": "222233334444"},
    {"username": "blocked_spam", "status": "blocked", "aadhaar": "222233335555"},
    {"username": "blocked_fake", "status": "blocked", "aadhaar": "222233336666"},
    {"username": "blocked_suspicious", "status": "blocked", "aadhaar": "222233337777"},
    
    # Flagged accounts  
    {"username": "flagged_review1", "status": "flagged", "aadhaar": "333344445555"},
    {"username": "flagged_behavior", "status": "flagged", "aadhaar": "333344446666"},
    {"username": "flagged_risk", "status": "flagged", "aadhaar": "333344447777"},
    
    # Pending accounts
    {"username": "pending_verify1", "status": "pending", "aadhaar": "444455556666"},
    {"username": "pending_verify2", "status": "pending", "aadhaar": "444455557777"},
]

print("\nCreating test accounts...")
now = datetime.now().isoformat()

for acc in test_accounts:
    # Hash the Aadhaar
    aadhaar_hash = hashlib.sha256(acc["aadhaar"].encode()).hexdigest()
    
    # Check if already exists
    cur.execute("SELECT username FROM users WHERE username = ?", (acc["username"],))
    if cur.fetchone():
        print(f"  {acc['username']} already exists, updating status")
        cur.execute("UPDATE users SET verification_status = ?, face_encoding = NULL WHERE username = ?",
                   (acc["status"], acc["username"]))
        continue
    
    # Insert new account with all required columns
    cur.execute("""
        INSERT INTO users (
            username, email_domain, device_fingerprint, last_device, last_location,
            verification_status, previous_liveness_passed, previous_govid_passed,
            suspicious_flags, risk_history, created_at, updated_at, 
            aadhaar_hash, face_encoding
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        acc["username"],   # username
        "gmail.com",       # email_domain
        "test_device",     # device_fingerprint
        None,              # last_device
        None,              # last_location
        acc["status"],     # verification_status
        0,                 # previous_liveness_passed
        0,                 # previous_govid_passed
        "[]",              # suspicious_flags (JSON)
        "[]",              # risk_history (JSON)
        now,               # created_at
        now,               # updated_at
        aadhaar_hash,      # aadhaar_hash
        None               # face_encoding (NULL - for testing)
    ))
    print(f"  Created: {acc['username']} ({acc['status']})")

conn.commit()

# Show final count
print("\n" + "="*50)
print("DATABASE SUMMARY:")
print("="*50)
cur.execute("SELECT verification_status, COUNT(*) FROM users GROUP BY verification_status")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]} accounts")

print("\nTest accounts ready for testing:")
cur.execute("SELECT username, verification_status FROM users ORDER BY verification_status, username")
for row in cur.fetchall():
    print(f"  {row[0]:<25} - {row[1]}")

conn.close()
print("\nDone! Refresh your Streamlit app and test with these accounts.")
