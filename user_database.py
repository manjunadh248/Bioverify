"""
BioVerify: User Database Module
SQLite-based storage for user verification history and behavior tracking.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class UserRecord:
    """User record with verification and behavior data."""
    username: str
    aadhaar_hash: str = ""  # Hashed Aadhaar for privacy (prevents duplicate registration)
    email_domain: str = ""
    device_fingerprint: str = ""
    verification_status: str = "unverified"  # unverified, verified, flagged, blocked
    previous_liveness_passed: bool = False
    previous_govid_passed: bool = False
    login_count: int = 0
    last_login: str = ""
    last_device: str = ""
    last_location: str = ""
    risk_history: str = "[]"  # JSON array of risk assessments
    suspicious_flags: str = "[]"  # JSON array of flag codes
    face_encoding: str = ""  # JSON array of face landmarks for duplicate detection
    created_at: str = ""
    updated_at: str = ""
    
    def get_risk_history(self) -> List[Dict]:
        """Parse risk history JSON."""
        try:
            return json.loads(self.risk_history)
        except:
            return []
    
    def get_suspicious_flags(self) -> List[str]:
        """Parse suspicious flags JSON."""
        try:
            return json.loads(self.suspicious_flags)
        except:
            return []
    
    def add_risk_entry(self, score: float, tier: str, reason_codes: List[str]):
        """Add a risk assessment to history."""
        history = self.get_risk_history()
        history.append({
            "score": score,
            "tier": tier,
            "reason_codes": reason_codes,
            "timestamp": datetime.now().isoformat()
        })
        # Keep last 10 entries
        self.risk_history = json.dumps(history[-10:])
    
    def add_flag(self, flag_code: str):
        """Add a suspicious flag."""
        flags = self.get_suspicious_flags()
        if flag_code not in flags:
            flags.append(flag_code)
        self.suspicious_flags = json.dumps(flags)


def validate_aadhaar(aadhaar_number: str) -> tuple:
    """
    Validate Aadhaar number format.
    
    Args:
        aadhaar_number: The Aadhaar number to validate
        
    Returns:
        Tuple of (is_valid: bool, error_message: str, cleaned_aadhaar: str)
    """
    # Remove any spaces or dashes
    clean_aadhaar = aadhaar_number.replace(" ", "").replace("-", "")
    
    # Check if exactly 12 characters
    if len(clean_aadhaar) != 12:
        return False, f"Aadhaar must be exactly 12 digits. You entered {len(clean_aadhaar)} characters.", ""
    
    # Check if all digits
    if not clean_aadhaar.isdigit():
        return False, "Aadhaar must contain only numbers (0-9). No letters or special characters allowed.", ""
    
    # Check for invalid patterns (Aadhaar cannot start with 0 or 1)
    if clean_aadhaar[0] in ('0', '1'):
        return False, "Invalid Aadhaar number. Aadhaar cannot start with 0 or 1.", ""
    
    return True, "", clean_aadhaar


def hash_aadhaar(aadhaar_number: str) -> str:
    """Hash Aadhaar number for secure storage (one-way hash)."""
    # Remove any spaces or dashes
    clean_aadhaar = aadhaar_number.replace(" ", "").replace("-", "")
    # Use SHA-256 for secure hashing
    return hashlib.sha256(clean_aadhaar.encode()).hexdigest()


class UserDatabase:
    """SQLite database for user management."""
    
    DB_PATH = "data/users.db"
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or self.DB_PATH
        self._ensure_data_dir()
        self._init_db()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
    
    def _init_db(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                aadhaar_hash TEXT DEFAULT '',
                email_domain TEXT DEFAULT '',
                device_fingerprint TEXT DEFAULT '',
                verification_status TEXT DEFAULT 'unverified',
                previous_liveness_passed INTEGER DEFAULT 0,
                previous_govid_passed INTEGER DEFAULT 0,
                login_count INTEGER DEFAULT 0,
                last_login TEXT DEFAULT '',
                last_device TEXT DEFAULT '',
                last_location TEXT DEFAULT '',
                risk_history TEXT DEFAULT '[]',
                suspicious_flags TEXT DEFAULT '[]',
                face_encoding TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Migration: Add aadhaar_hash column to existing databases
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN aadhaar_hash TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        # Migration: Add face_encoding column to existing databases
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN face_encoding TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            # Column already exists, ignore
            pass
        
        # Index for fast lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_username ON users(username)
        """)
        
        # Index for Aadhaar hash lookups (to check duplicates)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_aadhaar_hash ON users(aadhaar_hash)
        """)
        
        conn.commit()
        conn.close()
    
    def _row_to_user(self, row: tuple, cursor) -> UserRecord:
        """Convert database row to UserRecord using column names."""
        # Get column names from cursor description
        columns = [description[0] for description in cursor.description]
        row_dict = dict(zip(columns, row))
        
        return UserRecord(
            username=row_dict.get('username', ''),
            aadhaar_hash=row_dict.get('aadhaar_hash', ''),
            email_domain=row_dict.get('email_domain', ''),
            device_fingerprint=row_dict.get('device_fingerprint', ''),
            verification_status=row_dict.get('verification_status', 'unverified'),
            previous_liveness_passed=bool(row_dict.get('previous_liveness_passed', 0)),
            previous_govid_passed=bool(row_dict.get('previous_govid_passed', 0)),
            login_count=int(row_dict.get('login_count', 0) or 0),
            last_login=row_dict.get('last_login', ''),
            last_device=row_dict.get('last_device', ''),
            last_location=row_dict.get('last_location', ''),
            risk_history=row_dict.get('risk_history', '[]'),
            suspicious_flags=row_dict.get('suspicious_flags', '[]'),
            face_encoding=row_dict.get('face_encoding', ''),
            created_at=row_dict.get('created_at', ''),
            updated_at=row_dict.get('updated_at', '')
        )
    
    def get_user(self, username: str) -> Optional[UserRecord]:
        """Get user by username."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (username.lower(),))
        row = cursor.fetchone()
        
        if row:
            user = self._row_to_user(row, cursor)
            conn.close()
            return user
        conn.close()
        return None
    
    def user_exists(self, username: str) -> bool:
        """Check if user exists."""
        return self.get_user(username) is not None
    
    def aadhaar_exists(self, aadhaar_number: str) -> tuple:
        """
        Check if Aadhaar is already registered (prevents duplicates).
        
        Returns:
            Tuple of (exists: bool, existing_username: str or None)
        """
        aadhaar_hashed = hash_aadhaar(aadhaar_number)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT username FROM users WHERE aadhaar_hash = ?", (aadhaar_hashed,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return True, row[0]
        return False, None
    
    def delete_user(self, username: str) -> bool:
        """
        Delete a user account by username.
        This allows the user to re-register with the same Aadhaar.
        
        Args:
            username: The username of the account to delete
            
        Returns:
            True if user was deleted, False if user not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if user exists first
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = ?", (username.lower(),))
        if cursor.fetchone()[0] == 0:
            conn.close()
            return False
        
        # Delete the user
        cursor.execute("DELETE FROM users WHERE username = ?", (username.lower(),))
        conn.commit()
        conn.close()
        
        return True
    
    def delete_user_by_aadhaar(self, aadhaar_number: str) -> tuple:
        """
        Delete a user account by Aadhaar number.
        This allows the user to re-register with the same Aadhaar.
        
        Args:
            aadhaar_number: The Aadhaar number of the account to delete
            
        Returns:
            Tuple of (success: bool, deleted_username: str or None)
        """
        aadhaar_hashed = hash_aadhaar(aadhaar_number)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get username before deleting
        cursor.execute("SELECT username FROM users WHERE aadhaar_hash = ?", (aadhaar_hashed,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False, None
        
        username = row[0]
        
        # Delete the user
        cursor.execute("DELETE FROM users WHERE aadhaar_hash = ?", (aadhaar_hashed,))
        conn.commit()
        conn.close()
        
        return True, username
    
    def get_user_by_aadhaar(self, aadhaar_number: str) -> Optional[UserRecord]:
        """Get user by Aadhaar number."""
        aadhaar_hashed = hash_aadhaar(aadhaar_number)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users WHERE aadhaar_hash = ?", (aadhaar_hashed,))
        row = cursor.fetchone()
        
        if row:
            user = self._row_to_user(row, cursor)
            conn.close()
            return user
        conn.close()
        return None
    
    def create_user_with_aadhaar(self, username: str, aadhaar_number: str,
                                  device_fingerprint: str = "") -> Optional[UserRecord]:
        """Create a new user with Aadhaar verification. Returns None if Aadhaar already exists."""
        # Check if Aadhaar is already registered (unpack the tuple)
        exists, _ = self.aadhaar_exists(aadhaar_number)
        if exists:
            return None  # Aadhaar already registered
        
        # Check if username exists
        if self.user_exists(username):
            return None  # Username taken
        
        aadhaar_hashed = hash_aadhaar(aadhaar_number)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO users (username, aadhaar_hash, device_fingerprint, 
                              created_at, updated_at, last_login, last_device)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username.lower(), aadhaar_hashed, device_fingerprint, now, now, now, device_fingerprint))
        
        conn.commit()
        conn.close()
        
        return self.get_user(username)
    
    def create_user(self, username: str, email_domain: str = "", 
                    device_fingerprint: str = "") -> UserRecord:
        """Create a new user (legacy method without Aadhaar)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO users (username, email_domain, device_fingerprint, 
                              created_at, updated_at, last_login, last_device)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (username.lower(), email_domain, device_fingerprint, now, now, now, device_fingerprint))
        
        conn.commit()
        conn.close()
        
        return self.get_user(username)
    
    def update_user(self, username: str, **updates) -> Optional[UserRecord]:
        """Update user fields."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates['updated_at'] = datetime.now().isoformat()
        
        # Build UPDATE query
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [username.lower()]
        
        cursor.execute(f"UPDATE users SET {set_clause} WHERE username = ?", values)
        conn.commit()
        conn.close()
        
        return self.get_user(username)
    
    def record_login(self, username: str, device_fingerprint: str = "", 
                     location: str = "") -> Optional[UserRecord]:
        """Record a login attempt and update counters."""
        user = self.get_user(username)
        if not user:
            return None
        
        new_count = user.login_count + 1
        
        return self.update_user(
            username,
            login_count=new_count,
            last_login=datetime.now().isoformat(),
            last_device=device_fingerprint,
            last_location=location
        )
    
    def update_verification_status(self, username: str, status: str,
                                    liveness_passed: bool = None,
                                    govid_passed: bool = None,
                                    risk_score: float = None,
                                    risk_tier: str = None,
                                    reason_codes: List[str] = None) -> Optional[UserRecord]:
        """Update verification status and optionally add risk entry."""
        updates = {"verification_status": status}
        
        if liveness_passed is not None:
            updates["previous_liveness_passed"] = int(liveness_passed)
        if govid_passed is not None:
            updates["previous_govid_passed"] = int(govid_passed)
        
        user = self.update_user(username, **updates)
        
        # Add risk history entry
        if user and risk_score is not None and risk_tier is not None:
            user.add_risk_entry(risk_score, risk_tier, reason_codes or [])
            self.update_user(username, risk_history=user.risk_history)
        
        return self.get_user(username)
    
    def add_suspicious_flag(self, username: str, flag_code: str) -> Optional[UserRecord]:
        """Add a suspicious flag to user."""
        user = self.get_user(username)
        if not user:
            return None
        
        user.add_flag(flag_code)
        return self.update_user(username, suspicious_flags=user.suspicious_flags)
    
    def get_all_users(self, limit: int = 100) -> List[UserRecord]:
        """Get all users for admin dashboard."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users ORDER BY updated_at DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        users = [self._row_to_user(row, cursor) for row in rows]
        conn.close()
        
        return users
    
    def get_flagged_users(self) -> List[UserRecord]:
        """Get users that are flagged or have suspicious flags."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM users 
            WHERE verification_status IN ('flagged', 'blocked')
               OR suspicious_flags != '[]'
            ORDER BY updated_at DESC
        """)
        rows = cursor.fetchall()
        users = [self._row_to_user(row, cursor) for row in rows]
        conn.close()
        
        return users
    
    def get_stats(self) -> Dict[str, int]:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM users")
        stats["total_users"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE verification_status = 'verified'")
        stats["verified_users"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE verification_status = 'unverified'")
        stats["unverified_users"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE verification_status = 'flagged'")
        stats["flagged_users"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM users WHERE verification_status = 'blocked'")
        stats["blocked_users"] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def get_all_face_encodings(self) -> List[Tuple[str, str]]:
        """
        Get all stored face encodings for duplicate comparison.
        
        Returns:
            List of (username, face_encoding_json) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT username, face_encoding FROM users 
            WHERE face_encoding IS NOT NULL AND face_encoding != ''
        """)
        rows = cursor.fetchall()
        conn.close()
        
        return [(row[0], row[1]) for row in rows]
    
    def update_face_encoding(self, username: str, face_encoding: str) -> Optional[UserRecord]:
        """
        Update user's face encoding.
        
        Args:
            username: Username to update
            face_encoding: JSON-serialized face encoding
            
        Returns:
            Updated user record or None if user not found
        """
        return self.update_user(username, face_encoding=face_encoding)


def generate_device_fingerprint(user_agent: str = "", screen_res: str = "", 
                                 timezone: str = "") -> str:
    """Generate a simple device fingerprint (simulated)."""
    data = f"{user_agent}|{screen_res}|{timezone}|{datetime.now().strftime('%Y%m%d')}"
    return hashlib.md5(data.encode()).hexdigest()[:16]


# Convenience functions
_db_instance = None

def get_db() -> UserDatabase:
    """Get singleton database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = UserDatabase()
    return _db_instance
