"""
BioVerify: Government ID (Mock) Verifier
Simulated government ID verification for demonstration purposes.

⚠️ DISCLAIMER: This is a MOCK implementation for educational/demo purposes only.
It does NOT connect to any real government identity verification systems.
Aadhaar is used as an example of a government ID format.
This project does not perform real Aadhaar authentication.
"""

import re
import time
import random
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class GovIDResult:
    """Result of government ID verification."""
    verified: bool
    holder_name: str = ""
    masked_id: str = ""
    verification_method: str = "OTP"
    error_message: str = ""
    timestamp: str = ""
    
    def to_dict(self):
        return {
            "verified": self.verified,
            "holder_name": self.holder_name,
            "masked_id": self.masked_id,
            "verification_method": self.verification_method,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }


class GovIDVerifier:
    """
    Mock Government ID Verifier for demonstration.
    
    Simulates the verification process with:
    - Format validation
    - Artificial delay (simulating API call)
    - Randomized success/failure for demo
    - Mock OTP verification
    
    ⚠️ This is NOT a real government ID verification system.
    """
    
    # Mock success rate (100% for demo - no random failures)
    SUCCESS_RATE = 1.0
    
    # Sample names for mock verification
    SAMPLE_NAMES = [
        "Rahul Kumar", "Priya Sharma", "Amit Patel", "Sneha Singh",
        "Vikram Reddy", "Ananya Gupta", "Rohan Verma", "Kavita Nair",
        "Arjun Joshi", "Meera Iyer", "Sanjay Menon", "Divya Rao"
    ]
    
    def __init__(self, success_rate: float = None):
        """
        Initialize verifier.
        
        Args:
            success_rate: Mock success rate for verification (0.0-1.0)
        """
        self.success_rate = success_rate or self.SUCCESS_RATE
        self._pending_otp = None
        self._pending_id = None
    
    def validate_format(self, id_number: str) -> bool:
        """
        Validate government ID format.
        
        For demo purposes, accepts:
        - 12 digit numbers (Aadhaar-like format)
        - 10 character alphanumeric (PAN-like format)
        
        Args:
            id_number: The ID number to validate
            
        Returns:
            True if format is valid
        """
        # Remove spaces and dashes
        cleaned = re.sub(r'[\s\-]', '', id_number)
        
        # Check for 12-digit format (Aadhaar-like)
        if re.match(r'^\d{12}$', cleaned):
            return True
        
        # Check for 10-character alphanumeric (PAN-like)
        if re.match(r'^[A-Z]{5}\d{4}[A-Z]$', cleaned.upper()):
            return True
        
        return False
    
    def mask_id(self, id_number: str) -> str:
        """Mask ID number for display (show only last 4 digits)."""
        cleaned = re.sub(r'[\s\-]', '', id_number)
        if len(cleaned) >= 4:
            return 'X' * (len(cleaned) - 4) + cleaned[-4:]
        return 'X' * len(cleaned)
    
    def initiate_verification(self, id_number: str) -> tuple:
        """
        Initiate verification and generate mock OTP.
        
        Args:
            id_number: Government ID number
            
        Returns:
            Tuple of (success: bool, message: str, otp: str or None)
        """
        if not self.validate_format(id_number):
            return False, "Invalid ID format. Please enter a valid 12-digit ID.", None
        
        # Generate mock OTP
        otp = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        self._pending_otp = otp
        self._pending_id = id_number
        
        # Simulate network delay
        time.sleep(random.uniform(0.5, 1.5))
        
        masked = self.mask_id(id_number)
        return True, f"OTP sent to registered mobile (Demo OTP: {otp})", otp
    
    def verify_otp(self, entered_otp: str) -> GovIDResult:
        """
        Verify the entered OTP.
        
        Args:
            entered_otp: OTP entered by user
            
        Returns:
            GovIDResult with verification status
        """
        timestamp = datetime.now().isoformat()
        
        if not self._pending_otp or not self._pending_id:
            return GovIDResult(
                verified=False,
                error_message="No pending verification. Please initiate verification first.",
                timestamp=timestamp
            )
        
        # Simulate verification delay
        time.sleep(random.uniform(1.0, 2.5))
        
        # Check OTP
        if entered_otp != self._pending_otp:
            return GovIDResult(
                verified=False,
                masked_id=self.mask_id(self._pending_id),
                error_message="Invalid OTP. Please try again.",
                timestamp=timestamp
            )
        
        # Random success based on configured rate
        if random.random() > self.success_rate:
            return GovIDResult(
                verified=False,
                masked_id=self.mask_id(self._pending_id),
                error_message="Verification failed. Please try again or contact support.",
                timestamp=timestamp
            )
        
        # Success - generate mock holder info
        holder_name = random.choice(self.SAMPLE_NAMES)
        
        result = GovIDResult(
            verified=True,
            holder_name=holder_name,
            masked_id=self.mask_id(self._pending_id),
            verification_method="OTP",
            timestamp=timestamp
        )
        
        # Clear pending verification
        self._pending_otp = None
        self._pending_id = None
        
        return result
    
    def quick_verify(self, id_number: str, holder_name: str = None, simulate_delay: bool = True) -> GovIDResult:
        """
        Quick verification without OTP (for streamlined demo).
        
        Args:
            id_number: Government ID number
            holder_name: Name of the holder (if provided, uses this instead of random name)
            simulate_delay: Whether to simulate network delay
            
        Returns:
            GovIDResult with verification status
        """
        timestamp = datetime.now().isoformat()
        
        if not self.validate_format(id_number):
            return GovIDResult(
                verified=False,
                error_message="Invalid ID format. Please enter a valid 12-digit ID.",
                timestamp=timestamp
            )
        
        if simulate_delay:
            time.sleep(random.uniform(2.0, 3.5))
        
        # Random success based on configured rate
        if random.random() > self.success_rate:
            return GovIDResult(
                verified=False,
                masked_id=self.mask_id(id_number),
                error_message="Verification failed. Please try again.",
                timestamp=timestamp
            )
        
        # Success - use provided name or generate random one
        final_holder_name = holder_name if holder_name else random.choice(self.SAMPLE_NAMES)
        
        return GovIDResult(
            verified=True,
            holder_name=final_holder_name,
            masked_id=self.mask_id(id_number),
            verification_method="Direct",
            timestamp=timestamp
        )


# Singleton instance
_verifier_instance = None

def get_verifier(success_rate: float = None) -> GovIDVerifier:
    """Get government ID verifier instance."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = GovIDVerifier(success_rate)
    return _verifier_instance
