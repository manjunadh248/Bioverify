"""
BioVerify: Risk Analyzer Module
Intelligent risk assessment engine for new user registration and existing user login.
Implements configurable thresholds and reason code tracking for audit trails.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum


class RiskTier(Enum):
    """Risk classification tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ReasonCode(Enum):
    """Standardized reason codes for risk factors."""
    # Username patterns
    USERNAME_HIGH_NUMERIC = "USERNAME_HIGH_NUMERIC"
    USERNAME_RANDOM_PATTERN = "USERNAME_RANDOM_PATTERN"
    USERNAME_SUSPICIOUS_KEYWORDS = "USERNAME_SUSPICIOUS_KEYWORDS"
    
    # Email patterns
    EMAIL_DISPOSABLE_DOMAIN = "EMAIL_DISPOSABLE_DOMAIN"
    EMAIL_SUSPICIOUS_PATTERN = "EMAIL_SUSPICIOUS_PATTERN"
    
    # Account metrics
    ACCOUNT_LOW_FOLLOWERS = "ACCOUNT_LOW_FOLLOWERS"
    ACCOUNT_FOLLOW_RATIO_SUSPICIOUS = "ACCOUNT_FOLLOW_RATIO_SUSPICIOUS"
    ACCOUNT_NO_PROFILE_PIC = "ACCOUNT_NO_PROFILE_PIC"
    ACCOUNT_LOW_POSTS = "ACCOUNT_LOW_POSTS"
    ACCOUNT_NO_BIO = "ACCOUNT_NO_BIO"
    ACCOUNT_INCOMPLETE = "ACCOUNT_INCOMPLETE"
    
    # Device/Location
    DEVICE_NEW = "DEVICE_NEW"
    DEVICE_CHANGED = "DEVICE_CHANGED"
    LOCATION_CHANGED = "LOCATION_CHANGED"
    IP_HIGH_RISK = "IP_HIGH_RISK"
    
    # Behavior patterns
    BEHAVIOR_FREQUENT_LOGINS = "BEHAVIOR_FREQUENT_LOGINS"
    BEHAVIOR_SUSPICIOUS_ACTIVITY = "BEHAVIOR_SUSPICIOUS_ACTIVITY"
    BEHAVIOR_FAILED_VERIFICATIONS = "BEHAVIOR_FAILED_VERIFICATIONS"
    
    # Verification status
    NEVER_VERIFIED = "NEVER_VERIFIED"
    PREVIOUSLY_FAILED = "PREVIOUSLY_FAILED"
    PREVIOUSLY_FLAGGED = "PREVIOUSLY_FLAGGED"


@dataclass
class RiskResult:
    """Result of risk analysis with decision and explanation."""
    score: float  # 0-100
    tier: RiskTier
    decision: str
    reason_codes: List[str] = field(default_factory=list)
    requires_govid: bool = False
    requires_liveness: bool = False
    allow_login: bool = True
    explanation: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage/display."""
        return {
            "score": self.score,
            "tier": self.tier.value,
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "requires_govid": self.requires_govid,
            "requires_liveness": self.requires_liveness,
            "allow_login": self.allow_login,
            "explanation": self.explanation
        }


class RiskAnalyzer:
    """
    Risk analysis engine for BioVerify.
    
    Supports two flows:
    - Flow A: New user registration risk analysis
    - Flow B: Existing user login behavior analysis
    
    Thresholds are configurable via config.json or runtime parameters.
    """
    
    CONFIG_PATH = "config.json"
    
    # Disposable email domains (sample list)
    DISPOSABLE_DOMAINS = {
        "tempmail.com", "throwaway.com", "mailinator.com", "guerrillamail.com",
        "10minutemail.com", "temp-mail.org", "fakeinbox.com", "trashmail.com"
    }
    
    # Suspicious username keywords
    SUSPICIOUS_KEYWORDS = {
        "bot", "spam", "fake", "test", "temp", "admin", "official", "verified"
    }
    
    def __init__(self, low_threshold: int = None, high_threshold: int = None):
        """
        Initialize risk analyzer with configurable thresholds.
        
        Args:
            low_threshold: Score below this is LOW risk (default: 30)
            high_threshold: Score above this is HIGH risk (default: 70)
        """
        config = self._load_config()
        
        self.low_threshold = low_threshold or config.get("low_threshold", 30)
        self.high_threshold = high_threshold or config.get("high_threshold", 70)
        
        # Placeholder for future ML model integration
        self.ml_model = None
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if os.path.exists(self.CONFIG_PATH):
            try:
                with open(self.CONFIG_PATH, "r") as f:
                    config = json.load(f)
                    return config.get("risk_thresholds", {})
            except:
                pass
        return {}
    
    def _classify_tier(self, score: float) -> RiskTier:
        """Classify score into risk tier."""
        if score < self.low_threshold:
            return RiskTier.LOW
        elif score > self.high_threshold:
            return RiskTier.HIGH
        return RiskTier.MEDIUM
    
    def _calculate_username_risk(self, username: str) -> Tuple[float, List[str]]:
        """Analyze username for risk patterns."""
        score = 0
        codes = []
        
        if not username:
            return 20, [ReasonCode.USERNAME_RANDOM_PATTERN.value]
        
        # Numeric ratio
        digits = sum(c.isdigit() for c in username)
        numeric_ratio = digits / len(username)
        
        if numeric_ratio > 0.5:
            score += 25
            codes.append(ReasonCode.USERNAME_HIGH_NUMERIC.value)
        elif numeric_ratio > 0.3:
            score += 10
        
        # Random pattern detection (no vowels, too many consonants)
        vowels = sum(c.lower() in 'aeiou' for c in username if c.isalpha())
        letters = sum(c.isalpha() for c in username)
        
        if letters > 0 and vowels / letters < 0.15:
            score += 15
            codes.append(ReasonCode.USERNAME_RANDOM_PATTERN.value)
        
        # Suspicious keywords
        username_lower = username.lower()
        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword in username_lower:
                score += 20
                codes.append(ReasonCode.USERNAME_SUSPICIOUS_KEYWORDS.value)
                break
        
        return min(score, 40), codes
    
    def _calculate_email_risk(self, email_domain: str) -> Tuple[float, List[str]]:
        """Analyze email domain for risk."""
        score = 0
        codes = []
        
        if not email_domain:
            return 10, []
        
        domain_lower = email_domain.lower()
        
        # Disposable email check
        if domain_lower in self.DISPOSABLE_DOMAINS:
            score += 30
            codes.append(ReasonCode.EMAIL_DISPOSABLE_DOMAIN.value)
        
        # Suspicious patterns
        if re.match(r'^[a-z0-9]{10,}\.', domain_lower):
            score += 15
            codes.append(ReasonCode.EMAIL_SUSPICIOUS_PATTERN.value)
        
        return min(score, 35), codes
    
    def _calculate_account_completeness(self, account_data: Dict) -> Tuple[float, List[str]]:
        """Calculate account completeness score (inverse - higher = more risk)."""
        score = 0
        codes = []
        
        followers = account_data.get("followers", 0)
        following = account_data.get("following", 0)
        posts = account_data.get("posts", 0)
        has_pic = account_data.get("profile_pic", 0)
        bio_length = account_data.get("bio_length", 0)
        
        # ===== CHECK FOR CELEBRITY/INFLUENCER PROFILE =====
        # Real celebrities/influencers have: profile pic, bio, many posts
        is_complete_profile = has_pic and bio_length > 0 and posts >= 50
        is_likely_celebrity = followers > 1000000 and is_complete_profile and following >= 100
        
        # ===== EXTREME UNREALISTIC NUMBERS (Always suspicious) =====
        # Numbers that are impossibly high (more than population of Earth)
        if followers > 8000000000:  # More than 8 billion (world population)
            score += 50
            codes.append(ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value)
        
        # ===== SUSPICIOUS HIGH METRICS (Only if profile is incomplete) =====
        # High followers WITHOUT a complete profile = likely fake/bought
        elif not is_complete_profile:
            if followers > 10000000:  # More than 10 million without complete profile
                score += 35
                codes.append(ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value)
            elif followers > 1000000:  # More than 1 million without complete profile
                score += 20
                codes.append(ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value)
            elif followers > 100000:  # More than 100k without complete profile
                score += 10
        
        # ===== EXTREME FOLLOWER/FOLLOWING RATIO =====
        if followers > 0 and following > 0:
            ratio = followers / following
            # Very extreme ratios are always suspicious (even for celebrities)
            if ratio > 10000:  # 10000:1 ratio is always suspicious
                score += 45
                if ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value not in codes:
                    codes.append(ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value)
            elif ratio > 1000 and not is_likely_celebrity:  # 1000:1 for non-celebrities
                score += 30
                if ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value not in codes:
                    codes.append(ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value)
            elif ratio > 100 and not is_likely_celebrity:  # 100:1 for non-celebrities
                score += 15
                if ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value not in codes:
                    codes.append(ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value)
        
        # Opposite: Following much more than followers (follow-back scheme)
        if followers > 0 and following / followers > 5:
            score += 20
            if ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value not in codes:
                codes.append(ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value)
        elif following > 1000 and followers < 50:
            score += 25
            if ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value not in codes:
                codes.append(ReasonCode.ACCOUNT_FOLLOW_RATIO_SUSPICIOUS.value)
        
        # ===== LOW METRICS (Incomplete or new accounts) =====
        # Low followers
        if followers < 10:
            score += 15
            codes.append(ReasonCode.ACCOUNT_LOW_FOLLOWERS.value)
        elif followers < 50:
            score += 8
        
        # No profile picture
        if not has_pic:
            score += 20
            codes.append(ReasonCode.ACCOUNT_NO_PROFILE_PIC.value)
        
        # Low posts (suspicious if claiming high followers)
        if posts < 3:
            score += 15
            if followers > 10000:  # Extra suspicious if many followers but few posts
                score += 15
            codes.append(ReasonCode.ACCOUNT_LOW_POSTS.value)
        elif posts < 10:
            score += 8
            if followers > 100000:  # Extra suspicious if many followers but few posts
                score += 10
        
        # No bio
        if bio_length == 0:
            score += 10
            codes.append(ReasonCode.ACCOUNT_NO_BIO.value)
        
        # Overall incompleteness
        completeness = (
            (1 if has_pic else 0) * 0.3 +
            (1 if bio_length > 0 else 0) * 0.2 +
            (1 if posts > 5 else 0) * 0.25 +
            (1 if followers > 20 else 0) * 0.25
        )
        
        if completeness < 0.3:
            score += 10
            if ReasonCode.ACCOUNT_INCOMPLETE.value not in codes:
                codes.append(ReasonCode.ACCOUNT_INCOMPLETE.value)
        
        return min(score, 100), codes
    
    def analyze_new_user(self, account_data: Dict, device_info: str = "",
                         ip_risk: float = 0, email_domain: str = "") -> RiskResult:
        """
        Flow A: Analyze risk for new user registration.
        
        Args:
            account_data: Dict with account metrics (followers, posts, etc.)
            device_info: Device fingerprint string
            ip_risk: Simulated IP risk score (0-100)
            email_domain: Email domain for analysis
            
        Returns:
            RiskResult with decision and requirements
        """
        total_score = 0
        all_codes = []
        
        # 1. Username analysis (0-40 points)
        username = account_data.get("username", "")
        username_score, username_codes = self._calculate_username_risk(username)
        total_score += username_score
        all_codes.extend(username_codes)
        
        # 2. Email analysis (0-35 points)
        email_score, email_codes = self._calculate_email_risk(email_domain)
        total_score += email_score
        all_codes.extend(email_codes)
        
        # 3. Account completeness (0-50 points)
        account_score, account_codes = self._calculate_account_completeness(account_data)
        total_score += account_score
        all_codes.extend(account_codes)
        
        # 4. IP risk contribution (0-20 points)
        ip_contribution = ip_risk * 0.2
        if ip_contribution > 10:
            all_codes.append(ReasonCode.IP_HIGH_RISK.value)
        total_score += ip_contribution
        
        # 5. Placeholder for ML model prediction
        if self.ml_model is not None:
            # ml_score = self.ml_model.predict(features)
            # total_score = (total_score + ml_score) / 2
            pass
        
        # Normalize to 0-100
        final_score = min(max(total_score, 0), 100)
        tier = self._classify_tier(final_score)
        
        # Determine decision based on tier
        if tier == RiskTier.LOW:
            decision = "LIGHT_VERIFICATION"
            requires_govid = False
            requires_liveness = True  # Minimal liveness check
            allow_login = True
            explanation = "Low risk profile. Light verification only."
        elif tier == RiskTier.MEDIUM:
            decision = "FULL_VERIFICATION"
            requires_govid = True
            requires_liveness = True
            allow_login = True
            explanation = "Medium risk detected. Government ID and liveness verification required."
        else:  # HIGH
            decision = "BLOCK_MANUAL_REVIEW"
            requires_govid = True
            requires_liveness = True
            allow_login = False
            explanation = "High risk detected. Account blocked pending manual review."
        
        return RiskResult(
            score=round(final_score, 1),
            tier=tier,
            decision=decision,
            reason_codes=list(set(all_codes)),  # Remove duplicates
            requires_govid=requires_govid,
            requires_liveness=requires_liveness,
            allow_login=allow_login,
            explanation=explanation
        )
    
    def analyze_existing_user(self, user_record, current_device: str = "",
                              current_location: str = "") -> RiskResult:
        """
        Flow B: Analyze behavior risk for existing user login.
        
        Args:
            user_record: UserRecord from database
            current_device: Current device fingerprint
            current_location: Current location identifier
            
        Returns:
            RiskResult with re-verification requirements
        """
        total_score = 0
        all_codes = []
        
        # 1. Previous verification status
        if user_record.verification_status == "unverified":
            total_score += 30
            all_codes.append(ReasonCode.NEVER_VERIFIED.value)
        elif user_record.verification_status == "flagged":
            total_score += 50
            all_codes.append(ReasonCode.PREVIOUSLY_FLAGGED.value)
        elif user_record.verification_status == "blocked":
            total_score += 100
            all_codes.append(ReasonCode.PREVIOUSLY_FLAGGED.value)
        
        # 2. Previous liveness/govid status
        if not user_record.previous_liveness_passed:
            total_score += 20
            all_codes.append(ReasonCode.PREVIOUSLY_FAILED.value)
        
        # 3. Device change detection
        if current_device and user_record.last_device:
            if current_device != user_record.last_device:
                total_score += 25
                all_codes.append(ReasonCode.DEVICE_CHANGED.value)
        elif not user_record.last_device:
            total_score += 10
            all_codes.append(ReasonCode.DEVICE_NEW.value)
        
        # 4. Location change detection
        if current_location and user_record.last_location:
            if current_location != user_record.last_location:
                total_score += 15
                all_codes.append(ReasonCode.LOCATION_CHANGED.value)
        
        # 5. Suspicious flags from history
        flags = user_record.get_suspicious_flags()
        if len(flags) > 0:
            total_score += len(flags) * 10
            all_codes.append(ReasonCode.BEHAVIOR_SUSPICIOUS_ACTIVITY.value)
        
        # 6. Risk history trend
        risk_history = user_record.get_risk_history()
        if len(risk_history) >= 2:
            recent_scores = [r["score"] for r in risk_history[-3:]]
            avg_recent = sum(recent_scores) / len(recent_scores)
            if avg_recent > 50:
                total_score += 15
        
        # Normalize
        final_score = min(max(total_score, 0), 100)
        tier = self._classify_tier(final_score)
        
        # Decision based on tier + previous verification
        already_verified = (user_record.verification_status == "verified" and 
                           user_record.previous_liveness_passed)
        
        if tier == RiskTier.LOW and already_verified:
            decision = "LOGIN_ALLOWED"
            requires_govid = False
            requires_liveness = False
            allow_login = True
            explanation = "Low risk, previously verified. Login allowed."
        elif tier == RiskTier.LOW:
            decision = "LIGHT_VERIFICATION"
            requires_govid = False
            requires_liveness = True
            allow_login = True
            explanation = "Low risk but not fully verified. Liveness check required."
        elif tier == RiskTier.MEDIUM:
            decision = "LIVENESS_REQUIRED"
            requires_govid = False
            requires_liveness = True
            allow_login = True
            explanation = "Behavioral anomaly detected. Liveness re-verification required."
        else:  # HIGH
            decision = "FULL_REVERIFICATION"
            requires_govid = True
            requires_liveness = True
            allow_login = False
            explanation = "High risk behavior. Full re-verification with Government ID required."
        
        return RiskResult(
            score=round(final_score, 1),
            tier=tier,
            decision=decision,
            reason_codes=list(set(all_codes)),
            requires_govid=requires_govid,
            requires_liveness=requires_liveness,
            allow_login=allow_login,
            explanation=explanation
        )


# Singleton instance
_analyzer_instance = None

def get_analyzer(low_threshold: int = None, high_threshold: int = None) -> RiskAnalyzer:
    """Get risk analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None or low_threshold or high_threshold:
        _analyzer_instance = RiskAnalyzer(low_threshold, high_threshold)
    return _analyzer_instance
