# 🔍 BioVerify: Multi-Modal Fake Account Detector

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31%2B-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

**BioVerify** is an advanced AI-powered system that combines **Machine Learning-based Risk Analysis**, **Real-time Biometric Verification**, and **Face Encoding Comparison** to detect fake social media accounts and prevent account fraud.

---

## 🌟 Key Features

### 🔐 Dual Flow System
- **Flow A (New Users)**: Registration with Aadhaar verification, risk analysis, and face enrollment
- **Flow B (Existing Users)**: Login with social media data analysis and face matching

### 🤖 Machine Learning Detection
- **XGBoost Classifier** trained on multi-dataset (Instagram + Twitter)
- **20+ engineered features** for comprehensive analysis
- **~93% accuracy** with cross-validation
- **SMOTE balancing** for handling imbalanced data

### 👁️ Biometric Verification
- **Multi-factor liveness detection**: Head movement + Eye blink detection
- **Face encoding capture** using MediaPipe Face Mesh
- **Duplicate face detection** for new user registration
- **Face mismatch detection** for existing user login
- **Real-time visual feedback** with countdown timer

### 🎯 Intelligent Risk Assessment
- **Context-aware scoring** - Distinguishes celebrities from fake accounts
- **Configurable thresholds** via `config.json`
- **Reason codes** for audit trails
- **Multi-level decisions**: Allow, Verify, Block

### 🗄️ User Database
- **SQLite database** for persistent user storage
- **Aadhaar hashing** for privacy (SHA-256)
- **Face encoding storage** for identity verification
- **Risk history tracking** for pattern analysis

### 📊 Admin Dashboard
- **Real-time statistics** on verifications
- **User management** with status tracking
- **Verification history** and reports

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      BioVerify System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  FLOW A: New User Registration                               │
│  ├── Step 0: User Type Selection                             │
│  ├── Step 1: Aadhaar + Username Verification                 │
│  │   ├── Aadhaar Format Validation                           │
│  │   ├── Duplicate Aadhaar Check                             │
│  │   └── Username Availability Check                         │
│  ├── Step 2: Biometric Verification                          │
│  │   ├── Liveness Detection (Head + Blinks)                  │
│  │   ├── Face Encoding Capture                               │
│  │   └── Duplicate Face Detection                            │
│  └── Step 3: Final Results & Account Creation                │
│                                                              │
│  FLOW B: Existing User Login                                 │
│  ├── Step 0: Account Lookup (Aadhaar or Username)            │
│  ├── Step 1: Social Media Risk Analysis                      │
│  │   ├── Platform Selection (Instagram/Twitter)              │
│  │   ├── Followers/Following/Posts Analysis                  │
│  │   └── Profile Completeness Check                          │
│  ├── Step 2: Conditional Verification                        │
│  │   ├── LOW Risk → Direct Login                             │
│  │   ├── MEDIUM Risk → Liveness + Face Match                 │
│  │   └── HIGH Risk → Block Account                           │
│  └── Step 3: Final Results                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
BioVerify/
├── data/
│   ├── train.csv                    # Instagram dataset
│   ├── fakeAccountData.json         # InstaFake fake accounts
│   ├── realAccountData.json         # InstaFake real accounts
│   ├── twitter_human_bots_dataset.csv  # Twitter dataset
│   ├── users.db                     # SQLite user database
│   └── verifications.json           # Verification history
├── models/
│   └── fake_account_model.pkl       # Trained XGBoost model
├── pages/
│   └── 1_Admin_Dashboard.py         # Admin dashboard page
├── app.py                           # Main Streamlit application
├── face_encoder.py                  # Face encoding module
├── risk_analyzer.py                 # Risk analysis engine
├── user_database.py                 # User database management
├── govid_verifier.py                # Mock Government ID verifier
├── utils.py                         # Liveness detection utilities
├── report_generator.py              # PDF/Text report generation
├── train_model.py                   # Single dataset training
├── train_multi_dataset.py           # Multi-dataset training
├── config.json                      # Configuration settings
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10 or 3.11** (recommended)
- **Webcam** (built-in or external)
- **4GB+ RAM**

### Installation

#### 1. Clone or Download Project

```bash
git clone https://github.com/yourusername/bioverify.git
cd BioVerify
```

#### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Train the Model (Optional - Pre-trained model included)

```bash
# Train with all datasets
python train_multi_dataset.py
```

#### 5. Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📖 Usage Guide

### New User Registration (Flow A)

1. **Select "New User"** on the welcome screen
2. **Enter Aadhaar Number** (12 digits) and **choose a username**
3. **Complete Liveness Test**:
   - Move your head (nod, turn, tilt) 3 times
   - OR blink naturally 3 times
4. **Account Created** with your face encoded for future verification

### Existing User Login (Flow B)

1. **Select "Existing User"** on the welcome screen
2. **Login with Aadhaar or Username**
3. **Enter Social Media Data**:
   - Platform (Instagram/Twitter)
   - Followers, Following, Posts count
   - Profile completeness info
4. **Risk Analysis Results**:
   - **LOW Risk**: Direct login allowed
   - **MEDIUM Risk**: Complete liveness + face verification
   - **HIGH Risk**: Account blocked

---

## 🧮 Risk Calculation

### Risk Score Tiers

| Score | Tier | Action |
|-------|------|--------|
| 0-29 | ✅ LOW | Login Allowed |
| 30-69 | ⚠️ MEDIUM | Verification Required |
| 70-100 | 🚫 HIGH | Account Blocked |

### Risk Factors (New Users)

| Factor | Risk Added |
|--------|------------|
| Suspicious username patterns | +10 to +20 |
| Disposable email domain | +15 |
| Low followers (<50) | +8 to +15 |
| No profile picture | +20 |
| Low posts (<10) | +8 to +15 |
| Suspicious follow ratio | +15 to +45 |

### Risk Factors (Existing Users)

| Factor | Risk Added |
|--------|------------|
| Unverified status | +30 |
| Previously flagged | +50 |
| Previously blocked | +100 |
| Device changed | +25 |
| Location changed | +15 |
| Failed previous liveness | +20 |

### Celebrity Detection

The system distinguishes real celebrities from fake accounts:

**Treated as Celebrity if ALL true:**
- 1M+ followers
- Profile picture ✓
- Bio present ✓
- 50+ posts ✓
- 100+ following ✓

**Fake/Bot Pattern:**
- High followers + incomplete profile
- Extreme follower/following ratio (>1000:1)
- Impossibly high numbers (>8 billion)

---

## 🔬 Technical Details

### Machine Learning Model

**Algorithm**: XGBoost Classifier

**Training Data**:
- Instagram dataset: 576 samples
- InstaFake dataset: 1,194 samples
- **Total**: 1,770 samples

**Performance**:
- Accuracy: ~93%
- AUC-ROC: ~98%
- Cross-validation: 5-fold

**Features Used** (20 total):
- Profile pic, bio length, external URL
- Followers, following, posts count
- Username patterns (numeric ratio)
- Engagement ratios
- Suspicious score (engineered)

### Face Encoding System

**Technology**: MediaPipe Face Mesh (468 landmarks)

**Key Components**:
- 50 key landmarks for encoding
- Normalized feature vectors
- Cosine similarity comparison
- **Threshold**: 85% similarity for match

**Duplicate Detection**:
- Compares new face against all stored encodings
- Blocks registration if face already exists
- Suggests login or account deletion

### Liveness Detection

**Dual-factor verification**:
1. **Head Movement**: Detects nods, turns, tilts
2. **Eye Blinks**: Uses Eye Aspect Ratio (EAR)

**Requirements**:
- 3 total actions (movements + blinks)
- Face visible throughout test
- ~30 second timeout

---

## ⚙️ Configuration

### config.json

```json
{
  "risk_thresholds": {
    "low": 30,
    "high": 70
  },
  "verification": {
    "liveness_duration": 30,
    "liveness_pass_threshold": 50
  },
  "ui": {
    "show_risk_details": true,
    "show_reason_codes": true
  }
}
```

---

## 🗄️ Database Schema

### Users Table

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| username | TEXT | Unique username |
| aadhaar_hash | TEXT | SHA-256 hashed Aadhaar |
| face_encoding | TEXT | JSON-serialized face encoding |
| verification_status | TEXT | verified/unverified/flagged/blocked |
| previous_liveness_passed | INTEGER | Boolean |
| previous_govid_passed | INTEGER | Boolean |
| risk_history | TEXT | JSON array of past assessments |
| created_at | TEXT | ISO timestamp |
| updated_at | TEXT | ISO timestamp |

---

## 🧪 Testing

### Setup Test Accounts

```bash
python setup_test_accounts.py
```

This creates test accounts with different statuses:
- 3 Unverified accounts
- 4 Blocked accounts
- 3 Flagged accounts
- 2 Pending accounts

### Test Scenarios

1. **Face Mismatch Test**: Login as another user's account → Should detect face mismatch
2. **No Face Encoding Test**: Login as `blocked_bot1` → Shows "No Biometric Record Found"
3. **High Risk Test**: Enter suspicious social media data → Should block

---

## 🐛 Troubleshooting

### Camera Access Denied
- **Chrome**: Click camera icon in address bar → Allow
- **Windows**: Settings → Privacy → Camera → Enable

### "Module not found" errors
```bash
pip install -r requirements.txt --upgrade
```

### XGBoost Warning
```bash
# Retrain model to fix serialization warning
python train_multi_dataset.py
```

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 📊 Version History

### v3.0 (Current) - Full Identity Verification System
- ✨ Dual flow system (New + Existing users)
- 🔐 Aadhaar-based registration
- 👤 Face encoding & duplicate detection
- 🎭 Face mismatch detection for existing users
- 📊 Multi-dataset ML training
- ⚙️ Configurable risk thresholds
- 🗄️ SQLite user database
- 📋 Admin dashboard

### v2.0 - Enhanced Multi-Step Verification
- ⏱️ Timed camera verification
- 📊 Percentage-based liveness scoring
- 🎯 VETO logic with thresholds

### v1.0 - Initial Release
- 🤖 ML-based account analysis
- 👁️ Basic liveness detection

---

<div align="center">

**Built with ❤️ using Python, Streamlit, XGBoost, and MediaPipe**

Made for detecting fake accounts and protecting online communities

</div>
