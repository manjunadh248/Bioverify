"""
BioVerify: Multi-Modal Fake Account Detector
Risk-Based Verification System with Flow A (New Users) and Flow B (Existing Users)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import cv2
import json
from utils import LivenessDetector
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Import report generator
try:
    from report_generator import generate_pdf_report, generate_simple_report, REPORTLAB_AVAILABLE
except:
    REPORTLAB_AVAILABLE = False

# Import risk-based verification modules
from risk_analyzer import RiskAnalyzer, RiskTier, get_analyzer
from user_database import UserDatabase, UserRecord, get_db, generate_device_fingerprint, validate_aadhaar
from govid_verifier import GovIDVerifier, get_verifier
from face_encoder import FaceEncoder, get_encoder, find_matching_face

# Load configuration
def load_config():
    """Load configuration from config.json"""
    config_path = "config.json"
    default_config = {
        "risk_thresholds": {"low": 30, "high": 70},
        "verification": {"liveness_duration": 20, "liveness_pass_threshold": 50},
        "ui": {"show_risk_details": True, "show_reason_codes": True}
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except:
            return default_config
    return default_config

CONFIG = load_config()

# Page configuration
st.set_page_config(
    page_title="BioVerify: AI-Powered Fake Account Detector",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS with vibrant colors and dynamic effects
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 0;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* SIDEBAR STYLING - Dark Premium Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 50%, #16213e 100%) !important;
        border-right: 1px solid rgba(102, 126, 234, 0.3) !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 50%, #16213e 100%) !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    [data-testid="stSidebar"] span {
        color: rgba(255, 255, 255, 0.95) !important;
    }
    
    [data-testid="stSidebar"] p {
        color: rgba(255, 255, 255, 0.85) !important;
    }
    
    /* Sidebar Navigation Links */
    [data-testid="stSidebar"] a {
        color: #a8edea !important;
        text-decoration: none !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebar"] a:hover {
        color: #f093fb !important;
        text-shadow: 0 0 10px rgba(240, 147, 251, 0.5) !important;
    }
    
    /* Sidebar Selection */
    [data-testid="stSidebarNav"] {
        background: transparent !important;
    }
    
    [data-testid="stSidebarNav"] li {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
        margin: 5px 10px !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stSidebarNav"] li:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        transform: translateX(5px) !important;
        box-shadow: 0 0 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    [data-testid="stSidebarNav"] li[data-selected="true"] {
        background: linear-gradient(90deg, rgba(102, 126, 234, 0.4), rgba(118, 75, 162, 0.4)) !important;
        border-left: 3px solid #f093fb !important;
        box-shadow: 0 0 20px rgba(240, 147, 251, 0.3) !important;
    }
    
    /* Sidebar close button */
    [data-testid="stSidebar"] [data-testid="baseButton-header"] {
        color: white !important;
    }
    
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 5rem;
        font-weight: 900;
        color: #f093fb;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 3px;
        text-shadow: 0 0 30px #667eea, 0 0 60px #764ba2, 0 0 90px #f093fb;
        transition: all 0.5s ease;
        cursor: default;
    }
    
    .main-header:hover {
        text-shadow: 0 0 40px #667eea, 0 0 80px #764ba2, 0 0 120px #f093fb, 0 0 150px #667eea;
        transform: scale(1.02);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #a8edea;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
        text-shadow: 0 0 15px rgba(168, 237, 234, 0.5);
        transition: all 0.5s ease;
    }
    
    .sub-header:hover {
        text-shadow: 0 0 25px rgba(168, 237, 234, 0.8), 0 0 40px rgba(254, 214, 227, 0.6);
    }
    
    .content-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 25px;
        padding: 35px;
        box-shadow: 0 15px 50px rgba(0,0,0,0.3);
        margin: 20px 0;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .content-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transform: rotate(45deg);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .content-box:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .camera-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 30px;
        padding: 8px;
        box-shadow: 0 0 40px rgba(102, 126, 234, 0.6), 0 0 80px rgba(118, 75, 162, 0.4);
        animation: cameraGlow 2s ease-in-out infinite alternate;
        position: relative;
        overflow: hidden;
        margin: 0 auto;
        max-width: 900px;
    }
    
    @keyframes cameraGlow {
        from {
            box-shadow: 0 0 40px rgba(102, 126, 234, 0.6), 0 0 80px rgba(118, 75, 162, 0.4);
        }
        to {
            box-shadow: 0 0 60px rgba(118, 75, 162, 0.8), 0 0 120px rgba(240, 147, 251, 0.6);
        }
    }
    
    .camera-container::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300%;
        border-radius: 30px;
        z-index: -1;
        animation: borderFlow 3s linear infinite;
    }
    
    @keyframes borderFlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }
    
    .camera-stats {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 30px auto;
        flex-wrap: wrap;
        max-width: 100%;
        padding: 0 20px;
    }
    
    .stat-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.4);
        padding: 20px 15px;
        border-radius: 20px;
        text-align: center;
        min-width: 140px;
        max-width: 180px;
        flex: 1 1 140px;
        transition: all 0.3s ease;
        overflow: hidden;
    }
    
    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
    }
    
    .stat-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.5rem;
        font-weight: 900;
        color: white;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        margin: 5px 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.9);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .step-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
        border: 2px solid rgba(255, 255, 255, 0.2);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .step-card::before {
        content: '';
        position: absolute;
        width: 200%;
        height: 200%;
        top: -50%;
        left: -50%;
        background: conic-gradient(from 0deg, transparent, rgba(255,255,255,0.15), transparent 30%);
        animation: rotate 4s linear infinite;
    }
    
    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }
    
    .step-card-active {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        animation: activePulse 2s ease-in-out infinite;
        box-shadow: 0 0 50px rgba(240, 147, 251, 0.8);
        transform: scale(1.05);
    }
    
    @keyframes activePulse {
        0%, 100% { transform: scale(1.05); }
        50% { transform: scale(1.08); }
    }
    
    .step-card-completed {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 10px 40px rgba(56, 239, 125, 0.4);
    }
    
    .step-card-inactive {
        background: linear-gradient(135deg, #434343 0%, #000000 100%);
        opacity: 0.5;
    }
    
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-left: 6px solid #0b7a6f;
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        animation: slideInLeft 0.6s ease-out;
        box-shadow: 0 10px 30px rgba(56, 239, 125, 0.3);
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .danger-box {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border-left: 6px solid #c52a38;
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        animation: shake 0.6s ease-in-out;
        box-shadow: 0 10px 30px rgba(235, 51, 73, 0.4);
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
        20%, 40%, 60%, 80% { transform: translateX(10px); }
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-left: 6px solid #d84460;
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        animation: slideInRight 0.6s ease-out;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .info-box {
        background: rgba(102, 126, 234, 0.15);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.5);
        border-left: 6px solid #667eea;
        padding: 25px;
        border-radius: 15px;
        color: white;
        margin: 15px 0;
        animation: fadeIn 0.6s ease-out;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .verdict-box-real {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border: 4px solid #0b7a6f;
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 30px 0;
        animation: slideInLeft 0.8s ease-out;
        box-shadow: 0 20px 60px rgba(56, 239, 125, 0.5);
    }
    
    .verdict-box-fake {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        border: 4px solid #c52a38;
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 30px 0;
        animation: shake 0.8s ease-in-out;
        box-shadow: 0 20px 60px rgba(235, 51, 73, 0.5);
    }
    
    .verdict-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        margin: 20px 0;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.4);
        padding: 30px;
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-15px) scale(1.05);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.6);
        border-color: rgba(240, 147, 251, 0.6);
    }
    
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 10px 0;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 3px;
        font-weight: 600;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 200%;
        color: white;
        border: none;
        padding: 18px 45px;
        font-size: 1.3rem;
        font-weight: 700;
        border-radius: 50px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.5);
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: 2px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 20px 60px rgba(240, 147, 251, 0.7);
        background-position: 100%;
    }
    
    .badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 25px;
        font-size: 1rem;
        font-weight: 700;
        margin: 8px;
        animation: fadeIn 0.6s ease-out;
        border: 2px solid rgba(255, 255, 255, 0.3);
        letter-spacing: 1px;
        transition: all 0.3s ease;
    }
    
    .badge:hover {
        transform: scale(1.1);
        box-shadow: 0 5px 20px rgba(255, 255, 255, 0.3);
    }
    
    .badge-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .footer {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.3);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        margin-top: 60px;
        box-shadow: 0 -10px 40px rgba(102, 126, 234, 0.3);
    }
    
    h2, h3 {
        color: white !important;
        position: relative;
        z-index: 1;
    }
    
    p {
        color: rgba(255, 255, 255, 0.9) !important;
        position: relative;
        z-index: 1;
    }
    
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid rgba(102, 126, 234, 0.5) !important;
        color: #1a1a2e !important;
        border-radius: 10px !important;
        padding: 12px !important;
        font-size: 1.1rem !important;
    }
    
    .stTextInput>div>div>input::placeholder {
        color: #888 !important;
        opacity: 0.7 !important;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5) !important;
        outline: none !important;
    }
    
    label {
        color: #f093fb !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        font-size: 1.1rem !important;
        text-shadow: 0 0 10px rgba(240, 147, 251, 0.5) !important;
    }
    
    /* Ensure equal column heights for alignment */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="column"] > div {
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)

# Handle reset query parameter
query_params = st.query_params
if query_params.get("reset") == "true":
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Clear the reset param to avoid loop
    st.query_params.clear()

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0  # 0 = User type selection
if 'liveness_score' not in st.session_state:
    st.session_state.liveness_score = 0
if 'liveness_passed' not in st.session_state:
    st.session_state.liveness_passed = False
if 'blink_count' not in st.session_state:
    st.session_state.blink_count = 0
if 'face_detected' not in st.session_state:
    st.session_state.face_detected = False
if 'account_fake_prob' not in st.session_state:
    st.session_state.account_fake_prob = 0
if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {}
if 'verification_complete' not in st.session_state:
    st.session_state.verification_complete = False

# New risk-based flow session state
if 'user_type' not in st.session_state:
    st.session_state.user_type = None  # "new" or "existing"
if 'risk_result' not in st.session_state:
    st.session_state.risk_result = None  # RiskResult object
if 'govid_verified' not in st.session_state:
    st.session_state.govid_verified = False
if 'govid_result' not in st.session_state:
    st.session_state.govid_result = None
if 'device_fingerprint' not in st.session_state:
    st.session_state.device_fingerprint = generate_device_fingerprint()
if 'existing_user_record' not in st.session_state:
    st.session_state.existing_user_record = None
if 'verification_attempts' not in st.session_state:
    st.session_state.verification_attempts = 0
if 'no_face_detected' not in st.session_state:
    st.session_state.no_face_detected = False

def load_model():
    """Load trained model or return None with demo mode"""
    model_path = 'models/fake_account_model.pkl'
    
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            return model_data['model'], model_data['feature_names'], model_data['accuracy'], True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None, None, False
    else:
        return None, None, None, False

def demo_mode_prediction(features):
    """Simple rule-based demo prediction when model not available"""
    followers = features.get('followers', 0)
    following = features.get('following', 0)
    has_pic = features.get('profile_pic', 1)
    posts = features.get('posts', 0)
    
    risk_score = 0
    
    if followers < 100:
        risk_score += 30
    if following > followers * 2 and followers < 500:
        risk_score += 25
    if has_pic == 0:
        risk_score += 20
    if posts < 10:
        risk_score += 15
    if features.get('private', 0) == 1 and followers < 50:
        risk_score += 10
    
    return min(risk_score, 100)

def predict_fake_probability(model, feature_names, user_inputs):
    """Predict fake account probability with engineered features"""
    import numpy as np
    
    # Calculate engineered features
    followers = user_inputs.get('followers', 0)
    following = user_inputs.get('following', 0)
    posts = user_inputs.get('posts', 0)
    profile_pic = user_inputs.get('profile_pic', 1)
    bio_length = user_inputs.get('bio_length', 0)
    username_ratio = user_inputs.get('username_ratio', 0)
    
    # Engineered features
    followers_following_ratio = followers / (following + 1)
    posts_per_follower = posts / (followers + 1)
    engagement_potential = posts * followers / (following + 1)
    log_followers = np.log1p(followers)
    log_following = np.log1p(following)
    log_posts = np.log1p(posts)
    username_suspicion = username_ratio * (1 - profile_pic)
    profile_completeness = (
        profile_pic * 0.3 +
        (1 if bio_length > 0 else 0) * 0.3 +
        (1 if posts > 5 else 0) * 0.2 +
        (1 if followers > 10 else 0) * 0.2
    )
    suspicious_score = (
        (1 if following > followers * 3 else 0) * 0.3 +
        (1 if posts < 3 else 0) * 0.2 +
        (1 if profile_pic == 0 else 0) * 0.3 +
        (1 if username_ratio > 0.3 else 0) * 0.2
    )
    
    # Map all features
    feature_map = {
        '#followers': followers,
        '#follows': following,
        '#posts': posts,
        'profile pic': profile_pic,
        'description length': bio_length,
        'external URL': user_inputs.get('external_url', 0),
        'private': user_inputs.get('private', 0),
        'nums/length username': username_ratio,
        'fullname words': user_inputs.get('fullname_words', 0),
        'nums/length fullname': user_inputs.get('fullname_ratio', 0),
        'name==username': user_inputs.get('name_equals_username', 0),
        'followers_following_ratio': followers_following_ratio,
        'posts_per_follower': posts_per_follower,
        'engagement_potential': engagement_potential,
        'log_followers': log_followers,
        'log_following': log_following,
        'log_posts': log_posts,
        'username_suspicion': username_suspicion,
        'profile_completeness': profile_completeness,
        'suspicious_score': suspicious_score
    }
    
    feature_values = []
    for feature in feature_names:
        feature_values.append(feature_map.get(feature, 0))
    
    features_array = np.array(feature_values).reshape(1, -1)
    
    # Handle potential NaN/Inf
    features_array = np.nan_to_num(features_array, nan=0, posinf=1e10, neginf=-1e10)
    
    # Check if model has scaler
    probability = model.predict_proba(features_array)[0][1]
    
    return probability * 100

def create_animated_gauge(value, title, threshold=50):
    """Create an animated gauge chart with vibrant colors"""
    color = "#eb3349" if value >= 70 else "#f093fb" if value >= 40 else "#38ef7d"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"<b>{title}</b>", 'font': {'size': 24, 'color': 'white'}},
        delta={'reference': threshold, 'increasing': {'color': "#eb3349"}, 'decreasing': {'color': "#38ef7d"}},
        number={'font': {'size': 50, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white", 'tickfont': {'size': 14, 'color': 'white'}},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255, 255, 255, 0.1)",
            'borderwidth': 3,
            'bordercolor': "rgba(255, 255, 255, 0.3)",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(56, 239, 125, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(240, 147, 251, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(235, 51, 73, 0.2)'}
            ],
            'threshold': {'line': {'color': "#f093fb", 'width': 5}, 'thickness': 0.75, 'value': threshold}
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Orbitron', 'color': 'white'}
    )
    return fig

def calculate_liveness_percentage(blink_count, face_detected_duration, total_duration=20):
    """Calculate liveness percentage"""
    blink_score = min(blink_count * 20, 60)
    face_score = (face_detected_duration / total_duration) * 40
    return min(blink_score + face_score, 100)

def run_liveness_test_with_timer(duration=30):
    """Run multi-factor liveness test with challenges and face encoding capture"""
    st.markdown("""
    <div class="info-box">
        <h3 style="margin: 0;">🎥 Multi-Factor Liveness Verification</h3>
        <p style="margin: 10px 0 0 0;">Complete the challenges shown on screen: Move head, Smile, or Raise hand</p>
    </div>
    """, unsafe_allow_html=True)
    
    detector = LivenessDetector(num_challenges=3)
    face_encoder = get_encoder()  # Get face encoder for capturing face encoding
    face_encoder.clear_history()  # Clear any previous samples
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Camera access denied. Please enable camera permissions.")
        return 0, 0, False, None
    
    # Camera display
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Stats below camera - 4 columns for detailed stats
    st.markdown('<div class="camera-stats">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        timer_placeholder = st.empty()
    with col2:
        moves_placeholder = st.empty()
    with col3:
        blinks_placeholder = st.empty()
    with col4:
        face_status_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    progress_placeholder = st.empty()
    
    start_time = time.time()
    face_detected_time = 0
    last_time = start_time
    last_encoding_capture = 0  # Track when we last captured an encoding
    encoding_capture_interval = 1.0  # Capture face encoding every 1 second
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = max(0, duration - elapsed)
            progress = (elapsed / duration) * 100
            
            if elapsed >= duration or detector.is_liveness_passed():
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, movement, challenge_done, face_detected = detector.process_frame(frame)
            
            time_delta = current_time - last_time
            if face_detected:
                face_detected_time += time_delta
                
                # Capture face encoding periodically (every 1 second when face is detected)
                if current_time - last_encoding_capture >= encoding_capture_interval:
                    face_encoder.add_encoding_sample(frame)
                    last_encoding_capture = current_time
                    
            last_time = current_time
            
            # Draw header with challenge info
            h, w = processed_frame.shape[:2]
            cv2.rectangle(processed_frame, (0, 0), (w, 90), (15, 12, 41), -1)
            border_color = (56, 239, 125) if face_detected else (235, 51, 73)
            cv2.rectangle(processed_frame, (0, 0), (w, 90), border_color, 4)
            
            # Show time and challenges
            cv2.putText(processed_frame, f"TIME: {int(remaining)}s", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(processed_frame, f"CHALLENGES: {detector.challenges_completed}/3", (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 200), 2)
            
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
            
            # Update stats
            timer_placeholder.markdown(
                f'<div class="stat-box"><div class="stat-value">{int(remaining)}</div><div class="stat-label">⏱️ Seconds</div></div>', 
                unsafe_allow_html=True
            )
            
            moves_placeholder.markdown(
                f'<div class="stat-box"><div class="stat-value">{detector.moment_count}</div><div class="stat-label">🔄 Moves</div></div>', 
                unsafe_allow_html=True
            )
            
            blinks_placeholder.markdown(
                f'<div class="stat-box"><div class="stat-value">{detector.blink_count}</div><div class="stat-label">👁️ Blinks</div></div>', 
                unsafe_allow_html=True
            )
            
            if face_detected:
                face_status_placeholder.markdown(
                    '<div class="stat-box" style="border-color: #38ef7d;"><div class="stat-value">✅</div><div class="stat-label">Face OK</div></div>',
                    unsafe_allow_html=True
                )
            else:
                face_status_placeholder.markdown(
                    '<div class="stat-box" style="border-color: #eb3349;"><div class="stat-value">❌</div><div class="stat-label">No Face</div></div>',
                    unsafe_allow_html=True
                )
            
            progress_placeholder.progress(int(progress))
            time.sleep(0.03)
                
    finally:
        cap.release()
        detector.cleanup()
        frame_placeholder.empty()
    
    liveness_score = detector.get_liveness_score()
    
    # Get the averaged face encoding from all samples
    face_encoding = face_encoder.get_average_encoding()
    
    st.session_state.blink_count = detector.moment_count
    st.session_state.liveness_score = liveness_score
    st.session_state.face_detected = face_detected_time > 0
    st.session_state.verification_complete = True
    st.session_state.captured_face_encoding = face_encoding  # Store for duplicate check
    
    return detector.blink_count, liveness_score, face_detected_time > 0, face_encoding

def render_risk_decision(risk_result):
    """Render the risk decision UI based on RiskResult."""
    tier_colors = {
        RiskTier.LOW: ("#38ef7d", "#11998e", "✅"),
        RiskTier.MEDIUM: ("#f093fb", "#f5576c", "⚠️"),
        RiskTier.HIGH: ("#eb3349", "#f45c43", "🚫")
    }
    
    color1, color2, icon = tier_colors.get(risk_result.tier, ("#667eea", "#764ba2", "🔍"))
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color1} 0%, {color2} 100%);
                padding: 30px; border-radius: 20px; color: white; margin: 20px 0;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);">
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <span style="font-size: 3rem;">{icon}</span>
            <div>
                <h2 style="margin: 0; font-family: 'Orbitron', sans-serif;">RISK SCORE: {risk_result.score}</h2>
                <p style="margin: 5px 0 0 0; font-size: 1.2rem; opacity: 0.9;">Tier: {risk_result.tier.value}</p>
            </div>
        </div>
        <p style="font-size: 1.1rem; margin: 15px 0;"><b>Decision:</b> {risk_result.decision}</p>
        <p style="font-size: 1rem; opacity: 0.95;">{risk_result.explanation}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show reason codes if enabled in config
    if CONFIG.get("ui", {}).get("show_reason_codes", True) and risk_result.reason_codes:
        with st.expander("🔍 View Risk Factors (Reason Codes)"):
            for code in risk_result.reason_codes:
                st.markdown(f"• `{code}`")

def render_govid_verification():
    """Render the Government ID (Mock) verification UI."""
    st.markdown("""
    <div class="info-box">
        <h3>🆔 Government ID Verification (Mock)</h3>
        <p><b>⚠️ Demo Mode:</b> This is a simulated verification for educational purposes only.</p>
        <p>Enter a 12-digit number to simulate government ID verification.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        govid_number = st.text_input(
            "Government ID Number",
            placeholder="Enter 12-digit ID (e.g., 123456789012)",
            help="Enter any 12-digit number for demo"
        )
    
    with col2:
        if st.button("🔐 Verify ID", use_container_width=True):
            if govid_number:
                verifier = get_verifier()
                
                with st.spinner("Verifying ID... (Mock)"):
                    # Get username from session state if available
                    username = st.session_state.user_inputs.get('username', '')
                    result = verifier.quick_verify(govid_number, holder_name=username)
                
                if result.verified:
                    st.session_state.govid_verified = True
                    st.session_state.govid_result = result
                    st.success(f"✅ ID Verified! Holder: {result.holder_name}")
                    st.rerun()
                else:
                    st.error(f"❌ {result.error_message}")
            else:
                st.warning("Please enter an ID number")
    
    return st.session_state.govid_verified

def render_user_type_selection():
    """Render the user type selection (New vs Existing)."""
    st.markdown("""
    <div class="content-box" style="text-align: center;">
        <h2 style="margin-bottom: 30px;">👋 Welcome to BioVerify</h2>
        <p style="font-size: 1.2rem; margin-bottom: 40px;">
            Select your verification type to get started
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="step-card" style="min-height: 280px;">
            <h2 style="position: relative; z-index: 1; font-size: 4rem; margin-bottom: 20px;">🆕</h2>
            <h3 style="position: relative; z-index: 1; margin-bottom: 15px;">New User Registration</h3>
            <p style="position: relative; z-index: 1; opacity: 0.9;">
                First time here? Register your account with risk-based verification.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🆕 I'm a New User", use_container_width=True, key="new_user_btn"):
            st.session_state.user_type = "new"
            st.session_state.current_step = 1
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class="step-card" style="min-height: 280px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
            <h2 style="position: relative; z-index: 1; font-size: 4rem; margin-bottom: 20px;">🔑</h2>
            <h3 style="position: relative; z-index: 1; margin-bottom: 15px;">Existing Account Login</h3>
            <p style="position: relative; z-index: 1; opacity: 0.9;">
                Already registered? Login with behavior-based verification.
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔑 I'm an Existing User", use_container_width=True, key="existing_user_btn"):
            st.session_state.user_type = "existing"
            st.session_state.current_step = 1
            st.rerun()

def render_existing_user_lookup():
    """Render the existing user lookup form."""
    st.markdown("""
    <div class="content-box">
        <h2>🔑 Existing User Login</h2>
        <p>Enter your Aadhaar or username to access your account</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Back button to switch user type
    col_back, col_spacer = st.columns([1, 3])
    with col_back:
        if st.button("⬅️ Back to Start", key="back_to_start_existing"):
            st.session_state.user_type = None
            st.session_state.current_step = 0
            st.session_state.risk_result = None
            st.session_state.existing_user_record = None
            st.session_state.existing_lookup_done = False
            st.rerun()
    
    # Login method selector
    login_method = st.radio(
        "🔐 Login Method",
        options=["Aadhaar Number", "Username"],
        horizontal=True,
        key="login_method_radio"
    )
    
    if login_method == "Aadhaar Number":
        aadhaar_input = st.text_input(
            "🆔 Enter your Aadhaar (12 digits)",
            placeholder="Enter the same Aadhaar you registered with",
            key="existing_aadhaar_input"
        )
        lookup_value = aadhaar_input
        lookup_type = "aadhaar"
    else:
        username_input = st.text_input(
            "👤 Enter your username",
            placeholder="e.g., john_doe123",
            key="existing_username_input"
        )
        lookup_value = username_input
        lookup_type = "username"
    
    col_lookup, col_new = st.columns(2)
    
    with col_lookup:
        lookup_clicked = st.button("🔍 Find My Account", use_container_width=True, key="lookup_account_btn")
    
    with col_new:
        if st.button("🆕 Register New Account", use_container_width=True, key="switch_to_new_btn"):
            st.session_state.user_type = "new"
            st.session_state.current_step = 1
            st.session_state.risk_result = None
            st.session_state.existing_lookup_done = False
            st.rerun()
    
    if lookup_clicked:
        if lookup_value:
            db = get_db()
            
            # Look up by Aadhaar or username
            if lookup_type == "aadhaar":
                # Validate Aadhaar format using validation function
                is_valid, error_msg, clean_aadhaar = validate_aadhaar(lookup_value)
                if not is_valid:
                    st.error(f"❌ {error_msg}")
                    return
                user = db.get_user_by_aadhaar(lookup_value)
            else:
                user = db.get_user(lookup_value)
            
            if user:
                st.session_state.existing_user_record = user
                st.session_state.user_inputs['username'] = user.username
                st.session_state.existing_lookup_done = True
                st.session_state.existing_user_found = True
                st.session_state.existing_risk_done = False  # Reset - wait for form submission
                
                st.rerun()  # Rerun to show social media form
            else:
                st.session_state.existing_lookup_done = True
                st.session_state.existing_user_found = False
                st.rerun()
        else:
            st.warning(f"Please enter your {login_method.lower()}")
    
    # Show results OUTSIDE button handler
    if st.session_state.get('existing_lookup_done', False):
        if st.session_state.get('existing_user_found', False):
            user = st.session_state.existing_user_record
            
            st.success(f"✅ Account found! Username: **{user.username}** | Status: **{user.verification_status.upper()}**")
            
            # Check if risk analysis already done
            if st.session_state.get('existing_risk_done', False):
                risk_result = st.session_state.risk_result
                
                # Show risk result
                render_risk_decision(risk_result)
                
                # Determine next step based on risk
                if risk_result.tier == RiskTier.HIGH and not risk_result.allow_login:
                    st.markdown("""
                    <div class="error-box">
                        <h3>🚫 Account Blocked</h3>
                        <p>Your account has been blocked due to high risk indicators. Please contact support.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("📊 View Report", use_container_width=True, key="view_blocked_report_btn"):
                        st.session_state.current_step = 3
                        st.session_state.existing_lookup_done = False
                        st.session_state.existing_risk_done = False
                        st.session_state.liveness_passed = False
                        st.session_state.liveness_score = 0
                        st.rerun()
                elif risk_result.tier == RiskTier.LOW:
                    st.markdown("""
                    <div class="success-box">
                        <h3>✅ Low Risk - Login Allowed!</h3>
                        <p>Your account looks good. No additional verification needed.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("✅ Continue to Dashboard", use_container_width=True, key="continue_dashboard_btn"):
                        st.session_state.liveness_passed = True
                        st.session_state.liveness_score = 100
                        st.session_state.current_step = 3
                        st.session_state.existing_lookup_done = False
                        st.session_state.existing_risk_done = False
                        st.rerun()
                else:
                    # MEDIUM risk - needs verification
                    st.info(f"📋 Required: {'Aadhaar + ' if risk_result.requires_govid else ''}{'Liveness Check' if risk_result.requires_liveness else ''}")
                    if st.button("▶️ Proceed to Verification", use_container_width=True, key="proceed_verify_existing_btn"):
                        st.session_state.current_step = 2
                        st.session_state.existing_lookup_done = False
                        st.session_state.existing_risk_done = False
                        st.rerun()
            else:
                # Show social media form for risk analysis
                st.markdown("""
                <div class="info-box">
                    <h3>📊 Account Risk Analysis</h3>
                    <p>Enter your current social media account details for risk assessment.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Platform selector
                platform = st.radio(
                    "🌐 Platform",
                    options=["Instagram", "Twitter/X"],
                    horizontal=True,
                    key="existing_platform_radio"
                )
                
                with st.form("existing_account_analysis_form"):
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        followers = st.number_input("👥 Followers Count", min_value=0, value=100, step=1, key="ex_followers")
                        following = st.number_input("➕ Following Count", min_value=0, value=150, step=1, key="ex_following")
                        posts = st.number_input("📝 Posts Count", min_value=0, value=20, step=1, key="ex_posts")
                        profile_pic = st.selectbox("🖼️ Has Profile Picture?", options=[1, 0], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌", key="ex_pic")
                    
                    with c2:
                        bio_length = st.number_input("📝 Bio Length (chars)", min_value=0, value=50, step=1, key="ex_bio")
                        external_url = st.selectbox("🔗 Has External URL?", options=[0, 1], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌", key="ex_url")
                        private = st.selectbox("🔒 Private Account?", options=[0, 1], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌", key="ex_private")
                    
                    submit_analysis = st.form_submit_button("🔍 Analyze Risk", use_container_width=True)
                    
                    if submit_analysis:
                        # Store for processing outside form
                        st.session_state.pending_existing_analysis = {
                            'followers': followers,
                            'following': following,
                            'posts': posts,
                            'profile_pic': profile_pic,
                            'bio_length': bio_length,
                            'external_url': external_url,
                            'private': private,
                            'platform': platform,
                            'username': user.username
                        }
                        st.session_state.existing_analysis_requested = True
                        st.rerun()
                
                # Process analysis OUTSIDE form
                if st.session_state.get('existing_analysis_requested', False):
                    data = st.session_state.pending_existing_analysis
                    
                    # Calculate features
                    username = data['username']
                    username_digits = sum(c.isdigit() for c in username)
                    username_ratio = username_digits / len(username) if len(username) > 0 else 0
                    
                    account_data = {
                        'followers': data['followers'],
                        'following': data['following'],
                        'posts': data['posts'],
                        'profile_pic': data['profile_pic'],
                        'bio_length': data['bio_length'],
                        'external_url': data['external_url'],
                        'private': data['private'],
                        'username_ratio': username_ratio,
                        'fullname_words': 2,
                        'fullname_ratio': 0,
                        'name_equals_username': 0,
                        'username': username,
                        'fullname': user.username,
                        'platform': data['platform']
                    }
                    st.session_state.user_inputs = account_data
                    
                    # Run risk analysis
                    analyzer = get_analyzer(
                        CONFIG.get("risk_thresholds", {}).get("low", 30),
                        CONFIG.get("risk_thresholds", {}).get("high", 70)
                    )
                    
                    risk_result = analyzer.analyze_new_user(
                        account_data,
                        device_info=st.session_state.device_fingerprint,
                        ip_risk=0,
                        email_domain=""
                    )
                    
                    st.session_state.risk_result = risk_result
                    st.session_state.account_fake_prob = risk_result.score
                    st.session_state.existing_risk_done = True
                    st.session_state.existing_analysis_requested = False
                    
                    # Update database with risk
                    db = get_db()
                    if risk_result.tier == RiskTier.HIGH:
                        db.update_verification_status(
                            username, "blocked",
                            risk_score=risk_result.score,
                            risk_tier=risk_result.tier.value,
                            reason_codes=risk_result.reason_codes
                        )
                    
                    st.rerun()
        else:
            st.warning("⚠️ Account not found. Would you like to register as a new user?")
            if st.button("🆕 Register as New User", use_container_width=True, key="register_new_user_btn"):
                st.session_state.user_type = "new"
                st.session_state.current_step = 1
                st.session_state.existing_lookup_done = False
                st.rerun()


def main():
    # Premium Animated Header with LOGO
    st.markdown("""
    <div style="text-align: center; padding: 50px 0 30px 0;">
        <div style="font-size: 5rem; margin-bottom: 20px;">👤</div>
        <h1 class="main-header">BIOVERIFY</h1>
        <p class="sub-header">Next-Gen AI-Powered Multi-Modal Fake Account Detection</p>
        <div style="margin: 25px 0;">
            <span class="badge badge-success">🤖 Machine Learning</span>
            <span class="badge badge-warning">🎭 Biometric Verification</span>
            <span class="badge badge-danger">⚡ Real-time Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, feature_names, accuracy, model_loaded = load_model()
    
    # Model status
    if not model_loaded:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Demo Mode Active</h3>
            <p>ML model not loaded. Using intelligent rule-based predictions.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ ML Model Loaded Successfully</h3>
            <p><b>Model Accuracy:</b> {accuracy*100:.2f}% | <b>Status:</b> Operational</p>
        </div>
        """, unsafe_allow_html=True)
    
    # STEP 0: User Type Selection (show before progress steps)
    if st.session_state.current_step == 0:
        render_user_type_selection()
        return  # Don't show progress steps yet
    
    # Show user type badge
    user_type_label = "🆕 New User" if st.session_state.user_type == "new" else "🔑 Existing User"
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 10px;">
        <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 8px 20px; border-radius: 20px; color: white; 
                    font-weight: 600; font-size: 0.9rem;">{user_type_label}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress Steps
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        step_class = "step-card-completed" if st.session_state.current_step > 1 else "step-card-active" if st.session_state.current_step == 1 else "step-card-inactive"
        icon = "✅" if st.session_state.current_step > 1 else "🧠" if st.session_state.current_step == 1 else "1"
        status = "Completed" if st.session_state.current_step > 1 else "Active" if st.session_state.current_step == 1 else "Pending"
        st.markdown(f"""
        <div class="step-card {step_class}">
            <h2 style="margin: 0; position: relative; z-index: 1; font-size: 2.5rem;">{icon}</h2>
            <p style="margin: 15px 0 0 0; font-size: 1.3rem; position: relative; z-index: 1; font-weight: 700;">STEP 1</p>
            <p style="margin: 5px 0 0 0; font-size: 1rem; position: relative; z-index: 1;">Risk Evaluation</p>
            <p style="margin: 10px 0 0 0; font-size: 0.9rem; opacity: 0.9; position: relative; z-index: 1;">{status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        step_class = "step-card-completed" if st.session_state.current_step > 2 else "step-card-active" if st.session_state.current_step == 2 else "step-card-inactive"
        icon = "✅" if st.session_state.current_step > 2 else "🎥" if st.session_state.current_step == 2 else "2"
        status = "Completed" if st.session_state.current_step > 2 else "Active" if st.session_state.current_step == 2 else "Pending"
        
        # Dynamic label based on requirements
        step2_label = "Verification"
        if st.session_state.risk_result:
            if st.session_state.risk_result.requires_govid and st.session_state.risk_result.requires_liveness:
                step2_label = "ID + Liveness"
            elif st.session_state.risk_result.requires_liveness:
                step2_label = "Liveness Check"
            elif st.session_state.risk_result.requires_govid:
                step2_label = "ID Verification"
        
        st.markdown(f"""
        <div class="step-card {step_class}">
            <h2 style="margin: 0; position: relative; z-index: 1; font-size: 2.5rem;">{icon}</h2>
            <p style="margin: 15px 0 0 0; font-size: 1.3rem; position: relative; z-index: 1; font-weight: 700;">STEP 2</p>
            <p style="margin: 5px 0 0 0; font-size: 1rem; position: relative; z-index: 1;">{step2_label}</p>
            <p style="margin: 10px 0 0 0; font-size: 0.9rem; opacity: 0.9; position: relative; z-index: 1;">{status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        step_class = "step-card-completed" if st.session_state.current_step > 3 else "step-card-active" if st.session_state.current_step == 3 else "step-card-inactive"
        icon = "✅" if st.session_state.current_step > 3 else "📊" if st.session_state.current_step == 3 else "3"
        status = "Completed" if st.session_state.current_step > 3 else "Active" if st.session_state.current_step == 3 else "Pending"
        st.markdown(f"""
        <div class="step-card {step_class}">
            <h2 style="margin: 0; position: relative; z-index: 1; font-size: 2.5rem;">{icon}</h2>
            <p style="margin: 15px 0 0 0; font-size: 1.3rem; position: relative; z-index: 1; font-weight: 700;">STEP 3</p>
            <p style="margin: 5px 0 0 0; font-size: 1rem; position: relative; z-index: 1;">Final Results</p>
            <p style="margin: 10px 0 0 0; font-size: 0.9rem; opacity: 0.9; position: relative; z-index: 1;">{status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # STEP 1: Risk Evaluation (Different for new vs existing users)
    if st.session_state.current_step == 1:
        
        # EXISTING USER FLOW
        if st.session_state.user_type == "existing":
            render_existing_user_lookup()
        
        # NEW USER FLOW - Aadhaar Verification First
        else:
            st.markdown("""
            <div class="content-box">
                <h2>🆔 Step 1: Identity Verification</h2>
                <p>Verify your Aadhaar to create a new account</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Back button to switch user type
            col_back, col_spacer = st.columns([1, 3])
            with col_back:
                if st.button("⬅️ Back to Start", key="back_to_start_new"):
                    st.session_state.user_type = None
                    st.session_state.current_step = 0
                    st.session_state.govid_verified = False
                    st.session_state.aadhaar_checked = False
                    st.rerun()
            
            # Info box
            st.markdown("""
            <div class="info-box">
                <h3>📋 New User Registration Process:</h3>
                <ol style="font-size: 1.1rem; line-height: 1.8;">
                    <li>🆔 <b>Verify Aadhaar</b> - Confirm your identity (mock verification)</li>
                    <li>🎥 <b>Liveness Check</b> - Prove you are a real person</li>
                    <li>✅ <b>Create Account</b> - Choose your username and complete registration</li>
                </ol>
                <p style="margin-top: 15px; opacity: 0.9;">
                    ⚠️ <b>Note:</b> Each Aadhaar can only be registered once.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # If Aadhaar already verified, show success and proceed
            if st.session_state.get('govid_verified', False) and st.session_state.get('govid_result'):
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ Aadhaar Verified Successfully!</h3>
                    <p><b>Holder:</b> {st.session_state.govid_result.holder_name}</p>
                    <p><b>ID:</b> {st.session_state.govid_result.masked_id}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("▶️ Proceed to Liveness Check", use_container_width=True, key="proceed_liveness_new"):
                    st.session_state.current_step = 2
                    st.rerun()
            else:
                # Aadhaar verification form
                st.markdown("### 🆔 Enter Your Details")
                
                with st.form("new_user_aadhaar_form"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        aadhaar_number = st.text_input(
                            "Aadhaar Number (12 digits)",
                            placeholder="Enter your 12-digit Aadhaar",
                            help="Your Aadhaar will be hashed for privacy - we never store the actual number"
                        )
                    
                    with col2:
                        desired_username = st.text_input(
                            "Choose a Username",
                            placeholder="e.g., john_doe",
                            help="This will be your unique login identifier"
                        )
                    
                    submit_aadhaar = st.form_submit_button("🔐 Verify Aadhaar", use_container_width=True)
                    
                    if submit_aadhaar:
                        # Validate Aadhaar format using new validation function
                        is_valid, error_msg, clean_aadhaar = validate_aadhaar(aadhaar_number or "")
                        
                        if not is_valid:
                            st.error(f"❌ {error_msg}")
                        elif not desired_username or len(desired_username) < 3:
                            st.error("❌ Username must be at least 3 characters")
                        else:
                            # Store for processing outside form
                            st.session_state.pending_aadhaar = aadhaar_number
                            st.session_state.pending_username = desired_username
                            st.session_state.aadhaar_check_requested = True
                            st.rerun()
                
                # Process Aadhaar check OUTSIDE form
                if st.session_state.get('aadhaar_check_requested', False):
                    aadhaar_number = st.session_state.pending_aadhaar
                    desired_username = st.session_state.pending_username
                    
                    # Validate again for safety
                    is_valid, error_msg, clean_aadhaar = validate_aadhaar(aadhaar_number)
                    if not is_valid:
                        st.session_state.aadhaar_check_requested = False
                        st.error(f"❌ {error_msg}")
                    else:
                        db = get_db()
                        
                        with st.spinner("🔍 Checking Aadhaar status..."):
                            # Check if Aadhaar already registered (now returns tuple)
                            exists, existing_username = db.aadhaar_exists(aadhaar_number)
                            
                            if exists:
                                st.session_state.aadhaar_check_requested = False
                                st.error("🚫 **This Aadhaar is already registered!**")
                                st.warning(f"This Aadhaar is linked to account: **{existing_username}**")
                                st.info("📌 **You cannot register a new account until you delete your previous account.**")
                                
                                st.markdown("---")
                                st.markdown("### Choose an option:")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button("🔑 Login to Existing Account", key="go_existing_btn", use_container_width=True):
                                        st.session_state.user_type = "existing"
                                        st.session_state.current_step = 1
                                        st.rerun()
                                
                                with col2:
                                    # Store the info for delete confirmation
                                    if st.button("🗑️ Delete Previous Account", key="delete_prev_account_btn", use_container_width=True):
                                        st.session_state.show_delete_confirmation = True
                                        st.session_state.delete_aadhaar = aadhaar_number
                                        st.session_state.delete_username = existing_username
                                        st.rerun()
                                
                                # Handle delete confirmation
                                if st.session_state.get('show_delete_confirmation', False):
                                    st.markdown("---")
                                    st.markdown(f"""
                                    <div class="danger-box">
                                        <h3>⚠️ Delete Account Confirmation</h3>
                                        <p>You are about to <b>permanently delete</b> the account: <b>{st.session_state.delete_username}</b></p>
                                        <p>This action cannot be undone. All verification history will be lost.</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    confirm_col1, confirm_col2 = st.columns(2)
                                    
                                    with confirm_col1:
                                        if st.button("❌ Cancel", key="cancel_delete_btn", use_container_width=True):
                                            st.session_state.show_delete_confirmation = False
                                            st.rerun()
                                    
                                    with confirm_col2:
                                        if st.button("🗑️ Confirm Delete", key="confirm_delete_btn", type="primary", use_container_width=True):
                                            success, deleted_user = db.delete_user_by_aadhaar(st.session_state.delete_aadhaar)
                                            if success:
                                                st.session_state.show_delete_confirmation = False
                                                st.success(f"✅ Account '{deleted_user}' has been deleted. You can now register with a new account!")
                                                st.session_state.aadhaar_check_requested = True  # Trigger re-check
                                                time.sleep(2)
                                                st.rerun()
                                            else:
                                                st.error("❌ Failed to delete account. Please try again.")
                                                st.session_state.show_delete_confirmation = False
                            
                            # Check if username taken
                            elif db.user_exists(desired_username):
                                st.session_state.aadhaar_check_requested = False
                                st.error(f"❌ Username '{desired_username}' is already taken. Please choose a different one.")
                            
                            else:
                                # Verify Aadhaar using mock verifier
                                verifier = get_verifier()
                                result = verifier.quick_verify(aadhaar_number, holder_name=desired_username)
                                
                                if result.verified:
                                    st.session_state.govid_verified = True
                                    st.session_state.govid_result = result
                                    st.session_state.pending_aadhaar_verified = aadhaar_number
                                    st.session_state.user_inputs = {
                                        'username': desired_username,
                                        'aadhaar': aadhaar_number,
                                        'fullname': result.holder_name
                                    }
                                    st.session_state.aadhaar_check_requested = False
                                    st.success(f"✅ Aadhaar verified! Welcome, {result.holder_name}")
                                    st.rerun()
                                else:
                                    st.session_state.aadhaar_check_requested = False
                                    st.error(f"❌ Aadhaar verification failed: {result.error_message}")
    
    # STEP 2: Biometric Verification
    elif st.session_state.current_step == 2:
        # Determine what verification is needed
        risk_result = st.session_state.risk_result
        needs_govid = risk_result.requires_govid if risk_result else False
        needs_liveness = risk_result.requires_liveness if risk_result else True
        
        st.markdown("""
        <div class="content-box">
            <h2>🔐 Step 2: Identity Verification</h2>
            <p>Complete the required verification steps based on your risk profile</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show risk summary
        if risk_result:
            tier_badge = {
                RiskTier.LOW: ("✅ LOW RISK", "#38ef7d"),
                RiskTier.MEDIUM: ("⚠️ MEDIUM RISK", "#f093fb"),
                RiskTier.HIGH: ("🚫 HIGH RISK", "#eb3349")
            }
            badge_text, badge_color = tier_badge.get(risk_result.tier, ("🔍 UNKNOWN", "#667eea"))
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <span style="background: {badge_color}; color: white; padding: 10px 25px; 
                            border-radius: 25px; font-weight: 700; font-size: 1rem;">
                    {badge_text} • Score: {risk_result.score}
                </span>
            </div>
            """, unsafe_allow_html=True)
        
        # GOVERNMENT ID VERIFICATION (if required)
        if needs_govid and not st.session_state.govid_verified:
            st.markdown("""
            <div class="warning-box">
                <h3>🆔 Government ID Required</h3>
                <p>Based on your risk profile, Government ID verification is required before proceeding.</p>
            </div>
            """, unsafe_allow_html=True)
            
            render_govid_verification()
            
            st.markdown("---")
            st.info("⏳ Complete Government ID verification above to proceed to liveness check.")
            return  # Don't show liveness until govid is done
        
        # Show govid success if completed
        if st.session_state.govid_verified and st.session_state.govid_result:
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ Government ID Verified</h3>
                <p><b>Holder:</b> {st.session_state.govid_result.holder_name}</p>
                <p><b>ID:</b> {st.session_state.govid_result.masked_id}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # LIVENESS VERIFICATION (if required)
        if needs_liveness:
            st.markdown("---")
            st.markdown("""
            <div class="info-box">
                <h3>🎥 Liveness Verification</h3>
                <p>Complete the camera-based liveness check to verify you are a real person.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Check if face duplicate was detected
            if st.session_state.get('face_duplicate_found', False):
                matching_username = st.session_state.get('face_duplicate_username', 'Unknown')
                similarity = st.session_state.get('face_duplicate_similarity', 0) * 100
                
                st.markdown(f"""
                <div class="danger-box">
                    <h3>🚫 Duplicate Face Detected!</h3>
                    <p><b>This face is already registered</b> to account: <b>{matching_username}</b></p>
                    <p>Match confidence: <b>{similarity:.1f}%</b></p>
                    <p>You cannot create a new account with the same face. Each person can only have one account.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.info("📌 **You cannot register a new account until you delete your previous account.**")
                
                st.markdown("---")
                st.markdown("### Choose an option:")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔑 Login to Existing Account", key="face_dup_login_btn", use_container_width=True):
                        st.session_state.face_duplicate_found = False
                        st.session_state.user_type = "existing"
                        st.session_state.current_step = 1
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Delete Previous Account", key="face_dup_delete_btn", use_container_width=True):
                        st.session_state.show_face_delete_confirmation = True
                        st.rerun()
                
                # Handle delete confirmation
                if st.session_state.get('show_face_delete_confirmation', False):
                    st.markdown("---")
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3>⚠️ Delete Account Confirmation</h3>
                        <p>You are about to <b>permanently delete</b> the account: <b>{matching_username}</b></p>
                        <p>This action cannot be undone. All verification history will be lost.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    confirm_col1, confirm_col2 = st.columns(2)
                    
                    with confirm_col1:
                        if st.button("❌ Cancel", key="face_cancel_delete_btn", use_container_width=True):
                            st.session_state.show_face_delete_confirmation = False
                            st.rerun()
                    
                    with confirm_col2:
                        if st.button("🗑️ Confirm Delete", key="face_confirm_delete_btn", type="primary", use_container_width=True):
                            db = get_db()
                            success = db.delete_user(matching_username)
                            if success:
                                st.session_state.show_face_delete_confirmation = False
                                st.session_state.face_duplicate_found = False
                                st.success(f"✅ Account '{matching_username}' has been deleted. Please complete liveness check again!")
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete account. Please try again.")
                                st.session_state.show_face_delete_confirmation = False
                
                return  # Don't show liveness test button when duplicate found
            
            # Check if face mismatch was detected (for existing users)
            if st.session_state.get('face_mismatch_found', False):
                similarity = st.session_state.get('face_mismatch_similarity', 0) * 100
                
                st.markdown(f"""
                <div class="danger-box">
                    <h3>🚫 Face Mismatch Detected!</h3>
                    <p>The face detected <b>does not match</b> the registered face for this account.</p>
                    <p>Match similarity: <b>{similarity:.1f}%</b> (required: 85%)</p>
                    <p>This could indicate someone else is trying to access this account.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.warning("⚠️ **Security Alert**: If this is really your account, please ensure good lighting and face the camera directly.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔄 Try Again", key="face_mismatch_retry_btn", use_container_width=True):
                        st.session_state.face_mismatch_found = False
                        st.rerun()
                
                with col2:
                    if st.button("📧 Contact Support", key="face_mismatch_support_btn", use_container_width=True):
                        st.info("📞 Please contact support at support@bioverify.com for assistance with account recovery.")
                
                return  # Don't show liveness test button when face mismatch found
            
            # Show warning for existing users with no stored face
            user_record = st.session_state.get('existing_user_record')
            if st.session_state.user_type == "existing" and user_record:
                has_stored_face = hasattr(user_record, 'face_encoding') and user_record.face_encoding
                if not has_stored_face:
                    st.markdown("""
                    <div class="warning-box" style="border-color: #ff6b6b;">
                        <h3>⚠️ No Biometric Record Found</h3>
                        <p>This account has <b>no stored face encoding</b>. We cannot verify your identity against previous records.</p>
                        <p>Your face will be enrolled for future verification. This event has been logged for security.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show warning if account had high risk in Step 1
            if st.session_state.get('high_account_risk_warning', False) or (risk_result and risk_result.tier == RiskTier.MEDIUM):
                st.markdown(f"""
                <div class="warning-box">
                    <h3>⚠️ Enhanced Verification Required</h3>
                    <p><b>Risk Score:</b> {st.session_state.account_fake_prob:.1f}</p>
                    <p>Your risk profile requires careful liveness verification. Please follow the instructions carefully.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Initialize attempt counter
            if 'verification_attempts' not in st.session_state:
                st.session_state.verification_attempts = 0
            
            # Show retry warning if previous attempt failed
            if st.session_state.verification_attempts == 1:
                st.markdown("""
                <div class="warning-box">
                    <h3>⚠️ Please Move Your Head!</h3>
                    <p>No sufficient head movements detected in last attempt. This is your <b>final attempt</b>.</p>
                    <p>Move your head (nod, turn, or tilt) at least 3 times clearly during the verification.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Instructions
            st.markdown("""
            <div class="info-box">
                <h3>📋 Verification Instructions:</h3>
                <ul style="font-size: 1.1rem; line-height: 1.8;">
                    <li>🎯 Position your face clearly in the camera frame</li>
                    <li>👤 Look directly at the camera</li>
                    <li>🔄 Move your head naturally (nod, turn, or tilt 3+ times)</li>
                    <li>⏱️ Maintain visibility for full 20 seconds</li>
                    <li>💡 Ensure good lighting conditions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        col_center = st.columns([1, 2, 1])[1]
        with col_center:
            button_text = "🚀 START VERIFICATION" if st.session_state.verification_attempts == 0 else "🔄 TRY AGAIN (Final Attempt)"
            if st.button(button_text, use_container_width=True):
                st.session_state.verification_attempts += 1
                blink_count, liveness_score, face_detected, face_encoding = run_liveness_test_with_timer()
                
                # Get username for database update
                username = st.session_state.user_inputs.get('username', '')
                db = get_db()
                risk_result = st.session_state.risk_result
                
                # Case 1: No face detected at all - mark as FAKE
                if not face_detected:
                    st.session_state.liveness_passed = False
                    st.session_state.liveness_score = 0
                    st.session_state.blink_count = 0
                    st.session_state.face_detected = False
                    st.session_state.no_face_detected = True
                    st.session_state.current_step = 3
                    st.session_state.verification_attempts = 0
                    
                    # Update database
                    if username:
                        db.update_verification_status(
                            username, "flagged",
                            liveness_passed=False,
                            govid_passed=st.session_state.govid_verified,
                            risk_score=risk_result.score if risk_result else 0,
                            risk_tier=risk_result.tier.value if risk_result else "UNKNOWN",
                            reason_codes=["NO_FACE_DETECTED"]
                        )
                    st.rerun()
                
                # Case 2: Passed liveness test - Check for face duplicates before proceeding
                elif liveness_score >= 50:
                    # Check for face duplicates (only for new users)
                    if st.session_state.user_type == "new" and face_encoding is not None:
                        # Get all existing face encodings
                        all_face_data = db.get_all_face_encodings()
                        stored_encodings = []
                        for stored_username, stored_encoding_json in all_face_data:
                            stored_encoding = FaceEncoder.deserialize_encoding(stored_encoding_json)
                            if stored_encoding is not None:
                                stored_encodings.append((stored_username, stored_encoding))
                        
                        # Check for matching face
                        match_result = find_matching_face(face_encoding, stored_encodings)
                        
                        if match_result:
                            matching_username, similarity = match_result
                            # Duplicate face found! Block registration
                            st.session_state.face_duplicate_found = True
                            st.session_state.face_duplicate_username = matching_username
                            st.session_state.face_duplicate_similarity = similarity
                            st.session_state.verification_attempts = 0
                            st.rerun()
                    
                    # No duplicate found - proceed with success
                    st.session_state.liveness_passed = True
                    st.session_state.no_face_detected = False
                    st.session_state.face_duplicate_found = False
                    st.session_state.current_step = 3
                    st.session_state.verification_attempts = 0
                    
                    # Serialize face encoding for storage
                    face_encoding_json = FaceEncoder.serialize_encoding(face_encoding) if face_encoding is not None else ""
                    
                    # For NEW USERS - Create account with Aadhaar and face encoding
                    if st.session_state.user_type == "new" and st.session_state.get('pending_aadhaar_verified'):
                        aadhaar = st.session_state.pending_aadhaar_verified
                        user = db.create_user_with_aadhaar(
                            username=username,
                            aadhaar_number=aadhaar,
                            device_fingerprint=st.session_state.device_fingerprint
                        )
                        if user:
                            # Store face encoding
                            db.update_face_encoding(username, face_encoding_json)
                            db.update_verification_status(
                                username, "verified",
                                liveness_passed=True,
                                govid_passed=True,
                                risk_score=0,
                                risk_tier="LOW",
                                reason_codes=[]
                            )
                    # For EXISTING USERS - Verify face matches registered face
                    elif st.session_state.user_type == "existing" and username:
                        # Get the stored user record to check face
                        user_record = st.session_state.get('existing_user_record')
                        
                        has_stored_face = user_record and hasattr(user_record, 'face_encoding') and user_record.face_encoding
                        
                        if has_stored_face:
                            # User has stored face - verify it matches
                            stored_encoding = FaceEncoder.deserialize_encoding(user_record.face_encoding)
                            if stored_encoding is not None and face_encoding is not None:
                                is_same, similarity = FaceEncoder.is_same_person(face_encoding, stored_encoding)
                                if not is_same:
                                    # Face mismatch - block access
                                    st.session_state.face_mismatch_found = True
                                    st.session_state.face_mismatch_similarity = similarity
                                    st.session_state.verification_attempts = 0
                                    st.rerun()
                        else:
                            # User has NO stored face encoding - first biometric enrollment
                            # Flag this as a security event since we can't verify identity
                            st.session_state.first_face_enrollment = True
                        
                        # Face matches or no stored face - proceed with verification
                        st.session_state.face_mismatch_found = False
                        
                        # Update/store face encoding for existing users
                        if face_encoding_json:
                            db.update_face_encoding(username, face_encoding_json)
                        
                        # If this was first enrollment without prior face, flag the account
                        if not has_stored_face:
                            db.update_verification_status(
                                username, "verified",
                                liveness_passed=True,
                                govid_passed=st.session_state.govid_verified,
                                risk_score=risk_result.score if risk_result else 50,  # Higher base risk for no prior face
                                risk_tier=risk_result.tier.value if risk_result else "MEDIUM",
                                reason_codes=["FIRST_FACE_ENROLLMENT"]
                            )
                        else:
                            db.update_verification_status(
                                username, "verified",
                                liveness_passed=True,
                                govid_passed=st.session_state.govid_verified,
                                risk_score=risk_result.score if risk_result else 0,
                                risk_tier=risk_result.tier.value if risk_result else "LOW",
                                reason_codes=[]
                            )
                    st.balloons()
                    st.rerun()
                
                # Case 3: Failed but first attempt - show retry option
                elif st.session_state.verification_attempts == 1:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3>⚠️ Insufficient Head Movement Detected</h3>
                        <p><b>Liveness Score:</b> {liveness_score:.1f}%</p>
                        <p><b>Movements Detected:</b> {blink_count}</p>
                        <p>Please try again and move your head more clearly (nod, turn, or tilt).</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
                
                # Case 4: Failed second attempt - proceed to step 3 with low score
                else:
                    st.session_state.liveness_passed = False
                    st.session_state.no_face_detected = False
                    st.session_state.current_step = 3
                    st.session_state.verification_attempts = 0
                    
                    # Update database - flagged
                    if username:
                        db.update_verification_status(
                            username, "flagged",
                            liveness_passed=False,
                            govid_passed=st.session_state.govid_verified,
                            risk_score=risk_result.score if risk_result else 0,
                            risk_tier=risk_result.tier.value if risk_result else "UNKNOWN",
                            reason_codes=["LIVENESS_FAILED"]
                        )
                    st.rerun()
    
    # STEP 3: Final Results
    elif st.session_state.current_step == 3:
        st.markdown("""
        <div class="content-box">
            <h2>📊 Step 3: Comprehensive Verification Results</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display submitted account details
        username = st.session_state.user_inputs.get('username', 'N/A')
        fullname = st.session_state.user_inputs.get('fullname', 'N/A')
        platform = st.session_state.user_inputs.get('platform', 'Instagram')
        
        # Platform-specific styling
        if platform == "Instagram":
            platform_bg = "linear-gradient(45deg, #f09433, #e6683c, #dc2743)"
            platform_icon = "📸"
        else:
            platform_bg = "linear-gradient(45deg, #1DA1F2, #0d8bd9)"
            platform_icon = "🐦"
        
        st.markdown(f"""
        <div class="info-box">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <h3 style="margin: 0;">📋 Account Details</h3>
                <span style="background: {platform_bg}; color: white; padding: 5px 15px; 
                            border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                    {platform_icon} {platform}
                </span>
            </div>
            <p style="font-size: 1.2rem;"><b>👤 Username:</b> {username}</p>
            <p style="font-size: 1.2rem;"><b>📛 Display Name:</b> {fullname}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top metrics
        met1, met2 = st.columns(2)
        
        with met1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Account Risk Score</div>
                <div class="metric-value">{st.session_state.account_fake_prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_animated_gauge(st.session_state.account_fake_prob, "Fake Account Probability", 50), use_container_width=True)
        
        with met2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Biometric Liveness</div>
                <div class="metric-value">{st.session_state.liveness_score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(create_animated_gauge(st.session_state.liveness_score, "Liveness Verification", 50), use_container_width=True)
        
        # Combined Assessment with CLEAR VERDICT
        
        # Check if account was flagged as fake based on account details alone (VETO from Step 1)
        if st.session_state.get('account_only_verdict', False):
            combined_risk = 100
            verdict_class = "verdict-box-fake"
            verdict_icon = "🚫"
            verdict_text = "FAKE ACCOUNT DETECTED"
            verdict_desc = f"Account metadata analysis shows extremely high risk ({st.session_state.account_fake_prob:.1f}%). The account characteristics strongly indicate a fake, bot, or spam account. Camera verification was skipped as account is already flagged."
        # Check if no face was detected - directly mark as FAKE
        elif st.session_state.get('no_face_detected', False):
            combined_risk = 100
            verdict_class = "verdict-box-fake"
            verdict_icon = "🚫"
            verdict_text = "FAKE ACCOUNT DETECTED"
            verdict_desc = "No face visible during biometric verification. This is a strong indicator of a fraudulent or bot account. Immediate action recommended: Block or escalate for investigation."
        else:
            # Corrected formula: 70% account risk + 30% liveness risk (as per README documentation)
            # Liveness Risk = 100 - Liveness Score
            liveness_risk = 100 - st.session_state.liveness_score
            combined_risk = (st.session_state.account_fake_prob * 0.7) + (liveness_risk * 0.3)
            
            # VETO logic for very low liveness scores
            if st.session_state.liveness_score < 30:
                combined_risk = 100  # No face detected or no blinks - VETO
                verdict_class = "verdict-box-fake"
                verdict_icon = "🚫"
                verdict_text = "CONFIRMED FAKE/BOT"
                verdict_desc = "Liveness score below 30% indicates no real human presence. This is a confirmed bot or fake account attempt."
            elif st.session_state.liveness_score < 50:
                combined_risk = 100  # Failed verification threshold - VETO
                verdict_class = "verdict-box-fake"
                verdict_icon = "🚫"
                verdict_text = "LIKELY FAKE/BOT"
                verdict_desc = "Failed biometric verification threshold. Liveness indicators suggest this is not a genuine human user."
            elif combined_risk < 40:
                verdict_class = "verdict-box-real"
                verdict_icon = "✅"
                verdict_text = "REAL ACCOUNT VERIFIED"
                verdict_desc = "This account demonstrates strong authenticity indicators with successful biometric verification. Safe to proceed."
            elif combined_risk < 70:
                verdict_class = "warning-box"
                verdict_icon = "⚠️"
                verdict_text = "SUSPICIOUS - MANUAL REVIEW REQUIRED"
                verdict_desc = "Suspicious patterns detected. Additional verification and human review recommended before approval."
            else:
                verdict_class = "verdict-box-fake"
                verdict_icon = "🚫"
                verdict_text = "FAKE ACCOUNT DETECTED"
                verdict_desc = "Strong indicators of fraudulent account. Immediate action recommended: Block or escalate for investigation."
        
        st.markdown("### 🎯 Final Security Verdict")
        
        st.markdown(f"""
        <div class="{verdict_class}">
            <div class="verdict-title">{verdict_icon} {verdict_text}</div>
            <p style="font-size: 1.3rem; margin: 20px 0;"><b>Combined Risk Score:</b> {combined_risk:.1f}%</p>
            <p style="font-size: 1.1rem;">{verdict_desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed metrics
        st.markdown("### 📈 Detailed Verification Metrics")
        
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Face Moments</div>
                <div class="metric-value">{st.session_state.blink_count}</div>
                <p style="margin: 5px 0 0 0;">🎭 Facial Activity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with m2:
            status_icon = "✅" if st.session_state.face_detected else "❌"
            status_text = "Detected" if st.session_state.face_detected else "Failed"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Face Status</div>
                <div class="metric-value">{status_icon}</div>
                <p style="margin: 5px 0 0 0;">{status_text}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with m3:
            followers = st.session_state.user_inputs.get('followers', 0)
            following = st.session_state.user_inputs.get('following', 0)
            ratio = following / followers if followers > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Follow Ratio</div>
                <div class="metric-value">{ratio:.2f}</div>
                <p style="margin: 5px 0 0 0;">Following/Followers</p>
            </div>
            """, unsafe_allow_html=True)
        
        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Posts</div>
                <div class="metric-value">{st.session_state.user_inputs.get('posts', 0)}</div>
                <p style="margin: 5px 0 0 0;">📸 Total Content</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Save verification to history
        verification_data = {
            'username': st.session_state.user_inputs.get('username', 'unknown'),
            'account_score': st.session_state.account_fake_prob,
            'liveness_score': st.session_state.liveness_score,
            'verdict': verdict_text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_fake': 'FAKE' in verdict_text
        }
        
        # Store in session for PDF generation
        st.session_state.verification_data = verification_data
        
        # Save to history file
        history_file = 'data/verifications.json'
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except:
                history = {"verifications": [], "total_count": 0, "passed_count": 0, "failed_count": 0}
        else:
            history = {"verifications": [], "total_count": 0, "passed_count": 0, "failed_count": 0}
        
        # Add this verification
        if not hasattr(st.session_state, 'verification_saved') or not st.session_state.verification_saved:
            history["verifications"].append(verification_data)
            history["total_count"] += 1
            if verification_data['is_fake']:
                history["failed_count"] += 1
            else:
                history["passed_count"] += 1
            
            os.makedirs('data', exist_ok=True)
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            st.session_state.verification_saved = True
        
        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col_pdf, col_reset = st.columns(2)
        
        with col_pdf:
            if REPORTLAB_AVAILABLE:
                pdf_buffer = generate_pdf_report(verification_data)
                if pdf_buffer:
                    st.download_button(
                        label="📄 DOWNLOAD PDF REPORT",
                        data=pdf_buffer,
                        file_name=f"bioverify_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            else:
                text_report = generate_simple_report(verification_data) if 'generate_simple_report' in dir() else "Report generation not available"
                st.download_button(
                    label="📄 DOWNLOAD TEXT REPORT",
                    data=text_report,
                    file_name=f"bioverify_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        
        with col_reset:
            if st.button("🔄 START NEW VERIFICATION", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Premium Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3 style="font-family: 'Orbitron', sans-serif; font-size: 2rem; margin-bottom: 15px;">👤 BIOVERIFY</h3>
        <p style="font-size: 1.1rem; margin: 10px 0;">Next-Generation Security Platform</p>
        <p style="margin-top: 15px; font-size: 1rem; opacity: 0.9;">
            Powered by Advanced Machine Learning & Computer Vision Technology
        </p>
        <p style="margin-top: 10px; font-size: 0.9rem; opacity: 0.7;">
            © 2024 BioVerify Systems | Securing Digital Identities
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()