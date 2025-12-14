"""
BioVerify: Multi-Modal Fake Account Detector
Fixed Version with Improved UI/UX
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import cv2
from utils import LivenessDetector
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="BioVerify: AI-Powered Fake Account Detector",
    page_icon="🛡️",
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
    
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: neonGlow 3s ease-in-out infinite alternate;
        letter-spacing: 3px;
    }
    
    @keyframes neonGlow {
        from {
            filter: drop-shadow(0 0 20px #667eea) drop-shadow(0 0 40px #764ba2);
        }
        to {
            filter: drop-shadow(0 0 30px #764ba2) drop-shadow(0 0 60px #f093fb);
        }
    }
    
    .sub-header {
        font-size: 1.5rem;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
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
        gap: 40px;
        margin: 30px 0;
        flex-wrap: wrap;
    }
    
    .stat-box {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.4);
        padding: 25px 40px;
        border-radius: 20px;
        text-align: center;
        min-width: 180px;
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.5);
    }
    
    .stat-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        margin: 10px 0;
    }
    
    .stat-label {
        font-size: 1.1rem;
        color: rgba(255, 255, 255, 0.9);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
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
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px !important;
        backdrop-filter: blur(10px);
        font-size: 1.1rem !important;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5) !important;
    }
    
    label {
        color: white !important;
        font-weight: 600 !important;
        letter-spacing: 1px !important;
        font-size: 1.1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
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
    """Predict fake account probability"""
    feature_values = []
    for feature in feature_names:
        if feature == '#followers':
            feature_values.append(user_inputs.get('followers', 0))
        elif feature == '#follows':
            feature_values.append(user_inputs.get('following', 0))
        elif feature == '#posts':
            feature_values.append(user_inputs.get('posts', 0))
        elif feature == 'profile pic':
            feature_values.append(user_inputs.get('profile_pic', 1))
        elif feature == 'description length':
            feature_values.append(user_inputs.get('bio_length', 0))
        elif feature == 'external URL':
            feature_values.append(user_inputs.get('external_url', 0))
        elif feature == 'private':
            feature_values.append(user_inputs.get('private', 0))
        elif feature == 'nums/length username':
            feature_values.append(user_inputs.get('username_ratio', 0))
        elif feature == 'fullname words':
            feature_values.append(user_inputs.get('fullname_words', 0))
        elif feature == 'nums/length fullname':
            feature_values.append(user_inputs.get('fullname_ratio', 0))
        elif feature == 'name==username':
            feature_values.append(user_inputs.get('name_equals_username', 0))
        else:
            feature_values.append(0)
    
    features_array = np.array(feature_values).reshape(1, -1)
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

def run_liveness_test_with_timer(duration=20):
    """Run liveness test with timer and enhanced UI - CENTERED CAMERA"""
    st.markdown("""
    <div class="info-box">
        <h3 style="margin: 0;">🎥 Live Camera Verification in Progress</h3>
        <p style="margin: 10px 0 0 0;">Duration: 20 seconds | Keep face visible and blink naturally (3+ times required)</p>
    </div>
    """, unsafe_allow_html=True)
    
    detector = LivenessDetector(required_blinks=3)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Camera access denied. Please enable camera permissions.")
        return 0, 0, False
    
    # CENTERED CAMERA - Full width
    st.markdown('<div class="camera-container">', unsafe_allow_html=True)
    frame_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Stats below camera - CENTERED
    st.markdown('<div class="camera-stats">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        timer_placeholder = st.empty()
    with col2:
        blink_placeholder = st.empty()
    with col3:
        face_status_placeholder = st.empty()
    with col4:
        ear_placeholder = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)
    
    progress_placeholder = st.empty()
    
    start_time = time.time()
    face_detected_time = 0
    last_face_detected = False
    last_time = start_time
    last_blink_count = 0
    current_ear = 0
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = max(0, duration - elapsed)
            progress = (elapsed / duration) * 100
            
            if elapsed >= duration:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            processed_frame, ear, blink_detected, face_detected = detector.process_frame(frame)
            current_ear = ear
            
            time_delta = current_time - last_time
            if face_detected:
                face_detected_time += time_delta
                last_face_detected = True
            else:
                last_face_detected = False
            last_time = current_time
            
            # Enhanced frame overlays
            cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], 80), (15, 12, 41), -1)
            border_color = (56, 239, 125) if face_detected else (235, 51, 73)
            cv2.rectangle(processed_frame, (0, 0), (processed_frame.shape[1], 80), border_color, 5)
            
            cv2.putText(processed_frame, f"TIME: {int(remaining)}s | BLINKS: {detector.blink_count}", 
                       (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
            
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
            
            # Update stats below camera - CENTERED
            timer_placeholder.markdown(
                f'<div class="stat-box"><div class="stat-value">{int(remaining)}</div><div class="stat-label">⏱️ Seconds</div></div>', 
                unsafe_allow_html=True
            )
            
            blink_placeholder.markdown(
                f'<div class="stat-box"><div class="stat-value">{detector.blink_count}</div><div class="stat-label">👁️ Blinks</div></div>', 
                unsafe_allow_html=True
            )
            
            if last_face_detected:
                face_status_placeholder.markdown(
                    '<div class="stat-box" style="border-color: #38ef7d;"><div class="stat-value">✅</div><div class="stat-label">Face OK</div></div>',
                    unsafe_allow_html=True
                )
            else:
                face_status_placeholder.markdown(
                    '<div class="stat-box" style="border-color: #eb3349;"><div class="stat-value">❌</div><div class="stat-label">No Face</div></div>',
                    unsafe_allow_html=True
                )
            
            ear_placeholder.markdown(
                f'<div class="stat-box"><div class="stat-value">{current_ear:.3f}</div><div class="stat-label">👁️ EAR</div></div>',
                unsafe_allow_html=True
            )
            
            progress_placeholder.progress(int(progress))
            time.sleep(0.03)
                
    finally:
        cap.release()
        detector.cleanup()
        frame_placeholder.empty()
    
    liveness_percentage = calculate_liveness_percentage(detector.blink_count, face_detected_time, duration)
    
    st.session_state.blink_count = detector.blink_count
    st.session_state.liveness_score = liveness_percentage
    st.session_state.face_detected = face_detected_time > 0
    st.session_state.verification_complete = True
    
    return detector.blink_count, liveness_percentage, face_detected_time > 0

def main():
    # Premium Animated Header with LOGO
    st.markdown("""
    <div style="text-align: center; padding: 50px 0 30px 0;">
        <div style="font-size: 5rem; margin-bottom: 20px;">🛡️</div>
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
    
    # Progress Steps
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        step_class = "step-card-completed" if st.session_state.current_step > 1 else "step-card-active" if st.session_state.current_step == 1 else "step-card-inactive"
        icon = "✅" if st.session_state.current_step > 1 else "📝" if st.session_state.current_step == 1 else "1"
        status = "Completed" if st.session_state.current_step > 1 else "Active" if st.session_state.current_step == 1 else "Pending"
        st.markdown(f"""
        <div class="step-card {step_class}">
            <h2 style="margin: 0; position: relative; z-index: 1; font-size: 2.5rem;">{icon}</h2>
            <p style="margin: 15px 0 0 0; font-size: 1.3rem; position: relative; z-index: 1; font-weight: 700;">STEP 1</p>
            <p style="margin: 5px 0 0 0; font-size: 1rem; position: relative; z-index: 1;">Account Analysis</p>
            <p style="margin: 10px 0 0 0; font-size: 0.9rem; opacity: 0.9; position: relative; z-index: 1;">{status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        step_class = "step-card-completed" if st.session_state.current_step > 2 else "step-card-active" if st.session_state.current_step == 2 else "step-card-inactive"
        icon = "✅" if st.session_state.current_step > 2 else "🎥" if st.session_state.current_step == 2 else "2"
        status = "Completed" if st.session_state.current_step > 2 else "Active" if st.session_state.current_step == 2 else "Pending"
        st.markdown(f"""
        <div class="step-card {step_class}">
            <h2 style="margin: 0; position: relative; z-index: 1; font-size: 2.5rem;">{icon}</h2>
            <p style="margin: 15px 0 0 0; font-size: 1.3rem; position: relative; z-index: 1; font-weight: 700;">STEP 2</p>
            <p style="margin: 5px 0 0 0; font-size: 1rem; position: relative; z-index: 1;">Biometric Scan</p>
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
    
    # STEP 1: Account Analysis
    if st.session_state.current_step == 1:
        st.markdown("""
        <div class="content-box">
            <h2>📝 Step 1: Account Information Analysis</h2>
            <p>Provide account details for AI-powered fake account detection</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("account_form"):
            c1, c2 = st.columns(2)
            
            with c1:
                followers = st.number_input("👥 Followers Count", min_value=0, value=100, step=1)
                following = st.number_input("➕ Following Count", min_value=0, value=150, step=1)
                posts = st.number_input("📸 Posts Count", min_value=0, value=20, step=1)
                profile_pic = st.selectbox("🖼️ Has Profile Picture?", options=[1, 0], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
                bio_length = st.number_input("📝 Bio Length (characters)", min_value=0, value=50, step=1)
            
            with c2:
                external_url = st.selectbox("🔗 Has External URL?", options=[0, 1], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
                private = st.selectbox("🔒 Private Account?", options=[0, 1], format_func=lambda x: "Yes ✅" if x == 1 else "No ❌")
                username = st.text_input("👤 Username", value="user123", help="Enter the account username")
                fullname = st.text_input("📛 Full Name", value="John Doe", help="Enter the full name on the account")
            
            submit = st.form_submit_button("🔍 Analyze Account", use_container_width=True)
            
            if submit:
                # Calculate features
                username_digits = sum(c.isdigit() for c in username)
                username_ratio = username_digits / len(username) if len(username) > 0 else 0
                
                fullname_words = len(fullname.split())
                fullname_digits = sum(c.isdigit() for c in fullname)
                fullname_ratio = fullname_digits / len(fullname) if len(fullname) > 0 else 0
                
                name_equals_username = 1 if username.lower() == fullname.lower().replace(" ", "") else 0
                
                st.session_state.user_inputs = {
                    'followers': followers,
                    'following': following,
                    'posts': posts,
                    'profile_pic': profile_pic,
                    'bio_length': bio_length,
                    'external_url': external_url,
                    'private': private,
                    'username_ratio': username_ratio,
                    'fullname_words': fullname_words,
                    'fullname_ratio': fullname_ratio,
                    'name_equals_username': name_equals_username,
                    'username': username,
                    'fullname': fullname
                }
                
                # Predict
                if model_loaded:
                    fake_prob = predict_fake_probability(model, feature_names, st.session_state.user_inputs)
                else:
                    fake_prob = demo_mode_prediction(st.session_state.user_inputs)
                
                st.session_state.account_fake_prob = fake_prob
                st.session_state.current_step = 2
                st.rerun()
    
    # STEP 2: Biometric Verification
    elif st.session_state.current_step == 2:
        st.markdown("""
        <div class="content-box">
            <h2>🎥 Step 2: Live Biometric Verification</h2>
            <p>Real-time facial liveness detection for identity confirmation</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div class="info-box">
            <h3>📋 Verification Instructions:</h3>
            <ul style="font-size: 1.1rem; line-height: 1.8;">
                <li>🎯 Position your face clearly in the camera frame</li>
                <li>👀 Look directly at the camera</li>
                <li>😊 Blink naturally (minimum 3 times required)</li>
                <li>⏱️ Maintain visibility for full 20 seconds</li>
                <li>💡 Ensure good lighting conditions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col_center = st.columns([1, 2, 1])[1]
        with col_center:
            if st.button("🚀 START VERIFICATION", use_container_width=True):
                blink_count, liveness_score, face_detected = run_liveness_test_with_timer()
                
                if liveness_score >= 50:
                    st.session_state.liveness_passed = True
                    st.session_state.current_step = 3
                    st.balloons()
                    st.rerun()
                else:
                    st.markdown(f"""
                    <div class="danger-box">
                        <h3>❌ Verification Failed</h3>
                        <p><b>Liveness Score:</b> {liveness_score:.1f}%</p>
                        <p>Please try again. Ensure proper lighting and blink naturally.</p>
                    </div>
                    """, unsafe_allow_html=True)
    
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
        
        st.markdown(f"""
        <div class="info-box">
            <h3>📋 Account Details</h3>
            <p style="font-size: 1.2rem;"><b>👤 Username:</b> {username}</p>
            <p style="font-size: 1.2rem;"><b>📛 Full Name:</b> {fullname}</p>
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
        combined_risk = (st.session_state.account_fake_prob * 0.6 + (100 - st.session_state.liveness_score) * 0.4)
        
        st.markdown("### 🎯 Final Security Verdict")
        
        if combined_risk < 40:
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
                <div class="metric-label">Blinks</div>
                <div class="metric-value">{st.session_state.blink_count}</div>
                <p style="margin: 5px 0 0 0;">👁️ Eye Activity</p>
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
        
        # Reset button
        st.markdown("<br>", unsafe_allow_html=True)
        col_reset = st.columns([1, 2, 1])[1]
        with col_reset:
            if st.button("🔄 START NEW VERIFICATION", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Premium Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h3 style="font-family: 'Orbitron', sans-serif; font-size: 2rem; margin-bottom: 15px;">🛡️ BIOVERIFY</h3>
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