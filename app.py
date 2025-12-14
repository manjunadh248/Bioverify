"""
BioVerify: Multi-Modal Fake Account Detector
Main Streamlit Dashboard - Enhanced Version with Timer
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
    page_title="BioVerify: Fake Account Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .step-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .timer-display {
        font-size: 3rem;
        font-weight: bold;
        color: #FF5722;
        text-align: center;
        padding: 20px;
    }
    .blink-counter {
        font-size: 2rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        padding: 10px;
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

def create_gauge_chart(value, title, threshold=50):
    """Create a gauge chart for risk visualization"""
    color = "red" if value >= 70 else "orange" if value >= 40 else "green"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': threshold},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'lightgreen'},
                {'range': [40, 70], 'color': 'lightyellow'},
                {'range': [70, 100], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def calculate_liveness_percentage(blink_count, face_detected_duration, total_duration=20):
    """
    Calculate liveness percentage based on:
    - Number of blinks detected
    - Time face was detected vs total time
    """
    # Blink score (0-60 points): 20 points per blink, max 3 blinks
    blink_score = min(blink_count * 20, 60)
    
    # Face detection score (0-40 points): Based on how long face was visible
    face_score = (face_detected_duration / total_duration) * 40
    
    total_score = blink_score + face_score
    
    return min(total_score, 100)

def run_liveness_test_with_timer(duration=20):
    """
    Run liveness test with fixed duration timer
    Returns: (blink_count, liveness_percentage, face_detected)
    """
    st.markdown('<div class="info-box"><h4>📹 Camera Verification Started</h4><p>The test will run for 20 seconds. Please blink naturally and keep your face visible.</p></div>', unsafe_allow_html=True)
    
    detector = LivenessDetector(required_blinks=3)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check your camera permissions.")
        return 0, 0, False
    
    # Create placeholders
    col1, col2 = st.columns([3, 1])
    
    with col1:
        frame_placeholder = st.empty()
    
    with col2:
        timer_placeholder = st.empty()
        blink_placeholder = st.empty()
        face_status_placeholder = st.empty()
    
    # Timer variables
    start_time = time.time()
    face_detected_time = 0
    last_face_detected = False
    last_time = start_time
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = max(0, duration - elapsed)
            
            # Stop if time is up
            if elapsed >= duration:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, ear, blink_detected, face_detected = detector.process_frame(frame)
            
            # Track face detection time
            time_delta = current_time - last_time
            if face_detected:
                face_detected_time += time_delta
                last_face_detected = True
            else:
                last_face_detected = False
            last_time = current_time
            
            # Add overlays to frame
            # Timer overlay
            cv2.putText(processed_frame, f"Time: {int(remaining)}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            if face_detected:
                cv2.putText(processed_frame, f"EAR: {ear:.2f}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, f"Blinks: {detector.blink_count}", (10, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(processed_frame, "Face: DETECTED", (10, 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(processed_frame, "Face: NOT DETECTED", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert BGR to RGB for Streamlit
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            try:
                frame_placeholder.image(processed_frame, channels="RGB", use_container_width=True)
            except TypeError:
                frame_placeholder.image(processed_frame, channels="RGB", width=700)
            
            # Update side panel
            timer_placeholder.markdown(f'<div class="timer-display">{int(remaining)}s</div>', unsafe_allow_html=True)
            blink_placeholder.markdown(f'<div class="blink-counter">👁️ Blinks: {detector.blink_count}</div>', unsafe_allow_html=True)
            
            if last_face_detected:
                face_status_placeholder.success("✅ Face Detected")
            else:
                face_status_placeholder.error("❌ No Face Detected")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.03)
                
    finally:
        cap.release()
        detector.cleanup()
        frame_placeholder.empty()
    
    # Calculate final liveness percentage
    liveness_percentage = calculate_liveness_percentage(
        detector.blink_count, 
        face_detected_time, 
        duration
    )
    
    # Store results
    st.session_state.blink_count = detector.blink_count
    st.session_state.liveness_score = liveness_percentage
    st.session_state.face_detected = face_detected_time > 0
    st.session_state.verification_complete = True
    
    return detector.blink_count, liveness_percentage, face_detected_time > 0

# Main App
def main():
    # Header
    st.markdown('<div class="main-header">🔍 BioVerify</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Modal Fake Account Detection System</div>', unsafe_allow_html=True)
    
    # Load model
    model, feature_names, accuracy, model_loaded = load_model()
    
    # Model status
    if not model_loaded:
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Demo Mode Active</h4>
            <p>The ML model is not trained yet. Running in <b>Demo Mode</b> with rule-based predictions.</p>
            <p><b>To enable full ML predictions:</b></p>
            <ol>
                <li>Download dataset from <a href="https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts" target="_blank">Kaggle</a></li>
                <li>Place <code>train.csv</code> in <code>data/</code> folder</li>
                <li>Run: <code>python train_model.py</code></li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.success(f"✅ ML Model Loaded | Accuracy: {accuracy*100:.2f}%")
    
    # Progress indicator
    st.markdown("---")
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    
    with progress_col1:
        if st.session_state.current_step >= 1:
            st.markdown('<div class="step-box"><h3>✅ Step 1</h3><p>User Information</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="step-box" style="opacity: 0.5;"><h3>1</h3><p>User Information</p></div>', unsafe_allow_html=True)
    
    with progress_col2:
        if st.session_state.current_step >= 2:
            st.markdown('<div class="step-box"><h3>✅ Step 2</h3><p>Camera Verification</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="step-box" style="opacity: 0.5;"><h3>2</h3><p>Camera Verification</p></div>', unsafe_allow_html=True)
    
    with progress_col3:
        if st.session_state.current_step >= 3:
            st.markdown('<div class="step-box"><h3>✅ Step 3</h3><p>Final Results</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="step-box" style="opacity: 0.5;"><h3>3</h3><p>Final Results</p></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # STEP 1: User Information Collection
    if st.session_state.current_step == 1:
        st.subheader("📋 Step 1: Enter Account Information")
        
        with st.form("account_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Basic Statistics")
                followers = st.number_input("👥 Followers", min_value=0, value=150, step=10)
                following = st.number_input("➕ Following", min_value=0, value=200, step=10)
                posts = st.number_input("📸 Posts", min_value=0, value=25, step=1)
                
                st.markdown("#### Profile Features")
                has_profile_pic = st.selectbox("🖼️ Has Profile Picture?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
                bio_length = st.slider("📝 Bio Length (characters)", 0, 150, 50)
                has_external_url = st.selectbox("🔗 Has External URL?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            with col2:
                st.markdown("#### Account Settings")
                is_private = st.selectbox("🔒 Private Account?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                
                st.markdown("#### Advanced Metrics")
                username_ratio = st.slider("🔢 Numbers/Length in Username", 0.0, 1.0, 0.2, 0.01)
                fullname_words = st.number_input("📛 Words in Full Name", min_value=0, value=2, step=1)
                fullname_ratio = st.slider("🔢 Numbers/Length in Full Name", 0.0, 1.0, 0.0, 0.01)
                name_equals_username = st.selectbox("👤 Name == Username?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            
            submitted = st.form_submit_button("➡️ Proceed to Camera Verification", use_container_width=True)
            
            if submitted:
                # Store user inputs
                st.session_state.user_inputs = {
                    'followers': followers,
                    'following': following,
                    'posts': posts,
                    'profile_pic': has_profile_pic,
                    'bio_length': bio_length,
                    'external_url': has_external_url,
                    'private': is_private,
                    'username_ratio': username_ratio,
                    'fullname_words': fullname_words,
                    'fullname_ratio': fullname_ratio,
                    'name_equals_username': name_equals_username
                }
                
                # Calculate account risk
                if model_loaded:
                    st.session_state.account_fake_prob = predict_fake_probability(model, feature_names, st.session_state.user_inputs)
                else:
                    st.session_state.account_fake_prob = demo_mode_prediction(st.session_state.user_inputs)
                
                # Move to step 2
                st.session_state.current_step = 2
                st.rerun()
    
    # STEP 2: Camera Verification
    elif st.session_state.current_step == 2:
        st.subheader("📹 Step 2: Camera Verification (20 Seconds)")
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Verification Instructions:</h4>
            <ul>
                <li>✅ The test will run for <b>20 seconds</b></li>
                <li>👁️ Please <b>blink naturally</b> 2-3 times during the test</li>
                <li>😊 Keep your <b>face visible</b> to the camera</li>
                <li>💯 Your liveness score depends on blinks detected and face visibility</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🎥 Start 20-Second Camera Test", type="primary", use_container_width=True):
                blink_count, liveness_percentage, face_detected = run_liveness_test_with_timer(duration=20)
                
                # Show results
                st.markdown("---")
                st.subheader("📊 Camera Verification Results")
                
                result_col1, result_col2, result_col3 = st.columns(3)
                
                with result_col1:
                    st.metric("👁️ Blinks Detected", blink_count)
                
                with result_col2:
                    st.metric("✅ Liveness Score", f"{liveness_percentage:.1f}%")
                
                with result_col3:
                    if face_detected:
                        st.metric("😊 Face Detection", "Success")
                    else:
                        st.metric("😊 Face Detection", "Failed")
                
                # Interpretation
                if liveness_percentage >= 70:
                    st.success("✅ Excellent! Strong human verification detected.")
                elif liveness_percentage >= 40:
                    st.warning("⚠️ Moderate verification. Some concerns detected.")
                else:
                    st.error("❌ Poor verification. High risk of bot/fake activity.")
        
        with col2:
            if st.session_state.verification_complete:
                st.info("✅ Verification Complete!")
                if st.button("➡️ View Final Results", use_container_width=True):
                    st.session_state.current_step = 3
                    st.rerun()
            
            if st.button("⬅️ Back to Step 1", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
    
    # STEP 3: Final Results
    elif st.session_state.current_step == 3:
        st.subheader("🎯 Step 3: Final Risk Assessment")
        
        account_fake_prob = st.session_state.account_fake_prob
        liveness_score = st.session_state.liveness_score
        
        # Display individual scores
        st.markdown("### 📊 Individual Scores")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📱 Account Analysis")
            try:
                st.plotly_chart(create_gauge_chart(account_fake_prob, "Account Fake Probability (%)"), use_container_width=True)
            except TypeError:
                st.plotly_chart(create_gauge_chart(account_fake_prob, "Account Fake Probability (%)"), use_column_width=True)
            
            if account_fake_prob >= 70:
                st.error("🚨 High Risk Account")
            elif account_fake_prob >= 40:
                st.warning("⚠️ Moderate Risk Account")
            else:
                st.success("✅ Low Risk Account")
        
        with col2:
            st.markdown("#### 👁️ Biometric Verification")
            try:
                st.plotly_chart(create_gauge_chart(100 - liveness_score, "Liveness Risk (%)"), use_container_width=True)
            except TypeError:
                st.plotly_chart(create_gauge_chart(100 - liveness_score, "Liveness Risk (%)"), use_column_width=True)
            
            st.metric("Blinks Detected", st.session_state.blink_count)
            st.metric("Liveness Score", f"{liveness_score:.1f}%")
        
        # Calculate Final Risk with VETO logic
        st.markdown("---")
        st.markdown("### 🎯 Combined Final Risk Score")
        
        # CRITICAL VETO: If liveness is very poor (< 30%), override to 100% risk
        if liveness_score < 30:
            final_risk = 100
            st.markdown("""
            <div class="danger-box">
                <h3>🚨 CRITICAL: VERIFICATION COMPLETELY FAILED</h3>
                <p><b>Final Risk: 100% (CONFIRMED FAKE/BOT)</b></p>
                <p>❌ Liveness score is critically low ({:.1f}%)</p>
                <p>❌ No face detected OR no blinks OR camera was not used properly</p>
                <p><b>⚠️ IMMEDIATE ACTION REQUIRED: BLOCK THIS ACCOUNT</b></p>
            </div>
            """.format(liveness_score), unsafe_allow_html=True)
        
        # If liveness is poor but not critical (30-50%), heavily penalize
        elif liveness_score < 50:
            final_risk = 100  # Still block, but with different message
            st.markdown("""
            <div class="danger-box">
                <h3>🚨 ALERT: LIVENESS VERIFICATION FAILED</h3>
                <p><b>Final Risk: 100% (LIKELY FAKE/BOT)</b></p>
                <p>⚠️ Liveness score is below acceptable threshold ({:.1f}%)</p>
                <p>⚠️ Insufficient blinks detected or poor face visibility</p>
                <p><b>Recommendation: BLOCK or require re-verification</b></p>
            </div>
            """.format(liveness_score), unsafe_allow_html=True)
        
        # Normal calculation: Both verifications passed reasonably
        else:
            # Weighted formula: 70% account analysis + 30% liveness risk
            final_risk = (0.7 * account_fake_prob) + (0.3 * (100 - liveness_score))
            
            # Display gauge
            try:
                st.plotly_chart(create_gauge_chart(final_risk, "Final Combined Risk Score (%)"), use_container_width=True)
            except TypeError:
                st.plotly_chart(create_gauge_chart(final_risk, "Final Combined Risk Score (%)"), use_column_width=True)
            
            # Breakdown
            st.markdown("#### 📊 Risk Calculation Breakdown")
            breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
            
            with breakdown_col1:
                st.metric("📱 Account Risk (70% weight)", f"{account_fake_prob:.1f}%")
                st.caption(f"Contributes: {0.7 * account_fake_prob:.1f}%")
            
            with breakdown_col2:
                st.metric("👁️ Liveness Risk (30% weight)", f"{100-liveness_score:.1f}%")
                st.caption(f"Contributes: {0.3 * (100-liveness_score):.1f}%")
            
            with breakdown_col3:
                st.metric("🎯 Final Combined Risk", f"{final_risk:.1f}%")
                st.caption("Weighted average")
            
            # Final verdict
            st.markdown("---")
            st.markdown("### 📋 Final Verdict")
            
            if final_risk >= 70:
                st.markdown(f"""
                <div class="danger-box">
                    <h3>🚨 HIGH RISK - LIKELY FAKE ACCOUNT</h3>
                    <p><b>Final Risk Score: {final_risk:.1f}%</b></p>
                    <p><b>Account Probability:</b> {account_fake_prob:.1f}%</p>
                    <p><b>Liveness Score:</b> {liveness_score:.1f}%</p>
                    <p><b>Blinks Detected:</b> {st.session_state.blink_count}</p>
                    <hr>
                    <p><b>🚫 RECOMMENDATION: BLOCK THIS ACCOUNT</b></p>
                    <p>This account exhibits multiple red flags suggesting it is either a bot, spam account, or fraudulent profile.</p>
                </div>
                """, unsafe_allow_html=True)
            
            elif final_risk >= 40:
                st.markdown(f"""
                <div class="warning-box">
                    <h3>⚠️ MODERATE RISK - SUSPICIOUS ACCOUNT</h3>
                    <p><b>Final Risk Score: {final_risk:.1f}%</b></p>
                    <p><b>Account Probability:</b> {account_fake_prob:.1f}%</p>
                    <p><b>Liveness Score:</b> {liveness_score:.1f}%</p>
                    <p><b>Blinks Detected:</b> {st.session_state.blink_count}</p>
                    <hr>
                    <p><b>⚠️ RECOMMENDATION: MONITOR & RESTRICT</b></p>
                    <p>Require additional verification steps, limit account capabilities, or flag for manual review.</p>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown(f"""
                <div class="success-box">
                    <h3>✅ LOW RISK - LIKELY GENUINE ACCOUNT</h3>
                    <p><b>Final Risk Score: {final_risk:.1f}%</b></p>
                    <p><b>Account Probability:</b> {account_fake_prob:.1f}%</p>
                    <p><b>Liveness Score:</b> {liveness_score:.1f}%</p>
                    <p><b>Blinks Detected:</b> {st.session_state.blink_count}</p>
                    <hr>
                    <p><b>✅ RECOMMENDATION: APPROVE ACCOUNT</b></p>
                    <p>Account appears legitimate based on both metadata analysis and biometric verification. Safe to proceed with normal operations.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Start New Verification", use_container_width=True):
                # Reset all session state
                st.session_state.current_step = 1
                st.session_state.liveness_score = 0
                st.session_state.blink_count = 0
                st.session_state.face_detected = False
                st.session_state.account_fake_prob = 0
                st.session_state.user_inputs = {}
                st.session_state.verification_complete = False
                st.rerun()
        
        with col2:
            if st.button("⬅️ Re-do Camera Test", use_container_width=True):
                st.session_state.current_step = 2
                st.session_state.verification_complete = False
                st.rerun()
        
        with col3:
            if st.button("📝 Modify Account Info", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p><b>BioVerify v2.0</b> | Enhanced Multi-Step Verification System</p>
            <p>🔒 Metadata Analysis + ⏱️ Timed Biometric Verification</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    