"""
BioVerify Admin Dashboard
Shows verification statistics, charts, recent flagged accounts, and risk threshold settings
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
import random

# Import user database for user management
try:
    from user_database import get_db, UserDatabase
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

CONFIG_FILE = "config.json"

def load_config():
    """Load configuration from file"""
    default_config = {
        "risk_thresholds": {"low": 30, "high": 70},
        "verification": {"liveness_duration": 20, "liveness_pass_threshold": 50},
        "ui": {"show_risk_details": True, "show_reason_codes": True}
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return default_config
    return default_config

def save_config(config):
    """Save configuration to file"""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

st.set_page_config(
    page_title="BioVerify Dashboard",
    page_icon="📊",
    layout="wide"
)

# Premium Custom CSS with Sidebar Styling and Dynamic Effects
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Main App Background with Animation */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
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
    
    /* Premium Stat Cards with Glow Effect */
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 25px;
        text-align: center;
        border: 2px solid rgba(102, 126, 234, 0.4);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(102, 126, 234, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stat-card::before {
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
    
    .stat-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.5), 0 0 40px rgba(240, 147, 251, 0.3);
        border-color: rgba(240, 147, 251, 0.6);
    }
    
    .stat-number {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 3rem;
        font-weight: 900;
        color: #667eea;
        text-shadow: 0 0 20px rgba(102, 126, 234, 0.6);
        position: relative;
        z-index: 1;
        animation: pulseGlow 2s ease-in-out infinite;
    }
    
    @keyframes pulseGlow {
        0%, 100% { text-shadow: 0 0 20px rgba(102, 126, 234, 0.6); }
        50% { text-shadow: 0 0 35px rgba(102, 126, 234, 0.9), 0 0 50px rgba(240, 147, 251, 0.5); }
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #a8edea;
        text-transform: none;
        letter-spacing: 1px;
        font-weight: 500;
        margin-top: 12px;
        position: relative;
        z-index: 1;
        opacity: 0.9;
        text-shadow: 0 0 10px rgba(168, 237, 234, 0.3);
    }
    
    /* Dashboard Title Styling */
    h1 {
        font-family: 'Orbitron', sans-serif !important;
        color: #f093fb !important;
        text-shadow: 0 0 30px rgba(240, 147, 251, 0.5), 0 0 60px rgba(102, 126, 234, 0.3) !important;
        letter-spacing: 2px !important;
    }
    
    h2, h3 {
        color: white !important;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #a8edea !important;
    }
    
    /* Flagged Account Rows */
    .flagged-account {
        background: linear-gradient(135deg, rgba(235, 51, 73, 0.15), rgba(244, 92, 67, 0.15));
        border: 1px solid rgba(235, 51, 73, 0.4);
        border-radius: 15px;
        padding: 15px 20px;
        margin: 10px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .flagged-account:hover {
        transform: translateX(5px);
        box-shadow: 0 0 25px rgba(235, 51, 73, 0.4);
        border-color: rgba(235, 51, 73, 0.7);
    }
    
    .flagged-username {
        color: #f093fb;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .flagged-stat {
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.95rem;
    }
    
    .flagged-verdict {
        color: #eb3349;
        font-weight: 700;
        padding: 5px 15px;
        border-radius: 20px;
        background: rgba(235, 51, 73, 0.25);
        border: 1px solid rgba(235, 51, 73, 0.5);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
        background-size: 200% !important;
        color: white !important;
        border: none !important;
        padding: 12px 30px !important;
        font-weight: 700 !important;
        border-radius: 30px !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.4s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(240, 147, 251, 0.5) !important;
        background-position: 100% !important;
    }
    
    /* DataFrame Styling */
    .stDataFrame {
        border-radius: 15px !important;
        overflow: hidden !important;
    }
    
    [data-testid="stDataFrame"] > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
    }
    
    /* Info Box */
    .stAlert {
        background: rgba(102, 126, 234, 0.15) !important;
        border: 1px solid rgba(102, 126, 234, 0.4) !important;
        border-radius: 15px !important;
        color: white !important;
    }
    
    /* Horizontal Rule */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), rgba(240, 147, 251, 0.5), transparent) !important;
        margin: 25px 0 !important;
    }
    
    /* Section Headers with Icons */
    .section-header {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 15px 0;
    }
    
    .section-icon {
        font-size: 1.8rem;
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
</style>
""", unsafe_allow_html=True)

DATA_FILE = "data/verifications.json"

def load_verification_data():
    """Load verification history"""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"verifications": [], "total_count": 0, "passed_count": 0, "failed_count": 0}

def save_verification(username, account_score, liveness_score, verdict, is_fake):
    """Save a new verification record"""
    data = load_verification_data()
    
    record = {
        "timestamp": datetime.now().isoformat(),
        "username": username,
        "account_score": account_score,
        "liveness_score": liveness_score,
        "verdict": verdict,
        "is_fake": is_fake
    }
    
    data["verifications"].append(record)
    data["total_count"] += 1
    if is_fake:
        data["failed_count"] += 1
    else:
        data["passed_count"] += 1
    
    # Keep only last 100 records
    if len(data["verifications"]) > 100:
        data["verifications"] = data["verifications"][-100:]
    
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def generate_sample_data():
    """Generate sample data for demo"""
    data = {"verifications": [], "total_count": 0, "passed_count": 0, "failed_count": 0}
    
    verdicts = ["REAL", "SUSPICIOUS", "FAKE"]
    usernames = ["user" + str(i) for i in range(1, 51)]
    
    for i in range(50):
        is_fake = random.random() < 0.3
        verdict = random.choice(verdicts) if not is_fake else "FAKE"
        
        record = {
            "timestamp": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            "username": random.choice(usernames),
            "account_score": random.randint(10, 95),
            "liveness_score": random.randint(30, 100),
            "verdict": verdict,
            "is_fake": is_fake
        }
        
        data["verifications"].append(record)
        data["total_count"] += 1
        if is_fake:
            data["failed_count"] += 1
        else:
            data["passed_count"] += 1
    
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data


def main():
    st.title("📊 BioVerify Admin Dashboard")
    st.markdown("---")
    
    # Load data
    data = load_verification_data()
    
    # Add sample data button
    col_header1, col_header2 = st.columns([4, 1])
    with col_header2:
        if st.button("🔄 Generate Demo Data"):
            data = generate_sample_data()
            st.rerun()
    
    # Summary Stats
    st.subheader("📈 Verification Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{data['total_count']}</div>
            <div class="stat-label">📊 Total Verifications</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #38ef7d;">{data['passed_count']}</div>
            <div class="stat-label">✅ Verified Real</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #eb3349;">{data['failed_count']}</div>
            <div class="stat-label">🚫 Detected Fake</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        rate = (data['failed_count'] / data['total_count'] * 100) if data['total_count'] > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #f093fb;">{rate:.1f}%</div>
            <div class="stat-label">⚡ Detection Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if data["verifications"]:
        # Charts Row
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.subheader("🥧 Verification Results")
            
            # Pie chart
            labels = ['Real Accounts', 'Fake Accounts']
            values = [data['passed_count'], data['failed_count']]
            colors = ['#38ef7d', '#eb3349']
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values,
                hole=0.4,
                marker_colors=colors
            )])
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            st.subheader("📊 Risk Score Distribution")
            
            df = pd.DataFrame(data["verifications"])
            
            fig_hist = px.histogram(
                df, 
                x="account_score",
                nbins=20,
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                xaxis_title="Account Risk Score",
                yaxis_title="Count",
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        st.markdown("---")
        
        # Recent Flagged Accounts
        st.subheader("🚨 Recent Flagged Accounts")
        
        df = pd.DataFrame(data["verifications"])
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        df = df.sort_values('timestamp', ascending=False)
        
        flagged = df[df['is_fake'] == True].head(10)
        
        if len(flagged) > 0:
            for _, row in flagged.iterrows():
                st.markdown(f"""
                <div class="flagged-account">
                    <span class="flagged-username">@{row['username']}</span>
                    <span class="flagged-stat">Risk: {row['account_score']}%</span>
                    <span class="flagged-stat">Liveness: {row['liveness_score']}%</span>
                    <span class="flagged-verdict">🚫 FAKE</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No flagged accounts yet.")
        
        st.markdown("---")
        
        # Full History Table
        st.subheader("📋 Verification History")
        
        df_display = df[['timestamp', 'username', 'account_score', 'liveness_score', 'verdict']].copy()
        df_display.columns = ['Time', 'Username', 'Account Score', 'Liveness Score', 'Verdict']
        df_display['Time'] = df_display['Time'].dt.strftime('%Y-%m-%d %H:%M')
        
        st.dataframe(df_display.head(20), use_container_width=True)
    
    else:
        st.info("📭 No verification data yet. Run some verifications in the main app!")
    
    # ==================== USER DATABASE SECTION ====================
    st.markdown("---")
    st.subheader("👥 Registered Users")
    
    if DB_AVAILABLE:
        try:
            db = get_db()
            stats = db.get_stats()
            
            # User stats
            u_col1, u_col2, u_col3, u_col4 = st.columns(4)
            
            with u_col1:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{stats.get('total_users', 0)}</div>
                    <div class="stat-label">👥 Total Users</div>
                </div>
                """, unsafe_allow_html=True)
            
            with u_col2:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #38ef7d;">{stats.get('verified_users', 0)}</div>
                    <div class="stat-label">✅ Verified</div>
                </div>
                """, unsafe_allow_html=True)
            
            with u_col3:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #f093fb;">{stats.get('flagged_users', 0)}</div>
                    <div class="stat-label">⚠️ Flagged</div>
                </div>
                """, unsafe_allow_html=True)
            
            with u_col4:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #eb3349;">{stats.get('blocked_users', 0)}</div>
                    <div class="stat-label">🚫 Blocked</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Show recent users
            users = db.get_all_users(limit=15)
            if users:
                st.markdown("### Recent Users")
                user_data = []
                for user in users:
                    user_data.append({
                        "Username": user.username,
                        "Status": user.verification_status.upper(),
                        "Liveness": "✅" if user.previous_liveness_passed else "❌",
                        "Gov ID": "✅" if user.previous_govid_passed else "❌",
                        "Logins": user.login_count,
                        "Last Login": user.last_login[:10] if user.last_login else "N/A"
                    })
                st.dataframe(pd.DataFrame(user_data), use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not load user database: {str(e)}")
    else:
        st.info("User database module not available.")
    
    # ==================== RISK THRESHOLD SETTINGS ====================
    st.markdown("---")
    st.subheader("⚙️ Risk Threshold Settings")
    
    config = load_config()
    thresholds = config.get("risk_thresholds", {"low": 30, "high": 70})
    
    st.markdown("""
    <div style="background: rgba(102, 126, 234, 0.15); padding: 20px; border-radius: 15px; 
                border: 1px solid rgba(102, 126, 234, 0.4); margin-bottom: 20px;">
        <p style="color: white; margin: 0;">
            <b>🎚️ Configure risk thresholds</b> to control verification requirements:
        </p>
        <ul style="color: rgba(255, 255, 255, 0.8); margin: 10px 0 0 0;">
            <li><b>Below Low Threshold:</b> Light verification only</li>
            <li><b>Between Thresholds:</b> Government ID + Liveness required</li>
            <li><b>Above High Threshold:</b> Account blocked for manual review</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("threshold_form"):
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            low_threshold = st.slider(
                "Low Risk Threshold",
                min_value=10,
                max_value=50,
                value=thresholds.get("low", 30),
                help="Scores below this are considered LOW risk"
            )
        
        with col_t2:
            high_threshold = st.slider(
                "High Risk Threshold",
                min_value=50,
                max_value=90,
                value=thresholds.get("high", 70),
                help="Scores above this are considered HIGH risk"
            )
        
        # Validation
        if low_threshold >= high_threshold:
            st.error("⚠️ Low threshold must be less than high threshold!")
        
        save_button = st.form_submit_button("💾 Save Settings", use_container_width=True)
        
        if save_button and low_threshold < high_threshold:
            config["risk_thresholds"]["low"] = low_threshold
            config["risk_thresholds"]["high"] = high_threshold
            save_config(config)
            st.success("✅ Settings saved successfully! Changes will apply to new verifications.")
    
    # Show current thresholds visualization
    st.markdown("### 📊 Current Risk Tiers")
    
    fig = go.Figure()
    
    # Add colored bars for each tier
    fig.add_trace(go.Bar(
        x=[thresholds.get("low", 30)],
        y=["Risk"],
        orientation='h',
        name="Low Risk",
        marker_color="#38ef7d",
        text=f"LOW (0-{thresholds.get('low', 30)})",
        textposition="inside"
    ))
    
    fig.add_trace(go.Bar(
        x=[thresholds.get("high", 70) - thresholds.get("low", 30)],
        y=["Risk"],
        orientation='h',
        name="Medium Risk",
        marker_color="#f093fb",
        text=f"MEDIUM ({thresholds.get('low', 30)}-{thresholds.get('high', 70)})",
        textposition="inside"
    ))
    
    fig.add_trace(go.Bar(
        x=[100 - thresholds.get("high", 70)],
        y=["Risk"],
        orientation='h',
        name="High Risk",
        marker_color="#eb3349",
        text=f"HIGH ({thresholds.get('high', 70)}-100)",
        textposition="inside"
    ))
    
    fig.update_layout(
        barmode='stack',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=120,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        xaxis=dict(range=[0, 100], title="Risk Score"),
        yaxis=dict(showticklabels=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
