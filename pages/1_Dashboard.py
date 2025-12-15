"""
BioVerify Admin Dashboard
Shows verification statistics, charts, and recent flagged accounts
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
import random

st.set_page_config(
    page_title="BioVerify Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .stat-card {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #667eea;
    }
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
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
            <div class="stat-label">Total Verifications</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #38ef7d;">{data['passed_count']}</div>
            <div class="stat-label">Passed (Real)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #eb3349;">{data['failed_count']}</div>
            <div class="stat-label">Failed (Fake)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        rate = (data['failed_count'] / data['total_count'] * 100) if data['total_count'] > 0 else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number" style="color: #f093fb;">{rate:.1f}%</div>
            <div class="stat-label">Fake Detection Rate</div>
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
                col_a, col_b, col_c, col_d = st.columns([2, 2, 2, 2])
                with col_a:
                    st.markdown(f"**@{row['username']}**")
                with col_b:
                    st.markdown(f"Risk: {row['account_score']}%")
                with col_c:
                    st.markdown(f"Liveness: {row['liveness_score']}%")
                with col_d:
                    st.markdown(f"🚫 {row['verdict']}")
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


if __name__ == "__main__":
    main()
