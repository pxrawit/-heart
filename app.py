"""
============================================================
Heart Disease Prediction Web App
============================================================
รันด้วยคำสั่ง: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go

# ==================== Configuration ====================
# 🔧 แก้ไขข้อมูลผู้พัฒนาตรงนี้
DEVELOPER_INFO = {
    "name": "Your Name",
    "email": "your.email@example.com",
    "github": "https://github.com/your-username",
    "linkedin": "https://linkedin.com/in/your-profile",
    "institution": "Your University / Company"
}

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-title {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Result boxes */
    .result-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
    }
    
    .result-danger {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(235, 51, 73, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Contact Card */
    .contact-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    
    .contact-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        color: white !important;
    }
    
    .contact-item a {
        color: #ecf0f1 !important;
        text-decoration: none;
        transition: color 0.3s;
    }
    
    .contact-item a:hover {
        color: #3498db !important;
    }
    
    /* Developer Profile */
    .dev-profile {
        text-align: center;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .dev-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ecf0f1 !important;
        margin: 0.5rem 0;
    }
    
    .dev-role {
        font-size: 0.9rem;
        color: #bdc3c7 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Load Model ====================
@st.cache_resource
def load_model():
    """โหลดโมเดลที่บันทึกไว้"""
    model_path = "heart_disease_model.pkl"
    if not os.path.exists(model_path):
        st.error("❌ ไม่พบไฟล์โมเดล! กรุณาวางไฟล์ heart_disease_model.pkl ในโฟลเดอร์เดียวกัน")
        st.stop()
    return joblib.load(model_path)

try:
    artifacts = load_model()
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']
    model_metrics = artifacts.get('model_metrics', {})
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {e}")
    st.stop()

# ==================== Sidebar ====================
with st.sidebar:
    # Developer Profile
    st.markdown(
        f"""
        <div class="dev-profile">
            <div style="font-size: 3rem;">👨‍💻</div>
            <div class="dev-name">{DEVELOPER_INFO['name']}</div>
            <div class="dev-role">{DEVELOPER_INFO['institution']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("## 🫀 Heart Disease")
    st.markdown("### Prediction System")
    
    st.markdown("---")
    st.markdown("### 📊 โมเดล Information")
    st.info(f"""
    **Algorithm:** Decision Tree  
    **Accuracy:** {model_metrics.get('accuracy', 0):.2%}  
    **ROC-AUC:** {model_metrics.get('roc_auc', 0):.3f}  
    **Features:** {len(feature_names)}
    """)
    
    st.markdown("---")
    
    # 📧 Contact Information Section
    st.markdown("### 📞 ติดต่อผู้พัฒนา")
    
    st.markdown(
        f"""
        <div class="contact-card">
            <div class="contact-item">
                📧 <strong style="margin-left: 8px;">Email:</strong>
            </div>
            <div class="contact-item">
                <a href="mailto:{DEVELOPER_INFO['email']}?subject=Heart Disease App Inquiry" 
                   style="margin-left: 28px;">
                    {DEVELOPER_INFO['email']}
                </a>
            </div>
            <div class="contact-item">
                🐙 <strong style="margin-left: 8px;">GitHub:</strong>
            </div>
            <div class="contact-item">
                <a href="{DEVELOPER_INFO['github']}" target="_blank"
                   style="margin-left: 28px;">
                    View Repository
                </a>
            </div>
            <div class="contact-item">
                💼 <strong style="margin-left: 8px;">LinkedIn:</strong>
            </div>
            <div class="contact-item">
                <a href="{DEVELOPER_INFO['linkedin']}" target="_blank"
                   style="margin-left: 28px;">
                    Connect with me
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Quick Email Button
    st.markdown(
        f"""
        <a href="mailto:{DEVELOPER_INFO['email']}?subject=Heart Disease App Feedback&body=Hello {DEVELOPER_INFO['name']},%0D%0A%0D%0AI would like to provide feedback about your Heart Disease Prediction App.%0D%0A%0D%0A" 
           target="_blank" 
           style="text-decoration: none;">
            <div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                        color: white; 
                        padding: 0.8rem; 
                        border-radius: 10px; 
                        text-align: center; 
                        font-weight: 600;
                        margin-top: 0.5rem;">
                ✉️ ส่งอีเมลหาผู้พัฒนา
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.markdown("### 📝 คำแนะนำ")
    st.warning("""
    - กรอกข้อมูลให้ครบถ้วน
    - ผลลัพธ์เป็นการประเมินเบื้องต้น
    - ควรปรึกษาแพทย์เพื่อการวินิจฉัยที่ถูกต้อง
    """)
    
    st.markdown("---")
    st.markdown(
        f"""
        <div style='text-align: center; color: #bdc3c7; font-size: 0.85rem;'>
            Made with ❤️ by <strong>{DEVELOPER_INFO['name']}</strong><br>
            © 2026 {DEVELOPER_INFO['institution']}
        </div>
        """,
        unsafe_allow_html=True
    )

# ==================== Main Content ====================
st.markdown('<p class="main-title">🫀 ระบบทำนายความเสี่ยงโรคหัวใจ</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Heart Disease Risk Prediction using Decision Tree ML Model</p>', unsafe_allow_html=True)

# ==================== Input Form ====================
st.markdown("### 📝 กรุณากรอกข้อมูลสุขภาพ")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 👤 ข้อมูลพื้นฐาน")
    
    age = st.number_input("🎂 อายุ (Age)", min_value=20, max_value=100, value=55, step=1)
    
    sex = st.selectbox(
        "⚧ เพศ (Sex)",
        options=[1, 0],
        format_func=lambda x: "👨 ชาย (Male)" if x == 1 else "👩 หญิง (Female)"
    )
    
    chest_pain_type = st.selectbox(
        "💔 ประเภทอาการเจ็บหน้าอก (Chest Pain Type)",
        options=[1, 2, 3, 4],
        format_func=lambda x: {
            1: "Typical Angina",
            2: "Atypical Angina",
            3: "Non-Anginal Pain",
            4: "Asymptomatic"
        }[x]
    )
    
    resting_bp = st.number_input(
        "💉 ความดันโลหิตขณะพัก (Resting BP) [mm Hg]",
        min_value=80, max_value=200, value=130, step=1
    )
    
    cholesterol = st.number_input(
        "🩸 คอเลสเตอรอล (Cholesterol) [mg/dl]",
        min_value=100, max_value=600, value=220, step=1
    )
    
    fasting_bs = st.selectbox(
        "🍬 น้ำตาลในเลือดขณะอดอาหาร > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "✅ ใช่ (Yes)" if x == 1 else "❌ ไม่ใช่ (No)"
    )

with col2:
    st.markdown("#### 🏥 ผลการตรวจ")
    
    resting_ecg = st.selectbox(
        "📈 ผล ECG ขณะพัก (Resting ECG)",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Normal",
            2: "ST-T Wave Abnormality",
            3: "Left Ventricular Hypertrophy"
        }[x]
    )
    
    max_hr = st.number_input(
        "💓 อัตราการเต้นหัวใจสูงสุด (Max HR) [bpm]",
        min_value=60, max_value=220, value=150, step=1
    )
    
    exercise_angina = st.selectbox(
        "🏃 เจ็บหน้าอกเมื่อออกกำลังกาย (Exercise Angina)",
        options=[0, 1],
        format_func=lambda x: "✅ ใช่ (Yes)" if x == 1 else "❌ ไม่ใช่ (No)"
    )
    
    oldpeak = st.number_input(
        "📉 ST Depression (Oldpeak)",
        min_value=-3.0, max_value=7.0, value=1.0, step=0.1,
        format="%.1f"
    )
    
    st_slope = st.selectbox(
        "📊 Slope ของ ST Segment",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "Upsloping",
            2: "Flat",
            3: "Downsloping"
        }[x]
    )

# ==================== Prediction Button ====================
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])

with predict_col2:
    predict_clicked = st.button(
        "🔮 ทำนายผล (Predict)",
        use_container_width=True,
        type="primary"
    )

# ==================== Prediction Result ====================
if predict_clicked:
    input_data = pd.DataFrame([[
        age, sex, chest_pain_type, resting_bp, cholesterol,
        fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
    ]], columns=feature_names)
    
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    st.markdown("---")
    st.markdown("### 🎯 ผลการทำนาย")
    
    if prediction == 1:
        risk_prob = probability[1] * 100
        st.markdown(
            f'<div class="result-danger">'
            f'⚠️ <strong>มีความเสี่ยงเป็นโรคหัวใจ</strong><br>'
            f'<span style="font-size: 2rem;">ความน่าจะเป็น: {risk_prob:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        safe_prob = probability[0] * 100
        st.markdown(
            f'<div class="result-safe">'
            f'✅ <strong>ไม่พบความเสี่ยงโรคหัวใจ</strong><br>'
            f'<span style="font-size: 2rem;">ความน่าจะเป็น: {safe_prob:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # Gauge Chart
    st.markdown("#### 📊 Probability Distribution")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability[1] * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "ความเสี่ยงโรคหัวใจ (%)", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': '#38ef7d'},
                {'range': [30, 60], 'color': '#f39c12'},
                {'range': [60, 100], 'color': '#eb3349'}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Expander สำหรับดูข้อมูล
    with st.expander("📋 ดูข้อมูลที่คุณกรอก", expanded=False):
        st.dataframe(input_data, use_container_width=True)
    
    with st.expander("🔍 Feature Importance", expanded=False):
        importances = model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        fig_imp = go.Figure(data=[
            go.Bar(
                x=feature_imp_df['Feature'],
                y=feature_imp_df['Importance'],
                marker_color='rgb(102, 126, 234)'
            )
        ])
        fig_imp.update_layout(
            title='Feature Importance',
            xaxis_title='Features',
            yaxis_title='Importance',
            height=400
        )
        st.plotly_chart(fig_imp, use_container_width=True)

# ==================== Footer ====================
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
        <p style='font-size: 0.9rem;'>
            ⚕️ <strong>คำเตือน:</strong> ผลลัพธ์จากการทำนายเป็นเพียงการประเมินเบื้องต้น 
            ไม่สามารถใช้แทนการวินิจฉัยจากแพทย์ได้
        </p>
        <p style='font-size: 0.85rem; margin-top: 1rem;'>
            📧 ติดต่อผู้พัฒนา: 
            <a href="mailto:{DEVELOPER_INFO['email']}" style="color: #3498db;">
                {DEVELOPER_INFO['email']}
            </a>
            &nbsp;|&nbsp;
            🐙 <a href="{DEVELOPER_INFO['github']}" target="_blank" style="color: #3498db;">
                GitHub
            </a>
            &nbsp;|&nbsp;
            💼 <a href="{DEVELOPER_INFO['linkedin']}" target="_blank" style="color: #3498db;">
                LinkedIn
            </a>
        </p>
        <p style='font-size: 0.8rem; margin-top: 0.5rem;'>
            © 2026 {DEVELOPER_INFO['name']} - {DEVELOPER_INFO['institution']}
        </p>
    </div>
    """,
    unsafe_allow_html=True
)