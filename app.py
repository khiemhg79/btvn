

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎓 Academic Warning Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2980b9 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        text-align: center; color: white;
    }
    .main-header h1 { font-size: 2.4rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: .85; margin-top:.5rem; }

    .metric-card {
        background: #f8f9fa; border-left: 4px solid #2980b9;
        padding: 1rem 1.2rem; border-radius: 8px; margin-bottom: .8rem;
    }
    .metric-card h3 { margin: 0; font-size: .85rem; color: #666; }
    .metric-card p  { margin: .2rem 0 0; font-size: 1.6rem; font-weight: 700; color: #1e3a5f; }

    .warning-box {
        background: #fff3cd; border: 2px solid #ffc107;
        border-radius: 10px; padding: 1.2rem; text-align: center;
    }
    .safe-box {
        background: #d4edda; border: 2px solid #28a745;
        border-radius: 10px; padding: 1.2rem; text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg,#1e3a5f,#2980b9);
        color: white; border: none; padding: .6rem 2rem;
        border-radius: 8px; font-weight: 600; font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ───────────────────────────────────────────────────────────────
MODEL_PATH = "academic_warning_model.pkl"

def generate_dataset(n=3000):
    np.random.seed(42)
    data = {}
    data['age']              = np.random.randint(17, 30, n)
    data['gender']           = np.random.choice(['Male','Female'], n)
    data['scholarship']      = np.random.choice([0,1], n, p=[.55,.45])
    data['gpa_semester1']    = np.clip(np.random.normal(2.5,.8,n), 0, 4)
    data['gpa_semester2']    = np.clip(data['gpa_semester1'] + np.random.normal(0,.3,n), 0, 4)
    data['credits_enrolled'] = np.random.randint(9, 21, n)
    data['credits_passed']   = (data['credits_enrolled'] * np.clip(np.random.normal(.85,.15,n),.2,1)).astype(int)
    data['failed_courses']   = np.maximum(0, data['credits_enrolled'] - data['credits_passed'] - np.random.randint(0,3,n))
    data['retake_courses']   = np.random.randint(0, 4, n)
    data['attendance_rate']  = np.clip(np.random.normal(.78,.15,n), .2, 1.0)
    data['late_submissions'] = np.random.randint(0, 10, n)
    data['library_visits']   = np.random.randint(0, 30, n)
    data['tuition_debt']     = np.random.choice([0,1], n, p=[.7,.3])
    data['part_time_job']    = np.random.choice([0,1], n, p=[.6,.4])
    data['counseling_sessions'] = np.random.randint(0, 6, n)
    data['extracurricular']  = np.random.randint(0, 5, n)
    df = pd.DataFrame(data)
    risk = (
        -1.5*df['gpa_semester2'] -1.2*df['gpa_semester1']
        +0.8*df['failed_courses'] +0.6*df['retake_courses']
        -2.0*df['attendance_rate'] +0.4*df['late_submissions']
        +0.5*df['tuition_debt']  -0.3*df['library_visits']/10
        +0.3*df['part_time_job'] + np.random.normal(0,.5,n)
    )
    df['academic_warning'] = (risk > np.percentile(risk, 70)).astype(int)
    return df

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    # train inline if pkl not found
    return train_inline()

def train_inline():
    df = generate_dataset(3000)
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    feature_cols = [c for c in df.columns if c != 'academic_warning']
    X, y = df[feature_cols], df['academic_warning']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=200,learning_rate=.05,max_depth=4,subsample=.8,random_state=42))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    fi = pd.Series(model.named_steps['clf'].feature_importances_, index=feature_cols).sort_values(ascending=False)
    artifacts = {
        'model': model, 'feature_cols': feature_cols,
        'metrics': {
            'accuracy': accuracy_score(y_test,y_pred),
            'f1':       f1_score(y_test,y_pred),
            'roc_auc':  roc_auc_score(y_test,y_proba),
            'cv_mean':  cv_scores.mean(), 'cv_std': cv_scores.std(),
        },
        'feature_importance': fi.to_dict(),
    }
    os.makedirs('model', exist_ok=True)
    with open(MODEL_PATH,'wb') as f:
        pickle.dump(artifacts, f)
    return artifacts

# ─── Load model ────────────────────────────────────────────────────────────
artifacts     = load_or_train_model()
model         = artifacts['model']
feature_cols  = artifacts['feature_cols']
metrics       = artifacts['metrics']
fi_dict       = artifacts['feature_importance']

# ─── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🎓 Academic Warning Prediction</h1>
  <p>Kaggle Competition · Gradient Boosting Classifier · GPA & Behaviour Features</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar navigation ────────────────────────────────────────────────────
page = st.sidebar.radio("📌 Navigation", 
    ["🔮 Predict Single Student", "📊 Model Dashboard", "📁 Batch Prediction"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Model Performance")
st.sidebar.metric("ROC-AUC",  f"{metrics['roc_auc']:.4f}")
st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.4f}")
st.sidebar.metric("F1-Score", f"{metrics.get('f1',0):.4f}")
st.sidebar.metric("CV AUC",   f"{metrics['cv_mean']:.4f} ±{metrics['cv_std']:.4f}")

# ══════════════════════════════════════════════════════════
# PAGE 1 — Single Prediction
# ══════════════════════════════════════════════════════════
if page == "🔮 Predict Single Student":
    st.subheader("🔮 Predict Academic Warning for a Student")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Info**")
        age         = st.slider("Age", 17, 30, 20)
        gender      = st.selectbox("Gender", ["Male", "Female"])
        scholarship = st.selectbox("Scholarship", [0,1], format_func=lambda x: "Yes" if x else "No")
        part_time   = st.selectbox("Part-time Job", [0,1], format_func=lambda x: "Yes" if x else "No")
        tuition_debt= st.selectbox("Tuition Debt", [0,1], format_func=lambda x: "Yes" if x else "No")

    with col2:
        st.markdown("**📚 Academic Records**")
        gpa1 = st.slider("GPA Semester 1", 0.0, 4.0, 2.5, 0.1)
        gpa2 = st.slider("GPA Semester 2", 0.0, 4.0, 2.4, 0.1)
        credits_enrolled = st.slider("Credits Enrolled", 9, 21, 15)
        credits_passed   = st.slider("Credits Passed",   0, 21, 12)
        failed_courses   = st.slider("Failed Courses",   0, 10, 2)
        retake_courses   = st.slider("Retake Courses",   0,  6, 1)

    with col3:
        st.markdown("**🏫 Behaviour & Support**")
        attendance   = st.slider("Attendance Rate (%)", 20, 100, 75) / 100
        late_subs    = st.slider("Late Submissions",  0, 15, 3)
        library      = st.slider("Library Visits / month", 0, 30, 5)
        counseling   = st.slider("Counseling Sessions", 0, 10, 1)
        extra        = st.slider("Extracurricular Activities", 0, 5, 1)

    if st.button("🚀 Predict"):
        gender_enc = 1 if gender == "Male" else 0
        input_data = pd.DataFrame([[
            age, gender_enc, scholarship, gpa1, gpa2,
            credits_enrolled, credits_passed, failed_courses,
            retake_courses, attendance, late_subs, library,
            tuition_debt, part_time, counseling, extra
        ]], columns=feature_cols)

        prob    = model.predict_proba(input_data)[0][1]
        pred    = model.predict(input_data)[0]

        st.markdown("---")
        r1, r2, r3 = st.columns([1,2,1])
        with r2:
            if pred == 1:
                st.markdown(f"""
                <div class="warning-box">
                  <h2>⚠️ Academic Warning</h2>
                  <h1 style="color:#e67e22;font-size:3rem;">{prob:.1%}</h1>
                  <p>Probability of academic warning</p>
                  <p><b>Recommendation:</b> Schedule counseling, review study plan, improve attendance.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="safe-box">
                  <h2>✅ No Warning Detected</h2>
                  <h1 style="color:#27ae60;font-size:3rem;">{1-prob:.1%}</h1>
                  <p>Probability of safe academic standing</p>
                  <p><b>Keep it up!</b> Maintain attendance and GPA.</p>
                </div>
                """, unsafe_allow_html=True)

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Risk Score (%)"},
            gauge={
                'axis': {'range': [0,100]},
                'bar':  {'color': "#e74c3c" if prob > .5 else "#2ecc71"},
                'steps': [
                    {'range': [0,40],  'color': "#d4edda"},
                    {'range': [40,65], 'color': "#fff3cd"},
                    {'range': [65,100],'color': "#f8d7da"},
                ],
                'threshold': {'line':{'color':'red','width':4},'thickness':.75,'value':50}
            }
        ))
        fig.update_layout(height=300, margin=dict(t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — Model Dashboard
# ══════════════════════════════════════════════════════════
elif page == "📊 Model Dashboard":
    st.subheader("📊 Model Performance Dashboard")

    # Metric cards
    c1,c2,c3,c4 = st.columns(4)
    for col, label, val in [
        (c1,"🎯 ROC-AUC",  f"{metrics['roc_auc']:.4f}"),
        (c2,"✅ Accuracy", f"{metrics['accuracy']:.4f}"),
        (c3,"📐 F1-Score", f"{metrics.get('f1',0):.4f}"),
        (c4,"🔁 CV AUC",   f"{metrics['cv_mean']:.4f}"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <h3>{label}</h3><p>{val}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Feature importance
    fi_df = pd.DataFrame.from_dict(fi_dict, orient='index', columns=['importance'])\
              .sort_values('importance', ascending=True).tail(12)

    fig_fi = px.bar(fi_df, x='importance', y=fi_df.index, orientation='h',
                    color='importance', color_continuous_scale='Blues',
                    title="Top Feature Importances (Gradient Boosting)")
    fig_fi.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    # Dataset distribution preview
    st.markdown("---")
    st.markdown("### 📂 Training Data Overview")
    df_preview = generate_dataset(500)
    col_a, col_b = st.columns(2)

    with col_a:
        fig_gpa = px.histogram(df_preview, x='gpa_semester2', color='academic_warning',
                                barmode='overlay', title="GPA Semester 2 Distribution",
                                color_discrete_map={0:'#2ecc71',1:'#e74c3c'},
                                labels={'academic_warning':'Warning'})
        st.plotly_chart(fig_gpa, use_container_width=True)

    with col_b:
        fig_att = px.histogram(df_preview, x='attendance_rate', color='academic_warning',
                                barmode='overlay', title="Attendance Rate Distribution",
                                color_discrete_map={0:'#2ecc71',1:'#e74c3c'},
                                labels={'academic_warning':'Warning'})
        st.plotly_chart(fig_att, use_container_width=True)

    # Model pipeline info
    st.markdown("### 🛠️ Model Pipeline")
    st.code("""
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('clf',     GradientBoostingClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                ))
])
    """, language='python')

# ══════════════════════════════════════════════════════════
# PAGE 3 — Batch Prediction
# ══════════════════════════════════════════════════════════
elif page == "📁 Batch Prediction":
    st.subheader("📁 Batch Prediction from CSV")
    st.info("""
    Upload a CSV file with the following columns:  
    `age, gender, scholarship, gpa_semester1, gpa_semester2, credits_enrolled,
    credits_passed, failed_courses, retake_courses, attendance_rate, late_submissions,
    library_visits, tuition_debt, part_time_job, counseling_sessions, extracurricular`
    """)

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        if 'gender' in df_up.columns:
            df_up['gender'] = df_up['gender'].map({'Male':1,'Female':0}).fillna(0).astype(int)

        try:
            X_up = df_up[feature_cols]
            proba = model.predict_proba(X_up)[:,1]
            preds = model.predict(X_up)
            df_up['risk_probability'] = proba
            df_up['prediction']       = preds
            df_up['status'] = df_up['prediction'].map({1:'⚠️ Warning',0:'✅ Safe'})

            st.success(f"✅ Predicted {len(df_up)} students")
            st.dataframe(df_up[['risk_probability','prediction','status']].assign(
                risk_probability=df_up['risk_probability'].map('{:.2%}'.format)
            ).join(df_up.drop(columns=['risk_probability','prediction','status'])),
            use_container_width=True)

            # Summary
            n_warn = preds.sum()
            st.markdown(f"""
            **Summary:** {n_warn} / {len(preds)} students flagged for academic warning 
            ({n_warn/len(preds):.1%})
            """)

            csv_out = df_up.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Column mismatch: {e}")
    else:
        # Show sample
        st.markdown("#### 👉 Sample Input Preview")
        sample = generate_dataset(5).drop(columns=['academic_warning'])
        st.dataframe(sample, use_container_width=True)
        csv_sample = sample.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Sample CSV", csv_sample,
                           "quybeo.csv", "text/csv")

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#999;'>🎓 Academic Warning Predictor · "
    "Kaggle Competition · Built with Streamlit & Scikit-learn</p>",
    unsafe_allow_html=True
)
