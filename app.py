import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from data_generator import generate_synthetic_data
from model import (
    FEATURE_COLS,
    build_pipeline,
    evaluate,
    get_feature_importance,
    get_shap_values,
    predict_patient,
    preprocess_and_split,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cardiac Risk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border-left: 4px solid #4f8ef7;
    }
    .risk-high   { border-left-color: #e74c3c !important; }
    .risk-medium { border-left-color: #f39c12 !important; }
    .risk-low    { border-left-color: #2ecc71 !important; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 8px;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ── Session-state caching ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return generate_synthetic_data(1000)


@st.cache_resource
def train_models(model_type: str):
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    pipeline = build_pipeline(model_type)
    pipeline.fit(X_train, y_train)
    metrics  = evaluate(pipeline, X_test, y_test)
    fi       = get_feature_importance(pipeline, model_type)
    return pipeline, metrics, fi, X_test, y_test


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/ios/50/heart.png", width=72)
    st.title("Cardiac Risk AI")
    st.caption("Educational prototype · Not for clinical use")
    st.divider()

    model_type = st.selectbox(
        "Model",
        ["Random Forest", "Logistic Regression"],
        help="Switch between models to compare performance.",
    )
    st.divider()

    st.markdown("### 🧑‍⚕️ Patient Input")
    age         = st.slider("Age",           18, 90, 55)
    gender      = st.selectbox("Gender",     ["Female (0)", "Male (1)"])
    bmi         = st.slider("BMI",           15.0, 50.0, 27.5, 0.1)
    sbp         = st.slider("Systolic BP",   80,  220, 135)
    dbp         = st.slider("Diastolic BP",  50,  140,  85)
    hr          = st.slider("Heart Rate",    40,  130,  75)
    med_history = st.selectbox("Medical History", ["None (0)", "Existing (1)"])

    predict_btn = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

gender_val  = 1 if "Male" in gender else 0
med_hist_val = 1 if "Existing" in med_history else 0

patient = {
    "Age": age, "Gender": gender_val, "BMI": bmi,
    "SBP": sbp, "DBP": dbp, "HeartRate": hr,
    "MedHistory": med_hist_val,
}


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title("🫀 Cardiovascular Risk Decision-Support System")
st.caption(
    "An educational AI prototype demonstrating prediction, evaluation, "
    "and explainability in a clinical-AI workflow."
)

pipeline, metrics, fi, X_test, y_test = train_models(model_type)

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Evaluation",
    "🔬 Explainability",
    "🧑‍⚕️ Patient Prediction",
    "📋 Dataset Overview",
])


# ── Tab 1 · Model Evaluation ──────────────────────────────────────────────────
with tab1:
    st.subheader(f"Model Performance — {model_type}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall",    f"{metrics['recall']:.2%}")
    c4.metric("ROC-AUC",   f"{metrics['roc_auc']:.3f}")

    col_left, col_right = st.columns(2)

    # ROC Curve
    with col_left:
        st.markdown("**ROC Curve**")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=metrics["fpr"], y=metrics["tpr"],
            mode="lines",
            name=f"AUC = {metrics['roc_auc']:.3f}",
            line=dict(color="#4f8ef7", width=2.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash"),
        ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Confusion Matrix
    with col_right:
        st.markdown("**Confusion Matrix**")
        cm = metrics["conf_matrix"]
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["No Risk", "At Risk"],
            y=["No Risk", "At Risk"],
        )
        fig_cm.update_layout(height=350, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_cm, use_container_width=True)

    # Model comparison table
    st.divider()
    st.markdown("**Quick Model Comparison**")

    _, metrics_lr, _ , _, _ = train_models("Logistic Regression")
    _, metrics_rf, _ , _, _ = train_models("Random Forest")

    comparison = pd.DataFrame({
        "Metric":              ["Accuracy", "Precision", "Recall", "ROC-AUC"],
        "Logistic Regression": [
            f"{metrics_lr['accuracy']:.2%}", f"{metrics_lr['precision']:.2%}",
            f"{metrics_lr['recall']:.2%}",   f"{metrics_lr['roc_auc']:.3f}",
        ],
        "Random Forest":       [
            f"{metrics_rf['accuracy']:.2%}", f"{metrics_rf['precision']:.2%}",
            f"{metrics_rf['recall']:.2%}",   f"{metrics_rf['roc_auc']:.3f}",
        ],
    })
    st.dataframe(comparison, hide_index=True, use_container_width=True)


# ── Tab 2 · Explainability ────────────────────────────────────────────────────
with tab2:
    st.subheader("Explainability — Feature Importance & SHAP")

    col_fi, col_shap = st.columns(2)

    with col_fi:
        st.markdown("**Global Feature Importance**")
        fi_df = fi.reset_index()
        fi_df.columns = ["Feature", "Importance"]
        fig_fi = px.bar(
            fi_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(
            height=380,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    with col_shap:
        st.markdown("**SHAP Summary Plot**")
        with st.spinner("Computing SHAP values…"):
            _, shap_vals, X_proc = get_shap_values(pipeline, X_test, model_type)
        fig_shap, ax = plt.subplots(figsize=(6, 4.5))
        shap.summary_plot(
            shap_vals,
            X_proc,
            feature_names=FEATURE_COLS,
            plot_type="dot",
            show=False,
            color_bar=True,
        )
        plt.tight_layout()
        st.pyplot(fig_shap, use_container_width=True)
        plt.close()

    st.info(
        "**How to read SHAP:** Each dot is one patient. "
        "Position on the x-axis shows whether a feature pushed the prediction "
        "toward higher (right) or lower (left) cardiac risk. "
        "Color indicates the feature value (red = high, blue = low)."
    )


# ── Tab 3 · Patient Prediction ────────────────────────────────────────────────
with tab3:
    st.subheader("🧑‍⚕️ Individual Patient Risk Assessment")
    st.caption("Adjust inputs in the sidebar, then click **Predict Risk**.")

    if predict_btn or True:  # always show current input
        result = predict_patient(pipeline, patient)
        prob   = result["probability"]

        st.divider()
        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown("#### Risk Classification")
            st.markdown(f"## {result['risk_level']}")

        with r2:
            st.markdown("#### Cardiac Risk Probability")
            st.markdown(f"## {prob:.1%}")

        with r3:
            st.markdown("#### Model Confidence")
            confidence = max(prob, 1 - prob)
            st.progress(confidence)
            st.caption(f"{confidence:.1%} confident")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#4f8ef7"},
                "steps": [
                    {"range": [0,  35], "color": "#2ecc71"},
                    {"range": [35, 65], "color": "#f39c12"},
                    {"range": [65, 100], "color": "#e74c3c"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": prob * 100,
                },
            },
            title={"text": "Cardiac Risk Probability"},
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Top contributing features for this patient
        st.divider()
        st.markdown("**Top Contributing Clinical Factors**")

        model_inner = pipeline.named_steps["model"]
        imp_scores  = (
            model_inner.feature_importances_
            if model_type == "Random Forest"
            else np.abs(model_inner.coef_[0])
        )
        patient_df = pd.DataFrame([patient])
        contrib = pd.DataFrame({
            "Feature":    FEATURE_COLS,
            "Your Value": [patient[f] for f in FEATURE_COLS],
            "Importance": imp_scores,
        }).sort_values("Importance", ascending=False).head(5)

        contrib["Importance (%)"] = (
            contrib["Importance"] / contrib["Importance"].sum() * 100
        ).round(1)

        st.dataframe(
            contrib[["Feature", "Your Value", "Importance (%)"]],
            hide_index=True,
            use_container_width=True,
        )

        st.warning(
            "⚠️ **Clinical Disclaimer:** This system is an educational prototype only. "
            "It should never be used to guide actual clinical decisions. "
            "Always consult a qualified healthcare professional."
        )


# ── Tab 4 · Dataset Overview ──────────────────────────────────────────────────
with tab4:
    st.subheader("📋 Synthetic Dataset Overview")
    df = load_data()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Patients",  len(df))
    c2.metric("At-Risk Patients", int(df["CardiacRisk"].sum()))
    c3.metric("At-Risk Rate",    f"{df['CardiacRisk'].mean():.1%}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Age Distribution by Risk**")
        fig_age = px.histogram(
            df, x="Age", color="CardiacRisk",
            barmode="overlay",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={"CardiacRisk": "Cardiac Risk"},
            opacity=0.75,
        )
        fig_age.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_age, use_container_width=True)

    with col_b:
        st.markdown("**BMI vs Systolic BP (colored by Risk)**")
        fig_scatter = px.scatter(
            df.dropna(), x="BMI", y="SBP",
            color=df.dropna()["CardiacRisk"].astype(str),
            color_discrete_map={"0": "#2ecc71", "1": "#e74c3c"},
            opacity=0.5,
            labels={"color": "Risk"},
        )
        fig_scatter.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("**Summary Statistics**")
    st.dataframe(df[FEATURE_COLS].describe().round(2), use_container_width=True)

    st.markdown("**Sample Records**")
    st.dataframe(df.head(20), use_container_width=True)
