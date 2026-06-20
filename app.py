import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Insurance Claim Predictor", page_icon="🏥", layout="wide")

@st.cache_resource
def load_artifacts():
    return (
        joblib.load('best_model.pkl'),
        joblib.load('scaler.pkl'),
        joblib.load('label_encoder_gender.pkl'),
        joblib.load('label_encoder_diabetic.pkl'),
        joblib.load('label_encoder_smoker.pkl'),
    )

@st.cache_data
def load_data():
    return pd.read_csv('data/insurance.csv')

model, scaler, le_gender, le_diabetic, le_smoker = load_artifacts()
df = load_data()

FEATURES = ["Age", "Gender", "BMI", "Blood Pressure", "Diabetic", "Children", "Smoker"]

METRICS = {
    "Linear Regression": (0.657, 5414.05, 7105.46),
    "Polynomial (deg=3)": (0.752, 4527.03, 6044.07),
    "Random Forest": (0.799, 3981.90, 5433.84),
    "SVR": (0.475, 5908.14, 8792.01),
    "XGBoost": (0.802, 4018.31, 5396.91),
}
CLAIM_STATS = {"count": 1332, "mean": 13325.25, "p50": 9412.97, "max": 63770.43}
GENDER_SMOKER_CLAIM = {
    ("female", "No"): 8762.30, ("female", "Yes"): 30679.00,
    ("male", "No"): 8169.25, ("male", "Yes"): 33042.01,
}
REGION_STATS = pd.DataFrame({
    "region": ["northeast", "northwest", "southeast", "southwest"],
    "smoker_rate": [29.00, 16.81, 20.59, 18.47],
    "mean_claim": [16889.04, 11794.22, 13085.50, 12723.13],
})
BMI_COUNTS = {"Underweight": 21, "Normal": 222, "Overweight": 387, "Obese": 702}

DISPLAY_MAPS = {
    "gender": {0: "Female", 1: "Male", "female": "Female", "male": "Male"},
    "diabetic": {0: "No", 1: "Yes", "no": "No", "yes": "Yes"},
    "smoker": {0: "No", 1: "Yes", "no": "No", "yes": "Yes"},
}

def humanize(value, kind):
    key = value.lower() if isinstance(value, str) else value
    return DISPLAY_MAPS.get(kind, {}).get(key, str(value).capitalize())

def bmi_category(bmi):
    return "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"

def encode_and_scale(age, gender, bmi, bp, diabetic, children, smoker):
    row = [age, le_gender.transform([gender])[0], bmi, bp, le_diabetic.transform([diabetic])[0], children, le_smoker.transform([smoker])[0]]
    arr = np.array([row], dtype=float)
    arr[:, [0, 2, 3, 5]] = scaler.transform(arr[:, [0, 2, 3, 5]])
    return arr

PAGES = ["🔮 Predict", "📊 Insights", "🧠 Model Performance"]
with st.sidebar:
    st.title("🏥 Claim Predictor")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.metric("Model R²", f"{METRICS['XGBoost'][0]:.2f}")
    st.metric("Avg. Error (MAE)", f"${METRICS['XGBoost'][1]:,.0f}")
    st.caption("Streamlit • scikit-learn • XGBoost")

st.title("Insurance Claim Cost Predictor")

if page == PAGES[0]:
    col_in, col_out = st.columns([1, 1.3])
    with col_in:
        age = st.slider("Age", 18, 100, 30)
        bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)
        bp = st.slider("Blood Pressure", 80, 200, 120)
        children = st.select_slider("Children", options=[0, 1, 2, 3, 4, 5])
        gender = st.selectbox("Gender", options=list(le_gender.classes_), format_func=lambda v: humanize(v, "gender"))
        diabetic = st.selectbox("Diabetic", options=list(le_diabetic.classes_), format_func=lambda v: humanize(v, "diabetic"))
        smoker = st.selectbox("Smoker", options=list(le_smoker.classes_), format_func=lambda v: humanize(v, "smoker"))
        clicked = st.button("Predict Claim", type="primary", use_container_width=True)

    with col_out:
        if clicked:
            gender_l, diabetic_l, smoker_l = humanize(gender, "gender"), humanize(diabetic, "diabetic"), humanize(smoker, "smoker")
            pred = max(float(model.predict(encode_and_scale(age, gender, bmi, bp, diabetic, children, smoker))[0]), 0)
            alt_smoker = next((c for c in le_smoker.classes_ if c != smoker), smoker)
            alt_pred = max(float(model.predict(encode_and_scale(age, gender, bmi, bp, diabetic, children, alt_smoker))[0]), 0)
            risk = "High" if smoker_l == "Yes" or diabetic_l == "Yes" or bmi_category(bmi) == "Obese" else "Low"
            st.session_state.result = dict(
                pred=pred, alt_pred=alt_pred, risk=risk, bmi_cat=bmi_category(bmi),
                gender_l=gender_l, smoker_l=smoker_l,
                avg=GENDER_SMOKER_CLAIM.get((gender_l.lower(), smoker_l), CLAIM_STATS["mean"])
            )

        r = st.session_state.get("result")
        if r:
            st.metric("Estimated Claim Amount", f"${r['pred']:,.2f}")
            c1, c2 = st.columns(2)
            c1.info(f"BMI Category: **{r['bmi_cat']}**")
            (c2.error if r["risk"] == "High" else c2.success)(f"Risk Profile: **{r['risk']}**")

            fig = go.Figure(data=[
                go.Bar(name="Your Prediction", x=[""], y=[r["pred"]], marker_color="#1f77b4"),
                go.Bar(name=f"Avg ({r['gender_l']}, smoker={r['smoker_l']})", x=[""], y=[r["avg"]], marker_color="#ff7f0e"),
                go.Bar(name="Dataset Average", x=[""], y=[CLAIM_STATS["mean"]], marker_color="#7f7f7f"),
            ])
            fig.update_layout(title="How Your Prediction Compares", barmode="group", height=350, yaxis_title="Claim ($)")
            st.plotly_chart(fig, use_container_width=True)

            diff = abs(r["pred"] - r["alt_pred"])
            if r["smoker_l"] == "No":
                st.caption(f"If this profile smoked, the predicted claim would rise by about ${diff:,.0f}.")
            else:
                st.caption(f"If this profile quit smoking, the predicted claim would drop by about ${diff:,.0f}.")
        else:
            st.info("Set the patient details and click **Predict Claim**.")

elif page == PAGES[1]:
    m1, m2, m3 = st.columns(3)
    m1.metric("Records", f"{CLAIM_STATS['count']:,}")
    m2.metric("Average Claim", f"${CLAIM_STATS['mean']:,.2f}")
    m3.metric("Max Claim", f"${CLAIM_STATS['max']:,.2f}")

    c1, c2 = st.columns(2)
    with c1:
        gs_df = pd.DataFrame([{"Gender": g.capitalize(), "Smoker": s, "Avg Claim": v} for (g, s), v in GENDER_SMOKER_CLAIM.items()])
        st.plotly_chart(px.bar(gs_df, x="Gender", y="Avg Claim", color="Smoker", barmode="group", title="Avg Claim by Gender & Smoking"), use_container_width=True)
    with c2:
        bmi_df = pd.DataFrame({"Category": BMI_COUNTS.keys(), "Count": BMI_COUNTS.values()})
        st.plotly_chart(px.pie(bmi_df, names="Category", values="Count", hole=0.4, title="BMI Category Distribution"), use_container_width=True)

    fig = go.Figure()
    fig.add_bar(x=REGION_STATS["region"], y=REGION_STATS["smoker_rate"], name="Smoker Rate (%)", marker_color="#9467bd")
    fig.add_trace(go.Scatter(x=REGION_STATS["region"], y=REGION_STATS["mean_claim"], name="Avg Claim ($)", mode="lines+markers", yaxis="y2", marker_color="#d62728"))
    fig.update_layout(title="Smoker Rate & Avg Claim by Region", yaxis=dict(title="Smoker Rate (%)"), yaxis2=dict(title="Avg Claim ($)", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Smoking status is the strongest driver of claim amount — smokers average roughly 3-4x higher claims than non-smokers.")

else:
    metrics_df = pd.DataFrame(METRICS, index=["R2", "MAE", "RMSE"]).T.reset_index().rename(columns={"index": "Model"}).sort_values("R2", ascending=False)
    st.dataframe(metrics_df.style.format({"R2": "{:.3f}", "MAE": "${:,.2f}", "RMSE": "${:,.2f}"}), use_container_width=True, hide_index=True)

    fig = px.bar(metrics_df, x="Model", y="R2", color="Model", text="R2", title="R² Score by Model")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.success("XGBoost was selected as the production model (R² = 0.80) after comparing 5 algorithms with GridSearchCV tuning.")

    if hasattr(model, "feature_importances_"):
        imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_}).sort_values("Importance")
        st.plotly_chart(px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Feature Importance"), use_container_width=True)