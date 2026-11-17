import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# PAGE CONFIG
st.set_page_config(
    page_title="CampusPath | Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

# CUSTOM CSS DESIGN
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

.stApp {
    background:
    radial-gradient(circle at 8% 4%,#ffe9df 0,transparent 25%),
    radial-gradient(circle at 94% 8%,#e4eaff 0,transparent 26%),
    linear-gradient(135deg,#f8f7ff,#f2fbff 55%,#fff8f2);
    color:#29334b;
}

.block-container {
    max-width:1250px;
    padding-top:1.3rem;
}

.hero {
    background:linear-gradient(115deg,#51439f,#7868cb 52%,#6b9cde);
    color:white;
    padding:27px 32px;
    border-radius:25px;
    box-shadow:0 14px 35px #5549a533;
    margin-bottom:20px;
}

.hero h1 {
    font-family:Manrope;
    font-size:2.1rem;
    font-weight:800;
    margin:0;
}

.hero p {
    margin:7px 0 0;
    opacity:.92;
}

.tag {
    display:inline-block;
    background:#ffffff25;
    border:1px solid #ffffff55;
    border-radius:30px;
    padding:5px 12px;
    font-size:.78rem;
    margin-bottom:10px;
}

.section {
    font-family:Manrope;
    font-weight:800;
    color:#453b82;
    font-size:1.15rem;
    margin:8px 0 12px;
}

.panel {
    background:#ffffffdf;
    border:1px solid #e8e5f7;
    border-radius:20px;
    padding:20px 22px;
    box-shadow:0 7px 24px #434b780d;
}

.metric {
    background:linear-gradient(135deg,#fff,#f5f2ff);
    border:1px solid #e7e1fb;
    border-radius:17px;
    padding:15px 17px;
    min-height:98px;
}

.label {
    font-size:.76rem;
    color:#777995;
    font-weight:800;
    text-transform:uppercase;
}

.value {
    font-family:Manrope;
    font-size:1.65rem;
    font-weight:800;
    color:#403879;
    margin-top:4px;
}

.hint,.note {
    font-size:.78rem;
    color:#888aa3;
}

.result {
    border-radius:18px;
    padding:22px;
    text-align:center;
    border:1px solid #dcd8f6;
    background:linear-gradient(135deg,#f5f2ff,#eff8ff);
}

.result h2 {
    font-family:Manrope;
    margin:4px 0;
    color:#403879;
}

div.stButton>button {
    background:linear-gradient(100deg,#6254b8,#668fda);
    color:white;
    border:0;
    border-radius:12px;
    padding:.65rem 1rem;
    font-weight:800;
}

div.stButton>button:hover {
    color:white;
    border:0;
}

footer {
    visibility:hidden;
}
</style>
""", unsafe_allow_html=True)


# LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("student_placement_synthetic.pkl")


# SESSION HISTORY
if "history" not in st.session_state:
    st.session_state.history = []


# HEADER
st.markdown("""
<div class="hero">
    <div class="tag">
        🎓 SMART CAMPUS • AI PLACEMENT INSIGHTS
    </div>

    <h1>CampusPath</h1>

    <p>
    Student Placement Prediction Dashboard —
    turn student skills into actionable insights.
    </p>
</div>
""", unsafe_allow_html=True)


# DASHBOARD METRICS
total = len(st.session_state.history)

placed = sum(
    x["Prediction"] == "Placed"
    for x in st.session_state.history
)

not_placed = total - placed

rate = placed / total * 100 if total else 0


cols = st.columns(4)

for c, title, val, hint in zip(
    cols,
    [
        "Predictions made",
        "Placed predictions",
        "Not placed",
        "Predicted placement rate"
    ],
    [
        total,
        placed,
        not_placed,
        f"{rate:.1f}%"
    ],
    [
        "This session",
        "Model output",
        "Model output",
        "This session"
    ]
):
    c.markdown(
        f"""
        <div class="metric">
            <div class="label">{title}</div>
            <div class="value">{val}</div>
            <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# LOAD TRAINED MODEL
try:
    model = load_model()

except Exception as e:

    st.error(
        "Model file load nahi ho rahi. "
        "student_placement_synthetic.pkl ko isi folder mein rakhein."
    )

    st.code(str(e))
    st.stop()


st.write("")

left, right = st.columns([1, 1.15], gap="large")


# LEFT SIDE - STUDENT INPUT
with left:

    st.markdown(
        '<div class="section">✦ Student Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    st.write(
        "Student ki academic aur skill details enter karein."
    )

    cgpa = st.number_input(
        "📘 CGPA",
        0.0,
        10.0,
        7.0,
        0.1
    )

    coding = st.number_input(
        "💻 Coding Skills",
        0.0,
        10.0,
        6.0,
        0.1
    )

    communication = st.number_input(
        "🗣️ Communication Skills",
        0.0,
        10.0,
        6.0,
        0.1
    )

    predict = st.button(
        "✨ Predict Placement",
        use_container_width=True
    )

    st.markdown(
        '<p class="note">All scores are measured on a 0–10 scale.</p>',
        unsafe_allow_html=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    st.markdown(
        '<div class="panel">**💡 Skill snapshot**',
        unsafe_allow_html=True
    )

    st.bar_chart(
        pd.DataFrame(
            {"Score": [cgpa, coding, communication]},
            index=["CGPA", "Coding", "Communication"]
        ),
        height=180
    )

    st.markdown('</div>', unsafe_allow_html=True)


# RIGHT SIDE - PREDICTION
with right:

    st.markdown(
        '<div class="section">✧ Prediction Insights</div>',
        unsafe_allow_html=True
    )

    if predict:

        row = pd.DataFrame([{
            "cgpa": cgpa,
            "coding_skills": coding,
            "communication_skills": communication
        }])

        try:

            cls = int(model.predict(row)[0])

            probs = model.predict_proba(row)[0]

            classes = list(model.classes_)

            p1 = (
                float(probs[classes.index(1)])
                if 1 in classes else 0.0
            )

            p0 = (
                float(probs[classes.index(0)])
                if 0 in classes else 0.0
            )

            result = "Placed" if cls == 1 else "Not Placed"

            # SAVE HISTORY
            st.session_state.history.insert(0, {
                "Date & Time": datetime.now().strftime(
                    "%d-%m-%Y %I:%M %p"
                ),
                "CGPA": cgpa,
                "Coding Skills": coding,
                "Communication Skills": communication,
                "Prediction": result,
                "Placed Probability (%)": round(p1 * 100, 2)
            })

            # RESULT CARD
            if cls == 1:

                st.markdown("""
                <div class="result"
                style="background:linear-gradient(135deg,#e9fff5,#effaff);
                border-color:#bcebd8">

                    <div style="font-size:2.3rem">🎉</div>

                    <h2 style="color:#16845c">
                    PLACEMENT LIKELY
                    </h2>

                    <p style="color:#4c806e">
                    Model predicts this student as <b>Placed</b>.
                    </p>

                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result"
                style="background:linear-gradient(135deg,#fff1ed,#fff8ef);
                border-color:#f5d4c8">

                    <div style="font-size:2.3rem">🌱</div>

                    <h2 style="color:#bd634b">
                    PLACEMENT NOT LIKELY
                    </h2>

                    <p style="color:#98766c">
                    Model predicts this student as <b>Not Placed</b>.
                    </p>

                </div>
                """, unsafe_allow_html=True)


            # PROBABILITY CARDS
            a, b = st.columns(2)

            a.markdown(
                f"""
                <div class="metric">
                    <div class="label">Placed probability</div>
                    <div class="value">{p1*100:.1f}%</div>
                    <div class="hint">Model probability</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            b.markdown(
                f"""
                <div class="metric">
                    <div class="label">Not placed probability</div>
                    <div class="value">{p0*100:.1f}%</div>
                    <div class="hint">Model probability</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # PROBABILITY CHART
            st.write("**Probability breakdown**")

            st.bar_chart(
                pd.DataFrame(
                    {"Probability": [p1, p0]},
                    index=["Placed", "Not Placed"]
                ),
                height=175
            )

            st.info(
                "Prediction model ke output par based hai. "
                "Ye guaranteed job/placement result nahi hai."
            )

        except Exception as e:

            st.error(
                "Prediction error. Confirm karein ki PKL isi dataset "
                "ke features (cgpa, coding_skills, communication_skills) "
                "se train hua hai."
            )

            st.code(str(e))

    else:

        st.markdown("""
        <div class="panel"
        style="min-height:260px;text-align:center;padding-top:65px">

            <div style="font-size:3rem">🔮</div>

            <h2 style="font-family:Manrope;color:#494080">
            Your prediction appears here
            </h2>

            <p style="color:#85859d">
            Enter student details and click
            <b>Predict Placement</b>.
            </p>

        </div>
        """, unsafe_allow_html=True)


    st.write("")

    st.markdown("""
    <div class="panel">

        **🧭 How to read the result**

        <br>• Placed = model predicts class 1.

        <br>• Not Placed = model predicts class 0.

        <br>• Probability is the model-estimated likelihood for each class.

    </div>
    """, unsafe_allow_html=True)


# PREDICTION HISTORY
st.write("")

st.markdown(
    '<div class="section">▤ Recent Prediction History</div>',
    unsafe_allow_html=True
)

if st.session_state.history:

    hist = pd.DataFrame(st.session_state.history)

    st.dataframe(
        hist,
        use_container_width=True,
        hide_index=True
    )

    c1, c2 = st.columns([1, 5])

    c1.download_button(
        "⬇️ Download CSV",
        hist.to_csv(index=False).encode("utf-8"),
        "student_placement_predictions.csv",
        "text/csv",
        use_container_width=True
    )

    if c2.button("🗑️ Clear history"):

        st.session_state.history = []

        st.rerun()

else:

    st.markdown("""
    <div class="panel">
        <span class="note">
        No predictions yet. Your first result will appear here.
        </span>
    </div>
    """, unsafe_allow_html=True)


# FOOTER
st.markdown("""
<div style="text-align:center;color:#9695ad;
font-size:.78rem;padding:22px 0 4px">

CampusPath • Student Placement Prediction • Python + Streamlit

</div>
""", unsafe_allow_html=True)
