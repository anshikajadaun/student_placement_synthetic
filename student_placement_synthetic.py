import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# =====================================================================
# 1. MODEL (load + predict)
# =====================================================================
FEATURES = ["cgpa", "coding_skills", "communication_skills"]
MODEL_PATH = Path(__file__).resolve().parent / "student_placement_synthetic.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def predict(cgpa, coding, comm):
    """Returns (label 0/1, placement probability in %)."""
    model = load_model()
    X = pd.DataFrame([[cgpa, coding, comm]], columns=FEATURES)
    prob = float(model.predict_proba(X)[0][list(model.classes_).index(1)]) * 100
    return int(model.predict(X)[0]), prob


def advice(cgpa, coding, comm):
    scores = {"CGPA": cgpa, "Coding": coding, "Communication": comm}
    weakest = min(scores, key=scores.get)
    if min(scores.values()) >= 8:
        return "Strong profile across the board. Keep it up."
    return f"Biggest room to grow: {weakest} ({scores[weakest]:.1f}/10)."


# =====================================================================
# 2. DESIGN (CSS + HTML templates)
# =====================================================================
LIME, ORCHID, EMBER = "#C8FF2E", "#B26BFF", "#FF7A59"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap');
@property --angle{syntax:'<angle>';initial-value:0deg;inherits:false}
@property --n{syntax:'<integer>';initial-value:0;inherits:false}
:root{--bg:#07070D;--lime:#C8FF2E;--orchid:#B26BFF;--ember:#FF7A59;--ink:#ECEBFF;--mute:#8E8CA8}
html,body,.stApp{background:var(--bg)!important;color:var(--ink);font-family:'Space Grotesk',sans-serif}
.stApp::before,.stApp::after{content:"";position:fixed;width:46vmax;height:46vmax;border-radius:50%;filter:blur(90px);z-index:0;pointer-events:none;animation:drift 18s ease-in-out infinite alternate}
.stApp::before{background:radial-gradient(circle,var(--orchid),transparent 65%);top:-18vmax;left:-12vmax;opacity:.28}
.stApp::after{background:radial-gradient(circle,var(--lime),transparent 65%);bottom:-22vmax;right:-14vmax;opacity:.18;animation-delay:-9s}
@keyframes drift{to{transform:translate(8vmax,6vmax) scale(1.15)}}
header[data-testid="stHeader"]{background:transparent}
.block-container{position:relative;z-index:1;padding-top:2rem;max-width:1200px}
.stApp p,.stApp li,.stApp h1,.stApp h2,.stApp h3,.stApp label{color:var(--ink)}
.hero{position:relative;overflow:hidden;padding:2.2rem 2.4rem;border-radius:26px;border:1px solid rgba(200,255,46,.18);background:linear-gradient(135deg,rgba(178,107,255,.16),rgba(200,255,46,.05))}
.hero::after{content:"";position:absolute;top:-50%;left:-30%;width:25%;height:200%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.12),transparent);transform:rotate(18deg);animation:sweep 5s ease-in-out infinite}
@keyframes sweep{0%{left:-30%}60%,100%{left:130%}}
.hero h1{font-size:3rem;margin:.4rem 0;background:linear-gradient(90deg,var(--lime),var(--orchid),var(--ember),var(--lime));background-size:300% 100%;-webkit-background-clip:text;background-clip:text;color:transparent!important;animation:flow 8s linear infinite}
.hero p{color:var(--mute)!important;margin:0}
@keyframes flow{to{background-position:300% 0}}
.badge{display:inline-block;padding:.3rem .8rem;border-radius:99px;border:1px solid rgba(178,107,255,.5);color:var(--orchid);font-size:.72rem;letter-spacing:.12em}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}
.kpi{padding:16px 18px;border-radius:18px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);transition:.3s}
.kpi:hover{transform:translateY(-4px);box-shadow:0 0 28px rgba(200,255,46,.18);border-color:rgba(200,255,46,.5)}
.kpi small{color:var(--mute);letter-spacing:.1em;font-size:.68rem;text-transform:uppercase}
.kpi b{display:block;font-size:2rem;color:var(--lime);counter-reset:n var(--n);animation:count 1.1s ease-out forwards}
.kpi b::after{content:counter(n) attr(data-suffix)}
@keyframes count{from{--n:0}}
.st-key-profile,.st-key-result,.st-key-howto{border:1.5px solid transparent!important;border-radius:24px!important;padding:1.3rem!important;background:linear-gradient(#0C0C17,#0C0C17) padding-box,conic-gradient(from var(--angle),#C8FF2E,#B26BFF,#FF7A59,#C8FF2E) border-box!important;animation:spin 7s linear infinite;box-shadow:0 0 40px rgba(178,107,255,.12)}
@keyframes spin{to{--angle:360deg}}
.sec{font-size:1.25rem;font-weight:700;margin:.2rem 0 .3rem}
.sec i{color:var(--lime);font-style:normal}
.hint{color:var(--mute)!important;font-size:.85rem;margin-bottom:.6rem}
div[data-testid="stNumberInput"] label p{color:var(--mute)!important;letter-spacing:.08em;text-transform:uppercase;font-size:.75rem!important}
div[data-testid="stNumberInput"] div[data-baseweb="input"]{background:#12121F!important;border-radius:14px!important;border:1px solid rgba(255,255,255,.1)!important;transition:.3s}
div[data-testid="stNumberInput"] div[data-baseweb="input"]:hover{border-color:var(--orchid)!important}
div[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within{border-color:var(--lime)!important;transform:translateY(-2px);animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(200,255,46,.45)}50%{box-shadow:0 0 22px 4px rgba(200,255,46,.25)}}
div[data-testid="stNumberInput"] input{color:var(--ink)!important;font-family:'JetBrains Mono',monospace!important;font-size:1.1rem!important}
div[data-testid="stNumberInput"] button{background:transparent!important;color:var(--lime)!important}
.stButton>button{width:100%;border:0;border-radius:16px;padding:.85rem;font-weight:700;background:linear-gradient(110deg,#C8FF2E,#E5FF8A,#B26BFF,#C8FF2E);background-size:300% 100%;animation:flow 6s linear infinite;box-shadow:0 0 24px rgba(200,255,46,.35);transition:.25s}
.stButton>button p{color:#07070D!important}
.stButton>button:hover{transform:translateY(-3px) scale(1.01);box-shadow:0 0 42px rgba(178,107,255,.55);border:0}
.bar{margin:12px 0}
.bar span{display:flex;justify-content:space-between;color:var(--mute);font-size:.8rem;margin-bottom:5px}
.track{height:10px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden}
.fill{height:100%;width:var(--w);border-radius:99px;background:linear-gradient(90deg,var(--orchid),var(--lime));box-shadow:0 0 14px rgba(200,255,46,.6);animation:grow 1.2s cubic-bezier(.2,.8,.2,1)}
@keyframes grow{from{width:0}}
.idle{text-align:center;padding:3rem 1rem;color:var(--mute)}
.orb{width:120px;height:120px;margin:0 auto 1rem;border-radius:50%;background:radial-gradient(circle at 35% 30%,#E5FF8A,#B26BFF 60%,#07070D);animation:breathe 3s ease-in-out infinite;box-shadow:0 0 50px rgba(178,107,255,.6)}
@keyframes breathe{50%{transform:scale(1.08);box-shadow:0 0 80px rgba(200,255,46,.5)}}
"""


def html(s):
    """Render HTML safely: strips indentation + blank lines so Markdown
    never turns the HTML into a code block."""
    cleaned = "\n".join(line.strip() for line in s.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def hero():
    return """
    <div class="hero">
        <span class="badge">SMART CAMPUS · AI PLACEMENT INSIGHTS</span>
        <h1>CampusPath</h1>
        <p>Student Placement Prediction Dashboard: turn student skills into actionable insights.</p>
    </div>
    """


def kpis(made, placed, not_placed, rate):
    def tile(label, n, suffix=""):
        return (f'<div class="kpi"><small>{label}</small>'
                f'<b style="--n:{n}" data-suffix="{suffix}"></b></div>')
    return ('<div class="kpis">' + tile("Predictions made", made) + tile("Placed", placed)
            + tile("Not placed", not_placed) + tile("Placement rate", rate, "%") + "</div>")


def skill_bars(cgpa, coding, comm):
    rows = ""
    for name, val in (("CGPA", cgpa), ("Coding", coding), ("Communication", comm)):
        rows += (f'<div class="bar"><span><em>{name}</em><em>{val:.1f} / 10</em></span>'
                 f'<div class="track"><div class="fill" style="--w:{val * 10}%"></div></div></div>')
    return rows


def idle_card():
    return ('<div class="idle"><div class="orb"></div>'
            '<h3>Your prediction appears here</h3>'
            '<p>Enter student details and press <b>Predict Placement</b>.</p></div>')


RESULT_HTML = """
<style>
body{margin:0;background:transparent;font-family:'Space Grotesk',system-ui,sans-serif;color:#ECEBFF;text-align:center}
.ring{position:relative;width:210px;height:210px;margin:8px auto;filter:drop-shadow(0 0 18px __COLOR__aa)}
svg{transform:rotate(-90deg)}
.bg{stroke:rgba(255,255,255,.08)}
.fg{stroke:__COLOR__;stroke-linecap:round;stroke-dasharray:440;stroke-dashoffset:440;transition:stroke-dashoffset 1.6s cubic-bezier(.2,.8,.2,1)}
.num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.num b{font-size:2.7rem;color:__COLOR__;font-family:'JetBrains Mono',monospace}
.num small{color:#8E8CA8;letter-spacing:.1em;font-size:.7rem}
h2{margin:.4rem 0 .2rem;color:__COLOR__;letter-spacing:.06em;animation:glow 2.4s ease-in-out infinite}
@keyframes glow{50%{text-shadow:0 0 22px __COLOR__}}
p{color:#8E8CA8;margin:.3rem 1rem;font-size:.95rem}
</style>
<div class="ring">
  <svg width="210" height="210" viewBox="0 0 160 160">
    <circle class="bg" cx="80" cy="80" r="70" fill="none" stroke-width="10"/>
    <circle class="fg" cx="80" cy="80" r="70" fill="none" stroke-width="10"/>
  </svg>
  <div class="num"><b id="n">0.0%</b><small>PLACEMENT CHANCE</small></div>
</div>
<h2>__LABEL__</h2>
<p>__TIP__</p>
<script>
const p=__PROB__, fg=document.querySelector('.fg'), el=document.getElementById('n');
requestAnimationFrame(()=>{fg.style.strokeDashoffset=440*(1-p/100)});
const t0=performance.now(), D=1600;
(function tick(t){const k=Math.min((t-t0)/D,1), e=1-Math.pow(1-k,3);
el.textContent=(p*e).toFixed(1)+'%'; if(k<1) requestAnimationFrame(tick)})(t0);
</script>
"""


def result_card(prob, label, tip):
    color = LIME if label == 1 else EMBER
    text = "PLACED" if label == 1 else "NOT PLACED"
    return (RESULT_HTML.replace("__COLOR__", color).replace("__LABEL__", text)
            .replace("__TIP__", tip).replace("__PROB__", f"{prob:.2f}"))


# =====================================================================
# 3. PAGE
# =====================================================================
st.set_page_config(page_title="CampusPath", page_icon="🎓", layout="wide")
html(f"<style>{CSS}</style>")
st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)

html(hero())
kpi_slot = st.empty()          # prediction ke baad neeche fill hota hai

left, right = st.columns([1, 1.1], gap="large")

with left:
    with st.container(border=True, key="profile"):
        html('<div class="sec"><i>✦</i> Student Profile</div>'
             '<div class="hint">All scores are on a 0–10 scale.</div>')
        cgpa = st.number_input("📘 CGPA", 0.0, 10.0, 7.0, 0.1, format="%.2f")
        coding = st.number_input("💻 Coding skills", 0.0, 10.0, 6.0, 0.1, format="%.2f")
        comm = st.number_input("🗣️ Communication skills", 0.0, 10.0, 6.0, 0.1, format="%.2f")
        clicked = st.button("✨ Predict Placement", use_container_width=True)
    html('<div class="sec"><i>✦</i> Skill snapshot</div>' + skill_bars(cgpa, coding, comm))

if clicked:
    label, prob = predict(cgpa, coding, comm)
    st.session_state.last = (label, prob, advice(cgpa, coding, comm))
    st.session_state.history.append({
        "CGPA": cgpa, "Coding": coding, "Communication": comm,
        "Result": "Placed" if label else "Not placed", "Probability %": round(prob, 1)})

with right:
    with st.container(border=True, key="result"):
        html('<div class="sec"><i>✧</i> Prediction Insights</div>')
        if st.session_state.last:
            label, prob, tip = st.session_state.last
            components.html(result_card(prob, label, tip), height=340)
        else:
            html(idle_card())
    with st.container(border=True, key="howto"):
        html("""
            <div class="sec">🎯 How to read the result</div>
            <ul class="hint">
                <li>Placed = model predicts class 1.</li>
                <li>Not Placed = model predicts class 0.</li>
                <li>Probability is the model's estimated chance of placement.</li>
            </ul>
        """)

hist = st.session_state.history
placed = sum(1 for h in hist if h["Result"] == "Placed")
kpi_slot.markdown(
    kpis(len(hist), placed, len(hist) - placed,
         round(100 * placed / len(hist)) if hist else 0),
    unsafe_allow_html=True)

if hist:
    html('<div class="sec"><i>▤</i> Recent Prediction History</div>')
    st.dataframe(pd.DataFrame(hist[::-1][:10]), use_container_width=True, hide_index=True)
