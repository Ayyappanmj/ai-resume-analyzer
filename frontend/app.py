"""Streamlit dark-UI front end for the AI Resume Analyzer."""
import html
import os

import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.card {background:#131a2b; border:1px solid #232c45; border-radius:14px; padding:1.1rem 1.3rem; margin-bottom:1rem;}
.card h4 {margin:0 0 .6rem 0; font-size:1rem; color:#c4b5fd;}
.metric {background:#131a2b; border:1px solid #232c45; border-radius:12px; padding:.9rem; text-align:center;}
.metric .v {font-size:1.7rem; font-weight:700;} .metric .l {font-size:.75rem; color:#94a3b8;}
.pill {display:inline-block; padding:.18rem .65rem; margin:.15rem .2rem .15rem 0; border-radius:999px; font-size:.8rem; border:1px solid;}
.ok {background:#0f2a1d; color:#4ade80; border-color:#166534;}
.bad {background:#2c1416; color:#f87171; border-color:#7f1d1d;}
.neutral {background:#1a2140; color:#a5b4fc; border-color:#3730a3;}
.row {display:flex; justify-content:space-between; padding:.4rem 0; border-bottom:1px solid #1f2842; font-size:.9rem;}
.muted {color:#94a3b8; font-size:.85rem;}
</style>
""",
    unsafe_allow_html=True,
)


def pills(items, kind="neutral"):
    if not items:
        return "<span class='muted'>None</span>"
    return "".join(f"<span class='pill {kind}'>{html.escape(str(i))}</span>" for i in items)


def metric(label, value, color="#e5e7eb"):
    return f"<div class='metric'><div class='v' style='color:{color}'>{value}</div><div class='l'>{label}</div></div>"


def score_color(pct):
    return "#4ade80" if pct >= 70 else "#fbbf24" if pct >= 50 else "#f87171"


def gauge(score):
    fig = go.Figure(go.Pie(values=[score, 100 - score], hole=0.78, sort=False, direction="clockwise",
                           marker=dict(colors=[score_color(score), "#1f2842"]), textinfo="none", hoverinfo="skip"))
    fig.add_annotation(text=f"<b>{score}%</b>", x=0.5, y=0.53, showarrow=False, font=dict(size=42, color=score_color(score)))
    fig.add_annotation(text="ATS score", x=0.5, y=0.36, showarrow=False, font=dict(size=13, color="#94a3b8"))
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0), height=260,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def radar(breakdown):
    names = [b["name"] for b in breakdown]
    cur = [round(100 * b["score"] / b["max"], 1) for b in breakdown]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[100] * len(names) + [100], theta=names + names[:1], name="Target",
                                  line=dict(color="#8b5cf6", dash="dot"), fill="toself", fillcolor="rgba(139,92,246,.08)"))
    fig.add_trace(go.Scatterpolar(r=cur + cur[:1], theta=names + names[:1], name="Your resume",
                                  line=dict(color="#4ade80"), fill="toself", fillcolor="rgba(74,222,128,.25)"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 100], gridcolor="#232c45", tickfont=dict(size=9)),
                                 angularaxis=dict(gridcolor="#232c45")),
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#e5e7eb"), height=340,
                      margin=dict(l=40, r=40, t=20, b=20), legend=dict(orientation="h", y=-0.1))
    return fig


def check_backend():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json()
    except Exception:
        return None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 📄 Resume Analyzer")
    health = check_backend()
    if health:
        st.success("API online")
        st.caption(f"Ollama ({health['model']}): {'✅ ready' if health['ollama'] else '❌ not reachable'}  \n"
                   f"Database: {'✅ on' if health['database'] else '⚪ off'}")
    else:
        st.error(f"API offline ({API_URL})")
    use_llm = st.toggle("AI insights (Ollama / Llama 3.1)", value=True,
                        help="Adds an AI summary, bullet rewrites and tailored interview questions.")
    if health and health.get("database"):
        try:
            hist = requests.get(f"{API_URL}/history", timeout=5).json()
            if hist:
                st.markdown("**Recent analyses**")
                for h in hist[:8]:
                    st.caption(f"#{h['id']} · {h['filename']} · **{h['ats_score']}**")
        except Exception:
            pass

st.title("AI Resume Analyzer")
st.caption("Upload a resume and a job description - get an ATS score, skill gaps, keyword analysis and a downloadable report. "
           "Everything runs locally.")

# ---------------------------------------------------------------- inputs
c1, c2 = st.columns(2)
with c1:
    resume = st.file_uploader("1 · Resume (PDF)", type=["pdf"])
with c2:
    jd_upload = st.file_uploader("2 · Job description (optional file)", type=["pdf", "txt"])
jd_text = st.text_area("…or paste the job description", height=170, placeholder="Paste the job posting here")

if st.button("Analyze resume", type="primary", use_container_width=True, disabled=resume is None):
    files = {"resume": (resume.name, resume.getvalue(), "application/pdf")}
    if jd_upload is not None and not jd_text.strip():
        files["jd_file"] = (jd_upload.name, jd_upload.getvalue(), jd_upload.type or "application/octet-stream")
    with st.spinner("Analyzing… the first run downloads models and can take a minute"):
        try:
            resp = requests.post(f"{API_URL}/analyze", files=files,
                                 data={"jd_text": jd_text, "use_llm": str(use_llm).lower()}, timeout=400)
            if resp.status_code == 200:
                st.session_state["result"] = resp.json()
                st.session_state.pop("pdf", None)
            else:
                st.error(resp.json().get("detail", resp.text))
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")

r = st.session_state.get("result")
if not r:
    st.info("Upload a resume PDF and provide a job description to begin.")
    st.stop()

# ---------------------------------------------------------------- results
st.divider()
top = st.columns([1.1, 1.3, 1.3])
with top[0]:
    st.plotly_chart(gauge(r["ats_score"]), use_container_width=True, config={"displayModeBar": False})
    st.markdown(f"<div style='text-align:center'><span class='pill neutral'>{r['grade']}</span></div>", unsafe_allow_html=True)
with top[1]:
    rows = ""
    for b in r["breakdown"]:
        pct = b["score"] / b["max"] * 100
        icon = "✅" if pct >= 70 else "⚠️" if pct >= 40 else "❌"
        rows += f"<div class='row'><span>{icon} {html.escape(b['name'])}</span><span><b>{b['score']:g}</b> / {b['max']:g}</span></div>"
    st.markdown(f"<div class='card'><h4>Score breakdown</h4>{rows}</div>", unsafe_allow_html=True)
with top[2]:
    st.plotly_chart(radar(r["breakdown"]), use_container_width=True, config={"displayModeBar": False})

m = st.columns(4)
m[0].markdown(metric("JD similarity", f"{r['similarity_score']:.1f}%", "#a5b4fc"), unsafe_allow_html=True)
m[1].markdown(metric("Skills matched", f"{len(r['matched_skills'])}/{len(r['jd_skills'])}", "#4ade80"), unsafe_allow_html=True)
m[2].markdown(metric("Missing skills", len(r["missing_skills"]), "#f87171"), unsafe_allow_html=True)
m[3].markdown(metric("Words · pages", f"{r['word_count']} · {r['pages']}"), unsafe_allow_html=True)

if "pdf" not in st.session_state:
    try:
        rep = requests.post(f"{API_URL}/report", json=r, timeout=60)
        st.session_state["pdf"] = rep.content if rep.ok else None
    except requests.RequestException:
        st.session_state["pdf"] = None
if st.session_state.get("pdf"):
    st.download_button("⬇️ Download PDF report", st.session_state["pdf"], file_name="resume-analysis-report.pdf",
                       mime="application/pdf", use_container_width=True)

t_over, t_skills, t_kw, t_sec, t_ai, t_int = st.tabs(
    ["Overview", "Skills", "Keywords", "Sections", "AI insights", "Rewrite & interview"])

with t_over:
    st.markdown(f"<div class='card'><h4>Resume summary</h4>{html.escape(r['summary'])}</div>", unsafe_allow_html=True)
    tips = "".join(f"<li>{html.escape(t)}</li>" for t in r["suggestions"])
    st.markdown(f"<div class='card'><h4>Recommendations</h4><ul>{tips}</ul></div>", unsafe_allow_html=True)
    if r["contact"]:
        st.markdown(f"<div class='card'><h4>Contact detected</h4>{pills([f'{k}: {v}' for k, v in r['contact'].items()])}</div>",
                    unsafe_allow_html=True)

with t_skills:
    a, b = st.columns(2)
    a.markdown(f"<div class='card'><h4>✅ Matched skills</h4>{pills(r['matched_skills'], 'ok')}</div>", unsafe_allow_html=True)
    b.markdown(f"<div class='card'><h4>❌ Missing skills</h4>{pills(r['missing_skills'], 'bad')}</div>", unsafe_allow_html=True)
    st.markdown("##### All skills found in your resume")
    for cat, skills in r["resume_skills"].items():
        st.markdown(f"**{cat}**<br>{pills(skills)}", unsafe_allow_html=True)

with t_kw:
    a, b = st.columns(2)
    a.markdown(f"<div class='card'><h4>Top resume keywords</h4>{pills(r['keywords'])}</div>", unsafe_allow_html=True)
    b.markdown(f"<div class='card'><h4>Job keywords</h4>{pills(r['jd_keywords'])}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card'><h4>Job keywords missing from your resume</h4>{pills(r['missing_keywords'], 'bad')}</div>",
                unsafe_allow_html=True)

with t_sec:
    a, b = st.columns(2)
    a.markdown(f"<div class='card'><h4>Detected sections</h4>{pills(r['sections_detected'], 'ok')}</div>", unsafe_allow_html=True)
    b.markdown(f"<div class='card'><h4>Missing recommended sections</h4>{pills(r['sections_missing'], 'bad')}</div>", unsafe_allow_html=True)

with t_ai:
    if not r["llm_used"]:
        st.warning("AI insights unavailable - start Ollama (`ollama pull llama3.1`) and re-run, or enable the toggle.")
    a, b = st.columns(2)
    a.markdown("<div class='card'><h4>Strengths</h4><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in r["strengths"]) + "</ul></div>",
               unsafe_allow_html=True)
    b.markdown("<div class='card'><h4>Weaknesses</h4><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in r["weaknesses"]) + "</ul></div>",
               unsafe_allow_html=True)

with t_int:
    label = "Improved bullet points" if r["llm_used"] else "Bullet templates (enable Ollama for real rewrites)"
    st.markdown(f"<div class='card'><h4>{label}</h4><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in r["improved_bullets"]) + "</ul></div>",
                unsafe_allow_html=True)
    st.markdown("<div class='card'><h4>Likely interview questions</h4><ol>" +
                "".join(f"<li>{html.escape(x)}</li>" for x in r["interview_questions"]) + "</ol></div>", unsafe_allow_html=True)
