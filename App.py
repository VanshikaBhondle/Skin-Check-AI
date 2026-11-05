"""Skin Check AI - Streamlit app (Analyze / Feedback / Admin / About)."""
from __future__ import annotations

import hashlib
import hmac
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

import analyzer as az
import classifier as clf_mod
import db
import ui

st.set_page_config(page_title="Skin Check AI", page_icon="🩺", layout="wide")
db.init_db()
ui.inject_css()

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin@skin-check")
ACCESS_CODE = os.getenv("APP_ACCESS_CODE", "")          # optional: require a code before AI use
PER_SESSION = int(os.getenv("MAX_ANALYSES_PER_SESSION", "0"))  # 0 = no limit
PER_DAY = int(os.getenv("DAILY_ANALYSIS_LIMIT", "0"))           # 0 = no limit

BLUE = ui.TEAL  # single-series bar colour
URGENCY_STYLE = {  # status colour + icon + label: never colour alone
    "routine": ("#0ca30c", "✅"), "see_doctor_soon": ("#fab219", "⚠️"), "urgent": ("#d03b3b", "🚨")}
GRID = "rgba(127,127,127,0.18)"  # works on light and dark backgrounds
DISCLAIMER = ("**Not a diagnosis.** This tool gives educational AI suggestions from a photo. It can be wrong. "
              "Always ask a doctor or dermatologist about skin concerns - and get urgent help for fast-spreading "
              "redness with fever, a painful rash near the eye, or a spot that bleeds or changes.")


@st.cache_resource
def get_classifier():
    """Load the trained model once per server. Returns the classifier, None, or the load error."""
    try:
        return clf_mod.load()
    except Exception as e:  # e.g. a corrupt or mismatched model file
        return e


def style_fig(fig, height=300):
    fig.update_layout(height=height, margin=dict(l=8, r=40, t=36, b=8), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", bargap=0.35, showlegend=False,
                      font=dict(size=13), title=dict(font=dict(size=15)))
    fig.update_xaxes(gridcolor=GRID, zeroline=False, title=None)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", title=None)
    return fig


def hbar(df: pd.DataFrame, col: str, title: str, top: int = 10):
    """Horizontal bar of category counts - one hue, labelled values, sorted."""
    if df.empty or col not in df or df[col].dropna().empty:
        st.info(f"No data yet for {title.lower()}.")
        return
    counts = df[col].fillna("Not given").replace("", "Not given").value_counts().head(top)
    data = counts.rename_axis(col).reset_index(name="count").iloc[::-1]
    fig = px.bar(data, x="count", y=col, orientation="h", title=title, text="count")
    fig.update_traces(marker_color=BLUE, marker_cornerradius=4, textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}: %{x}<extra></extra>")
    st.plotly_chart(style_fig(fig, 80 + 34 * len(data)), use_container_width=True)


# ============================================================================ analyze
def analyze_page(clf, ai_ready: bool):
    ui.hero(f"Trained model · {len(clf.labels)} conditions" if clf else None, ai_ready)
    if not ai_ready and clf is None:
        st.error("Not set up yet: add a trained model to `models/` and/or `ANTHROPIC_API_KEY` to Secrets.")
    elif not ai_ready:
        st.info("Running with the trained model only. Add `ANTHROPIC_API_KEY` for the detailed AI report and chat.")
    elif clf is None:
        st.info("No trained model in `models/` yet - using the AI's own image analysis.")
    ui.steps()

    left, right = st.columns([1, 1], gap="large")
    with left.container(border=True):
        ui.section(1, "Your photo", "A clear close-up in good light works best")
        tab_up, tab_cam = st.tabs(["📁 Upload", "📷 Take a photo"])
        with tab_up:
            up = st.file_uploader("JPG, PNG or WEBP, up to 10 MB", type=["jpg", "jpeg", "png", "webp"])
        with tab_cam:
            cam = st.camera_input("Camera", label_visibility="collapsed")
        src = up or cam
        if src:
            st.image(src, use_container_width=True)
        with st.expander("💡 Tips for a good photo"):
            st.markdown("- Use daylight, no flash\n- Hold the camera 10-15 cm away, in focus\n"
                        "- Show the whole spot plus a little normal skin around it\n"
                        "- One concern per photo")

    with right.container(border=True):
        ui.section(2, "About you", "Helps the AI give better, safer advice")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=0, max_value=120, value=None, placeholder="e.g. 25")
        gender = c2.selectbox("Gender", az.GENDERS)
        c3, c4 = st.columns(2)
        skin_type = c3.selectbox("Skin type", az.SKIN_TYPES)
        body_area = c4.selectbox("Where is it?", az.BODY_AREAS)
        duration = st.select_slider("How long have you had it?", az.DURATIONS)
        symptoms = st.multiselect("Symptoms", az.SYMPTOMS)
        immune = st.checkbox("I have diabetes or a weakened immune system")
        history = st.text_area("Medical history / notes (optional)", max_chars=500,
                               placeholder="e.g. eczema as a child, started a new soap last week")
        st.caption("⚠️ " + DISCLAIMER.replace("**", ""))
        consent = st.checkbox("I understand this is not a medical diagnosis")
        code_ok = True
        if ACCESS_CODE:
            code = st.text_input("Access code", type="password")
            code_ok = hmac.compare_digest(code, ACCESS_CODE)

        used = st.session_state.get("used", 0)
        go = st.button("🔍 Analyze my skin", type="primary", use_container_width=True,
                       disabled=not (src and consent and code_ok and (clf or ai_ready)))
        if not src:
            st.caption("Add a photo to continue.")
        elif not consent:
            st.caption("Tick the box above to continue.")
        elif not code_ok:
            st.caption("Enter the access code to continue.")

    info = {"age": age, "gender": gender, "skin_type": skin_type, "body_area": body_area,
            "duration": duration, "symptoms": symptoms, "immune": immune, "history": history.strip()}

    if go:
        try:
            jpeg = az.prepare_image(src.getvalue())
        except az.AnalysisError as e:
            st.error(str(e))
            return
        model_info = clf.info if clf else ""
        preds = clf.predict(jpeg) if clf else None
        result = None
        if ai_ready:
            if PER_SESSION and used >= PER_SESSION:
                st.warning(f"AI report limit ({PER_SESSION} per visit) reached - showing the trained model only.")
            elif PER_DAY and db.count_today() >= PER_DAY:
                st.warning("Today's AI report limit is reached - showing the trained model only.")
            else:
                try:
                    with st.spinner("Analyzing your photo... this takes about 15-30 seconds."):
                        result = az.analyze_image(jpeg, info, model_preds=preds, model_info=model_info)
                    st.session_state["used"] = used + 1
                except az.AnalysisError as e:
                    st.error(str(e))
        if result is None and not preds:
            return
        rule_level, rule_flags = az.red_flag_rules(info)
        top = result.top if result else None
        rid = db.insert_analysis(
            timestamp=db.now(), age_band=db.age_band(age), gender=gender, skin_type=skin_type,
            body_area=body_area, duration=duration, symptoms=", ".join(symptoms),
            image_usable=int(result.usable) if result else 1,
            top_condition=top["name"] if top else (preds[0]["name"] if preds and not result else "Not analysed"),
            top_confidence=top["confidence"] if top else (preds[0]["confidence"] if preds and not result else None),
            conditions=json.dumps([c["name"] for c in (result.conditions if result else preds)]),
            ai_urgency=result.data["urgency"] if result else None,
            final_urgency=result.urgency if result else rule_level,
            rule_flags=json.dumps(result.rule_flags if result else rule_flags),
            model=result.model if result else "trained model only",
            model_top=preds[0]["name"] if preds else None,
            model_confidence=preds[0]["confidence"] if preds else None)
        st.session_state.update(
            result=result, preds=preds, model_info=model_info, info=info, rid=rid, chat_view=[],
            chat=az.start_chat(jpeg, info, result, preds, model_info) if result else None,
            img_key=hashlib.sha1(src.getvalue()).hexdigest())

    # Show the latest result, unless a different photo has been chosen since
    if "rid" in st.session_state and not (src and hashlib.sha1(src.getvalue()).hexdigest()
                                          != st.session_state.get("img_key")):
        show_result(st.session_state["result"], st.session_state["preds"], st.session_state["info"],
                    st.session_state["model_info"])


def show_result(a: az.Analysis | None, preds: list[dict] | None, info: dict, model_info: str):
    st.write("")
    ui.section("✓", "Your results", "Educational suggestions - not a diagnosis")
    if a:
        level, reason, flags = a.urgency, a.data["urgency_reason"], a.data["red_flags"] + a.rule_flags
    else:
        level, flags = az.red_flag_rules(info)
        reason = "Based on the symptoms you reported."
    ui.urgency_card(level, reason, flags, raised_by_rules=bool(a and a.urgency != a.data["urgency"]))

    ai_ok = bool(a and a.usable)
    cols = st.columns(2, gap="large") if (preds and ai_ok) else [st.container()]
    if preds:
        with cols[0].container(border=True):
            ui.panel_title("🧠 Trained model")
            ui.condition_cards(preds, "Top match")
            st.caption(f"Your {model_info}. It can only choose between the conditions it was trained on.")
    if ai_ok:
        with cols[-1].container(border=True):
            ui.panel_title("🤖 AI assessment")
            ui.condition_cards(a.conditions, "Most likely")
    if a and ai_ok and preds and a.data.get("classifier_assessment"):
        ui.quote("🔎 AI review of the model's prediction:", a.data["classifier_assessment"])

    if a is None:
        st.info("The detailed report (causes, care routine, when to see a doctor) and the follow-up chat need "
                "the AI. Add `ANTHROPIC_API_KEY` in Secrets to turn them on.")
        st.download_button("⬇️ Download report", az.report_markdown(info, None, preds, model_info),
                           "skin-check-report.md", "text/markdown")
        return

    d = a.data
    if not a.usable:
        st.error("😕 **The AI couldn't analyse this photo reliably.** " + d["image_quality"]["issues"]
                 + " Try a sharper close-up in daylight.")
        st.info(d["recommendation"])
        return

    ui.sees(d["visual_description"])
    st.write("")
    t1, t2, t3, t4 = st.tabs(["📖 Overview & causes", "🧴 Care routine", "🌿 Lifestyle tips", "🩺 When to see a doctor"])
    with t1:
        st.write(d["overview"])
        st.markdown("\n".join(f"- {x}" for x in d["causes"]))
    with t2:
        st.markdown("\n".join(f"{i}. {x}" for i, x in enumerate(d["care_routine"], 1)))
    with t3:
        st.markdown("\n".join(f"- {x}" for x in d["lifestyle_tips"]))
    with t4:
        st.write(d["when_to_see_doctor"])
    ui.quote("💡 Recommendation:", d["recommendation"])

    c1, c2 = st.columns([1, 1])
    c1.download_button("⬇️ Download report", az.report_markdown(info, a, preds, model_info),
                       "skin-check-report.md", "text/markdown")
    with c2:
        st.caption("Was this helpful?")
        fb = st.feedback("thumbs", key=f"helpful_{st.session_state['rid']}")
        if fb is not None and st.session_state.get("helpful_saved") != st.session_state["rid"]:
            db.set_helpful(st.session_state["rid"], int(fb))
            st.session_state["helpful_saved"] = st.session_state["rid"]

    st.write("")
    ui.section("💬", "Ask a follow-up question", "e.g. Is it contagious? Can I exercise? What should I avoid?")
    for role, text in st.session_state["chat_view"]:
        st.chat_message(role).write(text)
    if len(st.session_state["chat_view"]) >= 20:
        st.caption("Question limit reached for this analysis.")
    elif q := st.chat_input("e.g. Is it contagious? What should I avoid?"):
        st.chat_message("user").write(q)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                answer = az.ask_followup(st.session_state["chat"], q)
            except az.AnalysisError as e:
                st.error(str(e))
            else:
                st.write(answer)
                st.session_state["chat_view"] += [("user", q), ("assistant", answer)]


# ============================================================================ feedback
def feedback_page():
    ui.page_header("💬 Feedback", "Tell us how Skin Check AI worked for you.")
    with st.form("fb", clear_on_submit=True):
        name = st.text_input("Name (optional)")
        rating = st.slider("Rating", 1, 5, 4)
        comments = st.text_area("Comments", max_chars=1000)
        if st.form_submit_button("Submit"):
            db.insert_feedback(timestamp=db.now(), name=name.strip() or "Anonymous", rating=rating,
                               comments=comments.strip())
            st.success("Thanks for your feedback!")
    fb = db.feedback()
    if not fb.empty:
        st.metric("Average rating", f"{fb['rating'].astype(float).mean():.1f} / 5", f"{len(fb)} reviews",
                  delta_color="off")
        hbar(fb.assign(stars=fb["rating"].astype(int).map(lambda r: "★" * r)), "stars", "Ratings")
        st.subheader("Recent comments")
        st.dataframe(fb[fb["comments"].fillna("") != ""][["name", "rating", "comments", "timestamp"]].head(30),
                     hide_index=True, use_container_width=True)


# ============================================================================ admin
def admin_page():
    ui.page_header("🛠️ Admin", "Usage, results and feedback. Photos are never stored.")
    if not st.session_state.get("admin"):
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                if hmac.compare_digest(u, ADMIN_USER) and hmac.compare_digest(p, ADMIN_PASS):
                    st.session_state["admin"] = True
                    st.rerun()
                st.error("Wrong credentials.")
        if "ADMIN_PASS" not in os.environ:
            st.caption("Using the default password - set ADMIN_USER / ADMIN_PASS in Secrets before deploying.")
        return
    if st.button("Log out"):
        st.session_state["admin"] = False
        st.rerun()

    df, fb = db.analyses(), db.feedback()
    helpful = pd.to_numeric(df.get("helpful"), errors="coerce").dropna() if not df.empty else pd.Series(dtype=float)
    c = st.columns(4)
    c[0].metric("Analyses", len(df), border=True)
    c[1].metric("Analyses today", db.count_today(), f"limit {PER_DAY}" if PER_DAY else "no limit",
                delta_color="off", border=True)
    c[2].metric("Marked helpful", f"{helpful.mean() * 100:.0f}%" if len(helpful) else "-", border=True)
    c[3].metric("Avg rating", f"{fb['rating'].astype(float).mean():.1f}" if len(fb) else "-", border=True)

    st.subheader("Analyses")
    st.caption("Photos are never stored - only anonymous details and results.")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), "skin_analyses.csv", "text/csv")

    st.subheader("Analytics")
    if df.empty:
        st.info("No analyses yet.")
    else:
        usable = df[df["image_usable"].astype(int) == 1]
        urg = df["final_urgency"].map(lambda u: f"{URGENCY_STYLE.get(u, ('', ''))[1]} {az.URGENCY_LABEL.get(u, u)}")
        a, b = st.columns(2)
        with a:
            hbar(usable, "top_condition", "Most common conditions (final result)")
        with b:
            hbar(df.assign(urgency=urg), "urgency", "Urgency")
        if "model_top" in df and df["model_top"].notna().any():
            both = df[df["model_top"].notna() & (df["model"] != "trained model only") & (df["image_usable"].astype(int) == 1)]
            a, b = st.columns(2)
            with a:
                hbar(df, "model_top", "Trained model predictions")
            with b:
                if len(both):
                    agree = (both["model_top"].str.lower() == both["top_condition"].str.lower()).mean()
                    st.metric("Trained model agrees with AI's top answer", f"{agree * 100:.0f}%",
                              f"{len(both)} analyses with both", delta_color="off")
        for left, right in (("age_band", "gender"), ("body_area", "skin_type"), ("duration", "image_usable")):
            a, b = st.columns(2)
            with a:
                hbar(df, left, left.replace("_", " ").capitalize())
            with b:
                if right == "image_usable":
                    hbar(df.assign(photo=df["image_usable"].astype(int).map({1: "Usable", 0: "Not usable"})),
                         "photo", "Photo quality")
                else:
                    hbar(df, right, right.replace("_", " ").capitalize())
        days = pd.to_datetime(df["timestamp"]).dt.date.value_counts().sort_index()
        fig = px.bar(x=days.index.astype(str), y=days.values, title="Analyses per day", labels={"x": "", "y": ""})
        fig.update_traces(marker_color=BLUE, marker_cornerradius=4, hovertemplate="%{x}: %{y}<extra></extra>")
        fig.update_yaxes(gridcolor=GRID)
        fig.update_xaxes(gridcolor="rgba(0,0,0,0)", type="category")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)

    st.subheader("Feedback")
    st.dataframe(fb, hide_index=True, use_container_width=True)
    st.download_button("⬇️ Download feedback CSV", fb.to_csv(index=False).encode(), "skin_feedback.csv", "text/csv")


def about_page():
    ui.page_header("ℹ️ About", "How Skin Check AI works.")
    st.markdown(f"""
**Skin Check AI** looks at a photo of a skin concern together with details you enter, and gives:

- the most likely conditions with a confidence score,
- what it sees in the photo, common causes, a gentle care routine and lifestyle tips,
- an urgency level and clear advice on when to see a doctor,
- a downloadable report and follow-up questions.

**How it works:** the photo is cleaned (rotation fixed, location data removed, resized). A **trained image
classifier** (a deep-learning model trained on labelled skin-disease photos) predicts the most likely condition.
The photo, your details and that prediction then go to Claude, a vision-capable AI model, which checks the
prediction and writes a structured report. The app then applies its own
**safety rules** to the symptoms you reported and raises the urgency when needed (for example fever with
spreading redness, or a spot that bleeds or changes).

**Privacy:** photos are not saved. Only anonymous details (age band, area, result) are stored for statistics.

{DISCLAIMER}
""")


CLF = get_classifier()
if isinstance(CLF, Exception):
    st.warning(f"The trained model couldn't be loaded ({CLF}). Using the AI only.")
    CLF = None
AI_READY = az.get_client() is not None

ui.sidebar_brand(CLF.short if CLF else None, AI_READY)
PAGES = {"🔍 Analyze": lambda: analyze_page(CLF, AI_READY), "💬 Feedback": feedback_page,
         "ℹ️ About": about_page, "🛠️ Admin": admin_page}
choice = st.sidebar.radio("Menu", list(PAGES), label_visibility="collapsed")
st.sidebar.caption("Educational tool - not a medical diagnosis.")
PAGES[choice]()
ui.footer()
