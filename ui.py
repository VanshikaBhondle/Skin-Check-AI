"""Visual layer: custom CSS + small HTML components for a polished Streamlit UI.

Every component works in light AND dark mode (colours are translucent or theme-inherited),
and every piece of dynamic text is HTML-escaped (AI output never becomes markup).
"""
from __future__ import annotations

from html import escape

import streamlit as st

TEAL = "#0f9b8e"
URGENCY = {  # colour + icon + label + short advice; status colour is never used alone
    "routine": ("#16a34a", "✅", "Routine care", "Home care is usually fine. Keep an eye on it."),
    "see_doctor_soon": ("#f59e0b", "⚠️", "See a doctor soon", "Book an appointment in the next few days."),
    "urgent": ("#dc2626", "🚨", "Get medical help now", "Contact a doctor or emergency service today."),
}

CSS = """
<style>
/* ---------- layout ---------- */
.block-container {padding-top: 2rem; max-width: 1180px;}
[data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}

/* ---------- hero ---------- */
.sc-hero {position: relative; overflow: hidden; border-radius: 24px; padding: 2.4rem 2.4rem 2.2rem;
  color: #fff; background: linear-gradient(135deg, #0b5f58 0%, #0f9b8e 55%, #3ccf9f 100%);
  box-shadow: 0 18px 40px -18px rgba(15, 155, 142, .55); margin-bottom: 1.4rem;}
.sc-hero::before, .sc-hero::after {content: ""; position: absolute; border-radius: 50%;
  background: rgba(255,255,255,.10); pointer-events: none;}
.sc-hero::before {width: 320px; height: 320px; right: -80px; top: -120px;}
.sc-hero::after {width: 180px; height: 180px; right: 160px; bottom: -110px; background: rgba(255,255,255,.07);}
.sc-hero h1 {color: #fff !important; font-size: 2.6rem; font-weight: 800; line-height: 1.1;
  margin: .4rem 0 .6rem; padding: 0; letter-spacing: -.02em;}
.sc-hero p {color: rgba(255,255,255,.92); font-size: 1.08rem; max-width: 620px; margin: 0 0 1.2rem;}
.sc-eyebrow {display: inline-block; font-size: .78rem; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: rgba(255,255,255,.85);}
.sc-pills {display: flex; flex-wrap: wrap; gap: .5rem;}
.sc-pill {display: inline-flex; align-items: center; gap: .4rem; padding: .38rem .8rem; border-radius: 999px;
  font-size: .85rem; font-weight: 600; background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28);
  color: #fff; backdrop-filter: blur(4px);}

/* ---------- generic cards ---------- */
.sc-card {border: 1px solid rgba(127,127,127,.22); background: rgba(127,127,127,.05); border-radius: 18px;
  padding: 1.1rem 1.2rem;}
.sc-steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: .8rem; margin: 0 0 1.6rem;}
.sc-step {display: flex; gap: .8rem; align-items: flex-start;}
.sc-step b {display: block; font-size: .98rem; margin-bottom: .1rem;}
.sc-step span {font-size: .86rem; opacity: .75;}
.sc-num {flex: none; width: 2rem; height: 2rem; border-radius: 10px; display: grid; place-items: center;
  font-weight: 800; color: #fff; background: linear-gradient(135deg, #0b5f58, #0f9b8e);}
.sc-section {display: flex; align-items: center; gap: .75rem; margin: .2rem 0 .9rem;}
.sc-section h3 {margin: 0 !important; padding: 0 !important; font-size: 1.25rem !important; font-weight: 750;}
.sc-section small {display: block; opacity: .7; font-size: .85rem; margin-top: .1rem;}

/* ---------- status / urgency ---------- */
.sc-urgency {--c: #16a34a; border-radius: 20px; padding: 1.2rem 1.4rem; margin: .4rem 0 1.2rem;
  border: 1px solid color-mix(in srgb, var(--c) 45%, transparent); border-left: 8px solid var(--c);
  background: color-mix(in srgb, var(--c) 11%, transparent);}
.sc-urgency .sc-ulabel {font-size: .76rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; opacity: .75;}
.sc-urgency .sc-utitle {font-size: 1.55rem; font-weight: 800; margin: .1rem 0 .25rem;}
.sc-urgency p {margin: .15rem 0; }
.sc-urgency ul {margin: .5rem 0 0 1.1rem; padding: 0;}
.sc-urgency li {margin: .15rem 0;}

/* ---------- condition cards ---------- */
.sc-cond {border: 1px solid rgba(127,127,127,.22); border-radius: 16px; padding: .9rem 1.05rem; margin-bottom: .65rem;
  background: rgba(127,127,127,.04);}
.sc-cond.top {border: 1.5px solid color-mix(in srgb, #0f9b8e 60%, transparent);
  background: color-mix(in srgb, #0f9b8e 8%, transparent);}
.sc-cond-head {display: flex; justify-content: space-between; align-items: baseline; gap: .8rem;}
.sc-cond-name {font-weight: 750; font-size: 1.05rem;}
.sc-cond.top .sc-cond-name {font-size: 1.25rem;}
.sc-badge {display: inline-block; white-space: nowrap; font-size: .68rem; font-weight: 800; letter-spacing: .08em;
  text-transform: uppercase; color: #fff; background: #0f8a7e; border-radius: 999px; padding: .15rem .6rem;
  margin-bottom: .35rem;}
.sc-cond-name {min-width: 0; overflow-wrap: anywhere;}
.sc-pct {font-weight: 800; font-size: 1.05rem; white-space: nowrap;}
.sc-track {height: 8px; border-radius: 999px; background: rgba(127,127,127,.18); margin: .55rem 0 .45rem; overflow: hidden;}
.sc-fill {height: 100%; border-radius: 999px; background: linear-gradient(90deg, #0b5f58, #0f9b8e 60%, #3ccf9f);}
.sc-why {font-size: .88rem; opacity: .78; margin: 0;}

/* ---------- small things ---------- */
.sc-panel-title {font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; opacity: .65;
  margin: 0 0 .5rem;}
.sc-quote {border-left: 4px solid #0f9b8e; padding: .7rem 1rem; border-radius: 0 14px 14px 0;
  background: color-mix(in srgb, #0f9b8e 9%, transparent); margin: .6rem 0 1rem;}
.sc-sees {font-size: 1.02rem; line-height: 1.6;}
.sc-foot {text-align: center; font-size: .82rem; opacity: .65; margin: 2.5rem 0 .5rem;}
.sc-brand {display: flex; align-items: center; gap: .65rem; margin-bottom: 1.1rem;}
.sc-logo {width: 2.4rem; height: 2.4rem; border-radius: 12px; display: grid; place-items: center; font-size: 1.3rem;
  background: linear-gradient(135deg, #0b5f58, #0f9b8e 55%, #3ccf9f);}
.sc-brand b {font-size: 1.12rem; display: block; line-height: 1.1;}
.sc-brand span {font-size: .78rem; opacity: .65;}
.sc-status {display: flex; align-items: center; gap: .5rem; font-size: .85rem; padding: .45rem .7rem;
  border-radius: 12px; border: 1px solid rgba(127,127,127,.2); margin-bottom: .45rem;}
.sc-dot {width: .55rem; height: .55rem; border-radius: 50%; flex: none;}
.sc-page {margin: .2rem 0 1.4rem;}
.sc-page h1 {font-size: 2.1rem !important; font-weight: 800; margin: 0 !important; padding: 0 !important; letter-spacing: -.02em;}
.sc-page p {opacity: .72; margin: .3rem 0 0;}

/* ---------- Streamlit widgets ---------- */
button[kind="primary"] {background: linear-gradient(135deg, #0b5f58, #0f9b8e 60%, #22b89a) !important;
  border: none !important; font-weight: 700 !important; padding: .7rem 1rem !important;
  box-shadow: 0 10px 24px -12px rgba(15,155,142,.8); transition: transform .12s ease, box-shadow .12s ease;}
button[kind="primary"]:hover:not(:disabled) {transform: translateY(-1px); box-shadow: 0 14px 28px -12px rgba(15,155,142,.9);}
button[kind="primary"]:disabled {background: rgba(127,127,127,.35) !important; box-shadow: none;}
[data-testid="stFileUploaderDropzone"] {border: 2px dashed color-mix(in srgb, #0f9b8e 55%, transparent) !important;
  background: color-mix(in srgb, #0f9b8e 5%, transparent) !important; border-radius: 16px !important;}
.stTabs [data-baseweb="tab-list"] {gap: .35rem;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
[data-testid="stImage"] img {border-radius: 16px;}
[data-testid="stMetric"] {border-radius: 16px;}

@media (max-width: 760px) {
  .sc-hero {padding: 1.6rem 1.3rem; border-radius: 20px;}
  .sc-hero h1 {font-size: 1.9rem;}
  .sc-steps {grid-template-columns: 1fr;}
  .block-container {padding-top: 3.2rem;}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def hero(model_text: str | None, ai_name: str | None) -> None:
    pills = []
    if model_text:
        pills.append(f"🧠 {escape(model_text)}")
    if ai_name:
        pills.append(f"🤖 {escape(ai_name)} vision AI")
    pills += ["🔒 Photos never stored", "⚡ Results in ~20 s"]
    _html(f"""<div class="sc-hero"><span class="sc-eyebrow">AI skin health check</span>
<h1>Understand your skin,<br>in seconds.</h1>
<p>Upload a photo of a skin concern. A trained model and an AI assistant suggest what it could be,
how to care for it, and when to see a doctor.</p>
<div class="sc-pills">{''.join(f'<span class="sc-pill">{p}</span>' for p in pills)}</div></div>""")


def steps() -> None:
    items = [("Add a photo", "Upload or take a clear, close-up picture."),
             ("Tell us a little", "Age, where it is, how long, any symptoms."),
             ("Get your report", "Likely conditions, care tips, urgency.")]
    cells = "".join(f'<div class="sc-card sc-step"><div class="sc-num">{i}</div><div><b>{t}</b><span>{d}</span></div></div>'
                    for i, (t, d) in enumerate(items, 1))
    _html(f'<div class="sc-steps">{cells}</div>')


def section(num: int | str, title: str, subtitle: str = "") -> None:
    sub = f"<small>{escape(subtitle)}</small>" if subtitle else ""
    _html(f'<div class="sc-section"><div class="sc-num">{escape(str(num))}</div><div><h3>{escape(title)}</h3>{sub}</div></div>')


def page_header(title: str, subtitle: str = "") -> None:
    _html(f'<div class="sc-page"><h1>{escape(title)}</h1>{f"<p>{escape(subtitle)}</p>" if subtitle else ""}</div>')


def urgency_card(level: str, reason: str, flags: list[str], raised_by_rules: bool = False) -> None:
    color, icon, label, advice = URGENCY[level]
    items = "".join(f"<li>{escape(f)}</li>" for f in flags)
    extra = "<p><small>⬆️ Raised by the app's safety rules because of the symptoms you reported.</small></p>" \
        if raised_by_rules else ""
    _html(f"""<div class="sc-urgency" style="--c:{color}"><div class="sc-ulabel">Urgency</div>
<div class="sc-utitle">{icon} {label}</div><p><b>{escape(advice)}</b> {escape(reason)}</p>
{f'<ul>{items}</ul>' if items else ''}{extra}</div>""")


def condition_cards(items: list[dict], top_label: str = "Most likely") -> None:
    out = []
    for i, c in enumerate(items):
        pct = max(0.0, min(100.0, float(c["confidence"])))
        badge = f'<div><span class="sc-badge">{escape(top_label)}</span></div>' if i == 0 else ""
        why = f'<p class="sc-why">{escape(c["why"])}</p>' if c.get("why") else ""
        out.append(f"""<div class="sc-cond{' top' if i == 0 else ''}">{badge}<div class="sc-cond-head">
<span class="sc-cond-name">{escape(c['name'])}</span><span class="sc-pct">{pct:g}%</span></div>
<div class="sc-track" role="progressbar" aria-valuenow="{pct:g}" aria-valuemin="0" aria-valuemax="100">
<div class="sc-fill" style="width:{pct}%"></div></div>{why}</div>""")
    _html("".join(out))


def panel_title(text: str) -> None:
    _html(f'<div class="sc-panel-title">{escape(text)}</div>')


def quote(label: str, text: str) -> None:
    _html(f'<div class="sc-quote"><b>{escape(label)}</b> {escape(text)}</div>')


def sees(text: str) -> None:
    _html(f'<div class="sc-card sc-sees">👁️ <b>What the AI sees:</b> {escape(text)}</div>')


def sidebar_brand(model_text: str | None, ai_name: str | None) -> None:
    def status(ok: bool, on: str, off: str) -> str:
        color = "#16a34a" if ok else "#9ca3af"
        return f'<div class="sc-status"><span class="sc-dot" style="background:{color}"></span>{escape(on if ok else off)}</div>'
    st.sidebar.markdown('<div class="sc-brand"><div class="sc-logo">🩺</div><div><b>Skin Check AI</b>'
                        '<span>Educational skin assistant</span></div></div>', unsafe_allow_html=True)
    st.sidebar.markdown(status(bool(model_text), f"Trained model · {model_text}" if model_text else "", "Trained model not added")
                        + status(bool(ai_name), f"AI assistant · {ai_name}", "AI assistant not configured"),
                        unsafe_allow_html=True)


def footer() -> None:
    _html('<div class="sc-foot">Skin Check AI is an educational tool and does not provide medical diagnoses. '
          'Always consult a qualified doctor or dermatologist. · Photos are never stored.</div>')
