"""
frontend.py
===========
All Streamlit UI rendering functions and CSS.
No ML logic here — only display, layout, and HTML generation.

Public API
----------
    inject_css()
    render_header()
    render_status_pill(is_running)
    render_idle_placeholder(container)
    render_ended_placeholder(container)
    render_stats(f_ph, s_ph, p_ph, e_ph, total_frames, persons, start, end)
    render_live_panels(container, persons)
    render_timelines(container, persons)
    render_session_report(container, persons, session_start, session_end)
    render_csv_history(container)
"""

import streamlit as st
import time
import textwrap

from config import (
    EMOTION_LABELS, DISPLAY_LABELS, EMOTION_EMOJIS, EMOTION_COLORS,
    PERSON_COLORS_HEX, PERSON_LABELS, CSV_PATH,
)
from backend import fmt_dur, get_dominant_emotion, load_csv


# ─────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────

def inject_css() -> None:
    """Inject all global styles into the Streamlit page."""
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0e;
    color: #e4e1da;
}
.main { background-color: #0a0a0e; }
.block-container { padding: 1.8rem 2.2rem 2rem; max-width: 1500px; }
h1,h2,h3 { font-family: 'Space Mono', monospace; }

/* Header */
.app-header {
    display:flex; align-items:baseline; gap:14px;
    margin-bottom:1.6rem; border-bottom:1px solid #22222a; padding-bottom:1rem;
}
.app-title { font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700;
             color:#f0ede6; letter-spacing:-.02em; margin:0; }
.app-sub   { font-size:.78rem; color:#555; letter-spacing:.07em; text-transform:uppercase; }

/* Status pill */
.status-pill {
    display:inline-flex; align-items:center; gap:7px;
    font-family:'Space Mono',monospace; font-size:.7rem; padding:4px 13px;
    border-radius:20px; font-weight:700; letter-spacing:.05em;
}
.status-active { background:#0f2e1a; color:#4ade80; border:1px solid #16532e; }
.status-idle   { background:#1a1a22; color:#555;    border:1px solid #2a2a34; }
.dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
.dot-active { background:#4ade80; animation:pulse 1.4s infinite; }
.dot-idle   { background:#444; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Buttons */
div.stButton > button {
    font-family:'Space Mono',monospace !important; font-size:.75rem !important;
    font-weight:700 !important; letter-spacing:.04em !important;
    border-radius:8px !important; padding:.5rem 1.2rem !important;
}
div.stButton > button:first-child {
    background:#16161f !important; border:1px solid #36364a !important; color:#ddd !important;
}
div.stButton > button:first-child:hover {
    background:#1e1e2a !important; border-color:#52526a !important;
}

/* Emotion card */
.emo-card {
    display:flex; align-items:center; gap:12px;
    background:#111118; border:1px solid #1e1e28;
    border-radius:10px; padding:.6rem .9rem; margin-bottom:.5rem;
}
.emo-snapshot {
    width:54px; height:54px; border-radius:6px;
    object-fit:cover; flex-shrink:0; border:1px solid #2a2a38;
}
.emo-snapshot-ph {
    width:54px; height:54px; border-radius:6px; background:#1a1a24;
    flex-shrink:0; display:flex; align-items:center; justify-content:center;
    font-size:1.4rem; border:1px solid #2a2a38;
}
.emo-info  { flex:1; min-width:0; }
.emo-label { font-weight:600; font-size:.82rem; margin-bottom:2px; }
.emo-meta  { font-size:.7rem; color:#666; font-family:'Space Mono',monospace; }
.emo-dur   { font-family:'Space Mono',monospace; font-size:.8rem;
             font-weight:700; color:#aaa; margin-left:auto; flex-shrink:0; }

/* Duration bars */
.dur-bar-wrap  { margin-bottom:.45rem; }
.dur-bar-label { display:flex; justify-content:space-between; font-size:.72rem;
                 color:#999; margin-bottom:3px; font-family:'Space Mono',monospace; }
.dur-bar-track { height:5px; background:#1a1a24; border-radius:3px; overflow:hidden; }
.dur-bar-fill  { height:100%; border-radius:3px; }

/* Summary / report card */
.summary-wrap { background:#0e0e16; border:1px solid #2a2a38;
                border-radius:14px; padding:1.4rem 1.6rem; margin-bottom:1rem; }

/* Live probability bars */
.prob-row   { display:flex; align-items:center; gap:8px; margin-bottom:.38rem; }
.prob-lbl   { font-size:.7rem; color:#888; width:52px; flex-shrink:0;
              font-family:'Space Mono',monospace; }
.prob-track { flex:1; height:5px; background:#1a1a24; border-radius:3px; overflow:hidden; }
.prob-fill  { height:100%; border-radius:3px; }
.prob-val   { font-size:.68rem; color:#555; width:30px; text-align:right;
              font-family:'Space Mono',monospace; flex-shrink:0; }

/* Stat boxes */
.stat-box { background:#111118; border:1px solid #22222a; border-radius:9px;
            padding:.8rem 1rem; text-align:center; }
.stat-num { font-family:'Space Mono',monospace; font-size:1.35rem; font-weight:700; color:#f0ede6; }
.stat-lbl { font-size:.65rem; color:#555; letter-spacing:.06em; text-transform:uppercase; margin-top:2px; }

/* Section labels */
.sec-lbl { font-family:'Space Mono',monospace; font-size:.65rem; color:#444;
           letter-spacing:.1em; text-transform:uppercase; margin-bottom:.7rem; }

/* Camera placeholder */
.cam-idle {
    background:#0e0e16; border:1px dashed #22222a; border-radius:12px;
    height:360px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:10px; color:#2a2a38;
}

/* CSV history table */
.csv-badge {
    display:inline-block; font-size:.68rem; padding:2px 8px; border-radius:12px;
    font-family:'Space Mono',monospace; font-weight:700;
}
.csv-dominant { background:#16301a; color:#4ade80; border:1px solid #1e4a24; }
.csv-normal   { background:#1a1a24; color:#888;    border:1px solid #2a2a38; }

#MainMenu, footer, header { visibility:hidden; }
.stDeployButton { display:none; }
</style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────

def render_header() -> None:
    """Render the top app title bar."""
    st.markdown("""
    <div class="app-header">
      <span class="app-title">EMOTION LENS</span>
      <span class="app-sub">Multi-person · real-time affect analysis</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STATUS PILL
# ─────────────────────────────────────────────────────────

def render_status_pill(is_running: bool) -> None:
    """Render an animated DETECTING / IDLE status badge."""
    if is_running:
        st.markdown(
            '<span class="status-pill status-active">'
            '<span class="dot dot-active"></span>DETECTING</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-idle">'
            '<span class="dot dot-idle"></span>IDLE</span>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────
# CAMERA PLACEHOLDERS
# ─────────────────────────────────────────────────────────

def render_idle_placeholder(container) -> None:
    """Shown in the camera area before the session starts."""
    container.markdown("""
    <div class="cam-idle">
      <div style="font-size:2.6rem">🎭</div>
      <div style="font-family:'Space Mono',monospace;font-size:.72rem;letter-spacing:.1em;">
        PRESS START TO BEGIN
      </div>
    </div>""", unsafe_allow_html=True)


def render_ended_placeholder(container) -> None:
    """Shown in the camera area after the session stops."""
    container.markdown("""
    <div class="cam-idle">
      <div style="font-size:2rem">⏹</div>
      <div style="font-family:'Space Mono',monospace;font-size:.72rem;
                  letter-spacing:.1em;color:#555;">SESSION ENDED — DATA SAVED TO CSV</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# STATS BAR
# ─────────────────────────────────────────────────────────

def render_stats(
    frames_ph, session_ph, persons_ph, emos_ph,
    total_frames: int,
    persons: dict,
    session_start,
    session_end,
) -> None:
    """Render the four stat boxes below the camera feed."""
    elapsed = 0
    if session_start:
        end     = session_end or time.time()
        elapsed = int(end - session_start)
    m, s       = divmod(elapsed, 60)
    total_emos = sum(len(p['timeline']) for p in persons.values())

    for ph, num, lbl in [
        (frames_ph,  total_frames,        "Frames"),
        (session_ph, f"{m:02d}:{s:02d}",  "Duration"),
        (persons_ph, len(persons),         "Persons"),
        (emos_ph,    total_emos,           "Emotions"),
    ]:
        ph.markdown(f"""
        <div class="stat-box">
          <div class="stat-num">{num}</div>
          <div class="stat-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HTML FRAGMENT BUILDERS (private)
# ─────────────────────────────────────────────────────────

def _prob_bars_html(probs, current_emotion: str, person_idx: int) -> str:
    """Build HTML for one person's live probability bar panel."""
    col_hex = PERSON_COLORS_HEX[person_idx % len(PERSON_COLORS_HEX)]
    ec      = EMOTION_COLORS.get(current_emotion, '#888')
    emoji   = EMOTION_EMOJIS.get(current_emotion, '😐')

    rows = ""
    for i, (lbl, prob) in enumerate(zip(DISPLAY_LABELS, probs)):
        ec2 = EMOTION_COLORS[EMOTION_LABELS[i]]
        rows += f"""
      <div class="prob-row">
        <span class="prob-lbl">{lbl[:4]}</span>
        <div class="prob-track">
          <div class="prob-fill" style="width:{prob*100:.0f}%;background:{ec2};"></div>
        </div>
        <span class="prob-val">{prob:.2f}</span>
      </div>"""

    return f"""
    <div style="background:#0e0e16;border:1px solid {col_hex}44;border-radius:10px;
                padding:.8rem 1rem;margin-bottom:.7rem;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:.7rem;">
        <div style="width:9px;height:9px;border-radius:50%;background:{col_hex};flex-shrink:0;"></div>
        <span style="font-family:'Space Mono',monospace;font-size:.72rem;
                     font-weight:700;color:{col_hex};">{PERSON_LABELS[person_idx]}</span>
        <span style="margin-left:auto;font-size:.95rem;">{emoji}</span>
        <span style="font-family:'Space Mono',monospace;font-size:.68rem;
                     color:{ec};margin-left:4px;">{current_emotion.upper()}</span>
      </div>
      {rows}
    </div>"""


def _emotion_card_html(entry: dict, duration: float) -> str:
    """Build one emotion card (snapshot + label + timestamp + duration)."""
    emo   = entry['emotion']
    ec    = EMOTION_COLORS[emo]
    emoji = EMOTION_EMOJIS[emo]
    snap  = entry.get('snapshot', '')
    ts    = entry.get('first_seen', '--:--:--')

    img_html = (
        f'<img class="emo-snapshot" src="data:image/png;base64,{snap}" />'
        if snap else f'<div class="emo-snapshot-ph">{emoji}</div>'
    )
    return f"""
    <div class="emo-card">
      {img_html}
      <div class="emo-info">
        <div class="emo-label" style="color:{ec};">{emoji} {emo.capitalize()}</div>
        <div class="emo-meta">First detected · {ts}</div>
      </div>
      <div class="emo-dur">{fmt_dur(duration)}</div>
    </div>"""


def _duration_bar_html(emo: str, dur: float, total: float) -> str:
    """Build one duration bar row."""
    pct = (dur / total * 100) if total > 0 else 0
    ec  = EMOTION_COLORS[emo]
    return f"""
    <div class="dur-bar-wrap">
      <div class="dur-bar-label">
        <span>{EMOTION_EMOJIS[emo]} {emo.capitalize()}</span>
        <span>{fmt_dur(dur)} &nbsp; {pct:.0f}%</span>
      </div>
      <div class="dur-bar-track">
        <div class="dur-bar-fill" style="width:{min(pct,100):.1f}%;background:{ec};"></div>
      </div>
    </div>"""


def _person_timeline_html(person_idx: int, person: dict) -> str:
    """Build the live timeline block for one person."""
    col_hex  = PERSON_COLORS_HEX[person_idx % len(PERSON_COLORS_HEX)]
    timeline = person['timeline']
    dur_map  = person['duration_map']

    cards = "".join(
        _emotion_card_html(e, dur_map.get(e['emotion'], 0.0))
        for e in timeline
    )
    return f"""
    <div style="background:#0e0e16;border:1px solid {col_hex}33;border-radius:12px;
                padding:.9rem 1rem;margin-bottom:.8rem;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:.8rem;
                  padding-bottom:.5rem;border-bottom:1px solid #1e1e28;">
        <div style="width:9px;height:9px;border-radius:50%;background:{col_hex};"></div>
        <span style="font-family:'Space Mono',monospace;font-size:.72rem;
                     font-weight:700;color:{col_hex};">{PERSON_LABELS[person_idx]}</span>
        <span style="font-size:.65rem;color:#555;margin-left:auto;">
          {len(timeline)} emotions detected
        </span>
      </div>
      {cards}
    </div>"""


def _person_report_html(person_idx: int, person: dict) -> str:
    """Build the complete post-session summary block for one person."""
    col_hex  = PERSON_COLORS_HEX[person_idx % len(PERSON_COLORS_HEX)]
    dur_map  = person['duration_map']
    timeline = person['timeline']
    total_p  = sum(dur_map.values())
    dominant = get_dominant_emotion(dur_map)
    dom_dur  = dur_map[dominant]
    dom_pct  = (dom_dur / total_p * 100) if total_p > 0 else 0

    # Duration breakdown bars (only emotions ≥ 0.3 s)
    bars = "".join(
        _duration_bar_html(emo, dur_map.get(emo, 0.0), total_p)
        for emo in EMOTION_LABELS
        if dur_map.get(emo, 0.0) >= 0.3
    )

    # Emotion cards
    cards = "".join(
        _emotion_card_html(e, dur_map.get(e['emotion'], 0.0))
        for e in timeline
    )

    return f"""
    <div class="summary-wrap" style="border-color:{col_hex}55;">

      <div style="display:flex;align-items:center;gap:10px;margin-bottom:1rem;
                  padding-bottom:.7rem;border-bottom:1px solid #1e1e28;">
        <div style="width:10px;height:10px;border-radius:50%;background:{col_hex};"></div>
        <span style="font-family:'Space Mono',monospace;font-weight:700;
                     font-size:.82rem;color:{col_hex};">{PERSON_LABELS[person_idx]}</span>
        <span style="font-size:.68rem;color:#555;margin-left:auto;">
          {len(timeline)} emotions &nbsp;·&nbsp; {fmt_dur(total_p)}
        </span>
      </div>

      <div style="text-align:center;padding:.5rem 0 1.1rem;">
        <div style="font-size:2.8rem;line-height:1;margin-bottom:.3rem;">{EMOTION_EMOJIS[dominant]}</div>
        <div style="font-family:'Space Mono',monospace;font-size:1.1rem;font-weight:700;
                    color:{EMOTION_COLORS[dominant]};">{dominant.upper()}</div>
        <div style="font-size:.68rem;color:#666;letter-spacing:.05em;text-transform:uppercase;margin-top:5px;">
          DOMINANT EMOTION &nbsp;·&nbsp; {fmt_dur(dom_dur)} &nbsp;·&nbsp; {dom_pct:.0f}% of session
        </div>
      </div>

      <div style="margin-bottom:1.1rem;">
        <div style="font-family:'Space Mono',monospace;font-size:.62rem;color:#444;
                    letter-spacing:.08em;text-transform:uppercase;margin-bottom:.6rem;">Time breakdown</div>
        {bars}
      </div>

      <div style="font-family:'Space Mono',monospace;font-size:.62rem;color:#444;
                  letter-spacing:.08em;text-transform:uppercase;margin-bottom:.7rem;">Captured emotions</div>
      {cards}
    </div>"""


# ─────────────────────────────────────────────────────────
# PUBLIC RENDER FUNCTIONS
# ─────────────────────────────────────────────────────────

def render_live_panels(container, persons: dict) -> None:
    """Render live probability bar panels for all detected persons."""
    if not persons:
        container.markdown(
            '<div style="color:#333;font-size:.78rem;padding:.8rem 0">'
            'No faces detected yet…</div>',
            unsafe_allow_html=True,
        )
        return
    html = "".join(
        _prob_bars_html(
            persons[p]['last_probs'],
            persons[p]['current_emotion'] or 'neutral',
            p,
        )
        for p in sorted(persons)
    )
    container.markdown(html, unsafe_allow_html=True)


def render_timelines(container, persons: dict) -> None:
    """Render emotion timeline cards for all detected persons."""
    html = "".join(
        _person_timeline_html(p, persons[p])
        for p in sorted(persons)
        if persons[p]['timeline']
    )
    if html:
        container.markdown(html, unsafe_allow_html=True)


def render_session_report(
    container,
    persons: dict,
    session_start,
    session_end,
) -> None:
    """Render the full post-session summary shown after STOP."""
    if not persons:
        container.info("No data captured in this session.")
        return

    total_sec = (session_end - session_start) if (session_start and session_end) else 0

    container.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:.68rem;color:#555;
                letter-spacing:.1em;text-transform:uppercase;margin-bottom:1.2rem;
                border-bottom:1px solid #1e1e28;padding-bottom:.6rem;">
      SESSION REPORT &nbsp;·&nbsp; {fmt_dur(total_sec)} &nbsp;·&nbsp; {len(persons)} person(s)
    </div>""", unsafe_allow_html=True)

    for p_idx in sorted(persons):
        container.markdown(_person_report_html(p_idx, persons[p_idx]), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# CSV HISTORY TAB
# ─────────────────────────────────────────────────────────



import streamlit as st

def render_csv_history(container) -> None:
    """
    Display all past sessions loaded from the CSV file.
    """

    df = load_csv()

    # ── Empty state ─────────────────────
    if df is None:
        container.markdown(
            "<div style='background:#0e0e16;border:1px dashed #22222a;border-radius:12px;"
            "padding:2rem;text-align:center;color:#444;'>"
            "<div style='font-size:2rem;margin-bottom:.5rem'>📄</div>"
            "<div style='font-family:\"Space Mono\",monospace;font-size:.72rem;letter-spacing:.1em;'>"
            "NO SESSION DATA YET</div>"
            "<div style='font-size:.76rem;color:#333;margin-top:.5rem;'>"
            "Run a detection session to start recording data.</div>"
            "</div>",
            unsafe_allow_html=True
        )
        return

    # ── Header ─────────────────────────
    container.markdown(
        "<div class='sec-lbl'>Session History</div>",
        unsafe_allow_html=True
    )

    sessions = df.groupby('session_id')

    # ── Sessions loop ──────────────────
    for sid, group in sessions:
        date         = group['session_date'].iloc[0]
        sess_dur     = group['session_duration_sec'].iloc[0]
        persons_seen = group['person_label'].unique().tolist()

        person_html = ""

        for person in sorted(persons_seen):
            p_rows  = group[group['person_label'] == person]
            dom_row = p_rows[p_rows['dominant'] == True]

            dom_emo = dom_row['emotion'].iloc[0] if not dom_row.empty else 'neutral'
            dom_dur = dom_row['duration_sec'].iloc[0] if not dom_row.empty else 0

            ec    = EMOTION_COLORS.get(dom_emo, '#888')
            emoji = EMOTION_EMOJIS.get(dom_emo, '😐')

            # ── Emotion bars ──
            emo_bars = ""
            total_dur = p_rows['duration_sec'].sum()

            for _, row in p_rows.iterrows():
                emo = row['emotion']
                dur = row['duration_sec']

                pct = (dur / total_dur * 100) if total_dur > 0 else 0
                ec2 = EMOTION_COLORS.get(emo, '#888')

                dom_badge = (
                    "<span class='csv-badge csv-dominant'>DOMINANT</span>"
                    if row['dominant'] else ""
                )

                emo_bars += (
                    "<div style='display:flex;align-items:center;gap:8px;margin-bottom:.3rem;'>"

                    f"<span style='font-size:.7rem;color:{ec2};width:68px;"
                    f"font-family:\"Space Mono\",monospace;'>"
                    f"{EMOTION_EMOJIS.get(emo,'')} {emo.capitalize()[:7]}</span>"

                    "<div style='flex:1;height:4px;background:#1a1a24;"
                    "border-radius:2px;overflow:hidden;'>"

                    f"<div style='width:{min(pct,100):.1f}%;height:100%;"
                    f"background:{ec2};border-radius:2px;'></div></div>"

                    f"<span style='font-family:\"Space Mono\",monospace;font-size:.64rem;"
                    f"color:#666;width:32px;'>{fmt_dur(dur)}</span>"

                    f"{dom_badge}</div>"
                )

            # ── Person card ──
            person_html += (
                "<div style='background:#111118;border:1px solid #1e1e28;"
                "border-radius:9px;padding:.7rem .9rem;margin-bottom:.5rem;'>"

                "<div style='display:flex;align-items:center;gap:8px;margin-bottom:.6rem;'>"

                f"<span style='font-family:\"Space Mono\",monospace;font-size:.7rem;"
                f"font-weight:700;color:#aaa;'>{person}</span>"

                f"<span style='font-size:.85rem;'>{emoji}</span>"

                f"<span style='font-size:.7rem;color:{ec};"
                f"font-family:\"Space Mono\",monospace;'>{dom_emo.upper()}</span>"

                f"<span style='margin-left:auto;font-size:.65rem;color:#444;"
                f"font-family:\"Space Mono\",monospace;'>{fmt_dur(dom_dur)}</span>"

                "</div>"

                f"{emo_bars}"

                "</div>"
            )

        # ── Session card ──
        session_html = (
            "<div style='background:#0e0e16;border:1px solid #2a2a38;"
            "border-radius:12px;padding:1rem 1.1rem;margin-bottom:.8rem;'>"

            "<div style='display:flex;align-items:center;gap:10px;margin-bottom:.8rem;"
            "padding-bottom:.5rem;border-bottom:1px solid #1e1e28;'>"

            f"<span style='font-family:\"Space Mono\",monospace;font-size:.72rem;"
            f"font-weight:700;color:#f0ede6;'>#{sid}</span>"

            f"<span style='font-size:.68rem;color:#555;'>{date}</span>"

            f"<span style='margin-left:auto;font-size:.68rem;color:#666;"
            f"font-family:\"Space Mono\",monospace;'>"
            f"{fmt_dur(sess_dur)} · {len(persons_seen)} person(s)</span>"

            "</div>"

            f"{person_html}"

            "</div>"
        )

        container.markdown(session_html, unsafe_allow_html=True)

    # ── Raw Data Section ──
    with container.expander("📊 Raw CSV Data"):
        st.dataframe(df, width='stretch', hide_index=True)

        csv_bytes = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download CSV",
            data=csv_bytes,
            file_name="emotion_sessions.csv",
            mime="text/csv",
            width='stretch',
        )











