"""
app.py 

Streamlit entry point. Only handles:
  - Page config
  - Session state initialisation
  - Layout + control buttons
  - Detection loop (camera → backend → frontend)
  - CSV save trigger on STOP

Run:
    streamlit run app.py

Project structure:
    app.py        ← you are here  (orchestration)
    backend.py    ← ML inference, tracking, CSV storage
    frontend.py   ← all UI rendering
    config.py     ← constants and hyperparameters

Install:
    pip install streamlit==1.28.0 deepface mediapipe opencv-python tf-keras numpy pillow pandas
"""

import cv2
import streamlit as st
import time

from config import CSV_PATH
from backend import (
    get_detectors,
    clear_inference_cache,
    process_frame,
    close_current_emotion,
    save_session_to_csv,
    generate_session_id,
)
from frontend import (
    inject_css,
    render_header,
    render_status_pill,
    render_idle_placeholder,
    render_ended_placeholder,
    render_stats,
    render_live_panels,
    render_timelines,
    render_session_report,
    render_csv_history,
)

# PAGE CONFIG  (must be first Streamlit call)

st.set_page_config(
    page_title="Emotion Lens",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# STREAMLIT VERSION COMPAT

def _rerun() -> None:
    # Call st.rerun() or st.experimental_rerun() depending on version.
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# SESSION STATE


def init_state() -> None:
    # Initialise all session state keys to their default values.
    defaults = {
        'running':       False,
        'session_start': None,
        'session_end':   None,
        'session_id':    None,
        'total_frames':  0,
        'persons':       {},          # dict[int → person_state]  (see backend.make_person)
        'show_report':   False,
        'csv_saved':     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# CACHED RESOURCES


@st.cache_resource
def load_detectors():
    # Load MediaPipe detectors once; cached for the lifetime of the app process.
    return get_detectors()

face_detector, face_mesher, _ = load_detectors()


# CONTROL ACTIONS


def on_start() -> None:
    # Reset all state and begin a new detection session.
    st.session_state.running       = True
    st.session_state.session_start = time.time()
    st.session_state.session_end   = None
    st.session_state.session_id    = generate_session_id()
    st.session_state.total_frames  = 0
    st.session_state.persons       = {}
    st.session_state.show_report   = False
    st.session_state.csv_saved     = False
    clear_inference_cache()


def on_stop() -> None:
    # Stop detection, close all open segments, save CSV, trigger report.
    now = time.time()
    for person in st.session_state.persons.values():
        close_current_emotion(person, now)

    st.session_state.running     = False
    st.session_state.session_end = now
    st.session_state.show_report = True

    # Save to CSV
    if st.session_state.persons and not st.session_state.csv_saved:
        save_session_to_csv(
            persons       = st.session_state.persons,
            session_id    = st.session_state.session_id,
            session_start = st.session_state.session_start,
            session_end   = now,
        )
        st.session_state.csv_saved = True


def on_clear() -> None:
    """Wipe all session state and reinitialise defaults."""
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    clear_inference_cache()
    init_state()


# UI SETUP


inject_css()
render_header()

# Tab layout 
tab_live, tab_history = st.tabs(["🎥  Live Detection", "📂  Session History"])


# TAB 1  LIVE DETECTION

with tab_live:
    col_feed, col_right = st.columns([3, 2], gap="large")

    #  Left: camera + controls 
    with col_feed:

        # Control row
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            render_status_pill(st.session_state.running)
        with c2:
            btn_label = "⏹  STOP" if st.session_state.running else "▶  START"
            if st.button(btn_label, width='stretch'):
                if st.session_state.running:
                    on_stop()
                else:
                    on_start()
                _rerun()
        with c3:
            if st.button("↺  CLEAR", width='stretch'):
                on_clear()
                _rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Camera frame placeholder
        frame_ph = st.empty()

        # CSV save notice (shown briefly after stop)
        csv_notice_ph = st.empty()

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # Stats row
        s1, s2, s3, s4 = st.columns(4)
        frames_ph  = s1.empty()
        session_ph = s2.empty()
        persons_ph = s3.empty()
        emos_ph    = s4.empty()

    #  Right: live panels or session report 
    with col_right:
        right_ph = st.empty()

    # Idle state 
    if not st.session_state.running and not st.session_state.show_report:
        render_idle_placeholder(frame_ph)
        with right_ph.container():
            st.markdown(
                '<div style="color:#333;font-size:.78rem;padding:.8rem 0">'
                'Waiting for session to start…</div>',
                unsafe_allow_html=True,
            )
        render_stats(
            frames_ph, session_ph, persons_ph, emos_ph,
            st.session_state.total_frames,
            st.session_state.persons,
            st.session_state.session_start,
            st.session_state.session_end,
        )

    # Post-stop report 
    if st.session_state.show_report and not st.session_state.running:
        render_ended_placeholder(frame_ph)
        if st.session_state.csv_saved:
            csv_notice_ph.success(
                f"✅ Session **{st.session_state.session_id}** saved to `{CSV_PATH}`"
            )
        render_stats(
            frames_ph, session_ph, persons_ph, emos_ph,
            st.session_state.total_frames,
            st.session_state.persons,
            st.session_state.session_start,
            st.session_state.session_end,
        )
        with right_ph.container():
            render_session_report(
                st,
                st.session_state.persons,
                st.session_state.session_start,
                st.session_state.session_end,
            )

    # Detection loop 
    if st.session_state.running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam. Check camera permissions.")
            st.session_state.running = False
            st.stop()

        try:
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)   # mirror for natural feel
                now   = time.time()

                # Run inference + draw overlays
                annotated = process_frame(
                    frame,
                    now,
                    st.session_state.persons,
                    face_detector,
                    face_mesher,
                )
                st.session_state.total_frames += 1

                # Display annotated frame
                frame_ph.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    width='stretch',
                )

                # Update right panel
                with right_ph.container():
                    st.markdown("<div class='sec-lbl'>Live — Per Person</div>",
                                unsafe_allow_html=True)
                    live_c = st.empty()
                    render_live_panels(live_c.container(), st.session_state.persons)

                    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='sec-lbl'>Emotion Timeline</div>",
                                unsafe_allow_html=True)
                    tl_c = st.empty()
                    render_timelines(tl_c.container(), st.session_state.persons)

                # Update stats
                render_stats(
                    frames_ph, session_ph, persons_ph, emos_ph,
                    st.session_state.total_frames,
                    st.session_state.persons,
                    st.session_state.session_start,
                    st.session_state.session_end,
                )

        finally:
            cap.release()


# TAB 2 SESSION HISTORY


with tab_history:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    container = st.container()
    render_csv_history(container)
