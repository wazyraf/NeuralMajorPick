import streamlit as st
from ui.styles import inject_hud_css
from ui.tab_major import render_major_hub
from ui.tab_tournaments import render_other_tournaments
from ui.tab_fantasy import render_fantasy_hub
from ui.tab_data_lab import render_data_lab

DEFAULT_BLIND_CUTOFF_DATE = "2026-06-02"
DEFAULT_TEAM_STATS_AS_OF = "2026-06-02"


def _init_session_defaults():
    """Inițializează parametrii impliciți pentru blind backtest și filtrarea cronologică."""
    if "blind_test_enabled" not in st.session_state:
        st.session_state["blind_test_enabled"] = True
    if "cutoff_date" not in st.session_state:
        st.session_state["cutoff_date"] = DEFAULT_BLIND_CUTOFF_DATE
    if "team_stats_as_of" not in st.session_state:
        st.session_state["team_stats_as_of"] = DEFAULT_TEAM_STATS_AS_OF


st.set_page_config(
    page_title="NeuralMajorPick V2 // CS2 AI & Pick'Em HUD",
    page_icon="🎯",
    layout="wide"
)

_init_session_defaults()
inject_hud_css()

active_cutoff = (
    str(st.session_state.get("cutoff_date", DEFAULT_BLIND_CUTOFF_DATE))
    if st.session_state.get("blind_test_enabled", True)
    else None
)
active_stats_as_of = (
    str(st.session_state.get("team_stats_as_of", DEFAULT_TEAM_STATS_AS_OF))
    if st.session_state.get("blind_test_enabled", True)
    else None
)

mode_label = (
    f"BLIND TEST CUTOFF: < {active_cutoff}"
    if active_cutoff
    else "FULL DATASET (NO CUTOFF)"
)

st.markdown(f"""
<div class="hud-header">
    <h2 style="margin:0; color:#de9b35;">🎯 NEURAL MAJOR PICK // V2.0 HUD</h2>
    <p style="margin:0; color:#8b949e; font-size:14px;">
        Valve Major VRS + Buchholz Engine • BO1/BO3/BO5 Context • Playoffs Exact Solver • HLTV Pick'Em Bracket • <span style="color:#de9b35;">{mode_label}</span>
    </p>
</div>
""", unsafe_allow_html=True)

tab_major, tab_other, tab_fantasy, tab_lab = st.tabs([
    "🏆 MAJOR HUB (PICK'EM • MATCHES • PLAYOFFS)",
    "🌐 ALTE TURNEE (PGL • IEM)",
    "🎮 HLTV FANTASY",
    "🧪 DATA & MODEL LAB"
])

with tab_major:
    render_major_hub()

with tab_other:
    render_other_tournaments()

with tab_fantasy:
    render_fantasy_hub()

with tab_lab:
    render_data_lab()