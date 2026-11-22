import streamlit as st
from ui.styles import inject_hud_css
from ui.tab_major import render_major_hub
from ui.tab_tournaments import render_other_tournaments
from ui.tab_fantasy import render_fantasy_hub
from ui.tab_data_lab import render_data_lab

st.set_page_config(
    page_title="NeuralMajorPick V2 // CS2 AI & Pick'Em HUD",
    page_icon="🎯",
    layout="wide"
)

inject_hud_css()

st.markdown("""
<div class="hud-header">
    <h2 style="margin:0; color:#de9b35;">🎯 NEURAL MAJOR PICK // V2.0 HUD</h2>
    <p style="margin:0; color:#8b949e; font-size:14px;">
        Valve Major VRS + Buchholz Engine • BO1/BO3/BO5 Context • Playoffs Exact Solver • HLTV Pick'Em Bracket
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