import streamlit as st
import pandas as pd


def render_fantasy_hub():
    st.markdown("### 🎮 HLTV Fantasy League Optimizer ($1,000,000 Budget)")
    st.info(
        "🚧 **Modul dedicat HLTV Fantasy (pregătit pentru extindere post-Major):** "
        "După ce adăugăm statisticile per jucător (`Rating 2.1`, `ADR`, `KPR`, `Role`), acest fișier (`ui/tab_fantasy.py`) "
        "va rula un optimizator *Knapsack* pentru echipa optimă de 5 jucători în bugetul de $1.000.000."
    )

    sample_players = pd.DataFrame({
        'Player': ['ZywOo', 'donk', 'm0NESY', 'b1t', 'ropz'],
        'Team': ['Vitality', 'Spirit', 'Falcons', 'NaVi', 'Vitality'],
        'Role': ['AWPer / Star', 'Entry / Star', 'AWPer', 'Rifler', 'Lurker'],
        'Price ($)': [235000, 232000, 218000, 165000, 148000],
        'Est. Fantasy Pts': [84.5, 86.0, 79.2, 66.0, 64.5]
    })

    st.dataframe(sample_players, use_container_width=True)
    total_cost = int(sample_players['Price ($)'].sum())
    st.caption(f"Buget exemplu folosit: **${total_cost:,} / $1,000,000** (Rămas: **${1_000_000 - total_cost:,}**)")