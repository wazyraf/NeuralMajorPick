import streamlit as st
import pandas as pd
from core.model_engine import load_predictor, predict_match_context
from core.swiss_sim import simulate_swiss_stage


def render_other_tournaments():
    model, scaler = load_predictor()
    if model is None:
        st.warning("⚠️ Antrenează mai întâi rețeaua în tab-ul **🧪 DATA & MODEL LAB**!")
        return

    t_pgl, t_iem = st.tabs([
        "🌐 PGL / GENERIC SWISS",
        "🏟️ IEM GROUPS (GSL DOUBLE-ELIM)"
    ])

    with t_pgl:
        st.markdown("### 🌐 Simulator Turnee PGL / S-Tier Swiss (All BO3)")
        st.caption("Folosește motorul VRS + Buchholz, configurat implicit pentru turnee S-Tier LAN (non-Major).")
        pgl_df = pd.DataFrame({
            'team': [f"Team {i}" for i in range(1, 17)],
            'rank': list(range(1, 17)),
            'wr_3m': [round(0.65 - i * 0.01, 2) for i in range(16)]
        })
        edited_pgl = st.data_editor(pgl_df, num_rows="fixed", use_container_width=True, key="pgl_editor")
        if st.button("🚀 Simulează Turneu PGL Swiss", type="primary"):
            res = simulate_swiss_stage(edited_pgl, model, scaler, n_sims=5000, stage_mode="Stage 3 (All BO3)", is_major=False)
            st.dataframe(
                res.style.format({
                    '3-0 (%)': '{:.1f}%',
                    'Calificare Total (%)': '{:.1f}%',
                    '3-1 / 3-2 (%)': '{:.1f}%',
                    '0-3 (%)': '{:.1f}%'
                }),
                use_container_width=True
            )

    with t_iem:
        st.markdown("### 🏟️ IEM Group Stage (GSL Double-Elimination)")
        st.info("🔧 **Modul pregătit pentru sezonul următor:** Simulare grupe IEM de 8 echipe (Upper Bracket -> Upper Final + Lower Bracket -> Lower Final, toate BO3).")
        iem_df = pd.DataFrame({
            'seed': [f"GSL Seed {i}" for i in range(1, 9)],
            'team': ['Spirit', 'Vitality', 'MOUZ', 'G2', 'FaZe', 'NaVi', 'Aurora', 'FURIA'],
            'rank': [3, 1, 7, 13, 8, 2, 6, 5],
            'wr_3m': [0.70, 0.81, 0.59, 0.56, 0.60, 0.63, 0.56, 0.55]
        })
        edited_iem = st.data_editor(iem_df, num_rows="fixed", use_container_width=True, disabled=['seed'])
        if st.button("🏟️ Calculează Opening Matches (IEM Group)"):
            pairs = [(0, 7), (3, 4), (1, 6), (2, 5)]
            for idx, (a, b) in enumerate(pairs):
                r = predict_match_context(
                    model, scaler,
                    int(edited_iem.loc[a, 'rank']), int(edited_iem.loc[b, 'rank']),
                    float(edited_iem.loc[a, 'wr_3m']), float(edited_iem.loc[b, 'wr_3m']),
                    "BO3", "S-Tier LAN"
                )
                st.write(
                    f"**Opening Match {idx+1} (BO3):** {edited_iem.loc[a, 'team']} (**{r['prob_a']*100:.1f}%**) "
                    f"vs {edited_iem.loc[b, 'team']} (**{r['prob_b']*100:.1f}%**) → Scor probabil: `{r['predicted_score']}`"
                )