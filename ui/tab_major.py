# Hub-ul principal Major: Pick'Em Bracket + Match-by-Match (Map Veto/Score/Rounds) + Playoffs
import streamlit as st
import pandas as pd
from core.model_engine import load_predictor, predict_match_context
from core.swiss_sim import simulate_swiss_stage, REGION_SOS_DEFAULT
from core.playoff_sim import simulate_playoffs_exact
from core.team_stats_engine import load_team_stats, compute_map_veto_edge, assess_match_risk
from ui.styles import render_hltv_pickem_bracket

try:
    from core.model_engine import predict_match_with_team_stats
except ImportError:
    predict_match_with_team_stats = None


def render_major_hub():
    model, scaler = load_predictor()
    if model is None:
        st.warning("⚠️ Modelul nu este încă antrenat! Mergi în tab-ul **🧪 DATA & MODEL LAB** și apasă **ANTRENEAZĂ REȚEAUA**.")
        return

    sub_pickem, sub_match, sub_playoffs = st.tabs([
        "🏆 MAJOR PICK'EM (SWISS VRS + BUCHHOLZ)",
        "⚔️ MATCH-BY-MATCH & MAP ANALYZER",
        "👑 MAJOR PLAYOFFS BRACKET"
    ])

    # =========================================================
    # SUB-TAB 1: MAJOR PICK'EM (SWISS)
    # =========================================================
    with sub_pickem:
        st.markdown("### 🏆 Valve Major Swiss Pick'Em Engine")
        st.caption("Ordinea rândurilor 1–16 reprezintă **VRS Seeding-ul oficial** (Runda 1: 1 vs 9 ... 8 vs 16; Rundele 2–5: Buchholz + VRS Seed).")

        c_cfg1, c_cfg2, c_cfg3 = st.columns([2, 1, 1])
        with c_cfg1:
            stage_mode = st.selectbox(
                "Etapa Majorului (Format meciuri)",
                [
                    "Stage 1 / Stage 2 (BO1 + Decisive BO3)",
                    "Stage 3 (All BO3)"
                ],
                index=0
            )
        with c_cfg2:
            n_sims = st.selectbox("Simulări Monte Carlo", [2000, 5000, 10000], index=2)
        with c_cfg3:
            use_sos = st.checkbox("Ajustare Regiune / SoS", value=True, help="Penalizează ușor winrate-ul umflat din regiuni/turnee mai accesibile.")

        default_teams = pd.DataFrame({
            'team': [
                'Liquid', 'GamerLegion', '3DMAX', 'Astralis',
                'paiN', 'MIBR', 'B8', 'Imperial',
                'M80', 'Legacy', 'TYLOO', 'Lynn Vision',
                'FlyQuest', 'NRG', 'Wildcard', 'Rare Atom'
            ],
            'rank': [12, 14, 15, 16, 18, 19, 21, 23, 25, 26, 28, 30, 32, 34, 38, 45],
            'wr_3m': [0.58, 0.55, 0.56, 0.57, 0.59, 0.54, 0.53, 0.60, 0.57, 0.61, 0.64, 0.66, 0.52, 0.55, 0.51, 0.49],
            'region': [
                'Americas', 'Europe (Tier 1/1.5)', 'Europe (Tier 1/1.5)', 'Europe (Tier 1/1.5)',
                'Americas', 'Americas', 'Europe (Tier 1/1.5)', 'Americas',
                'Americas', 'Americas', 'Asia / Oceania', 'Asia / Oceania',
                'Asia / Oceania', 'Americas', 'Americas', 'Asia / Oceania'
            ]
        })

        edited_df = st.data_editor(
            default_teams,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "region": st.column_config.SelectboxColumn(
                    "Regiune / Nivel Turnee",
                    options=list(REGION_SOS_DEFAULT.keys()),
                    required=True
                )
            }
        )

        if st.button("🚀 GENEREAZĂ BRACKET PICK'EM", type="primary", use_container_width=True):
            sim_input = edited_df.copy()
            if not use_sos:
                sim_input['region'] = "Europe (Tier 1/1.5)"

            with st.spinner(f"Rulăm {n_sims} simulări oficiale Valve Major (VRS + Buchholz)..."):
                sim_res = simulate_swiss_stage(sim_input, model, scaler, n_sims=n_sims, stage_mode=stage_mode, is_major=True)

            df_3_0 = sim_res.sort_values('3-0 (%)', ascending=False).head(2)
            df_0_3 = sim_res.sort_values('0-3 (%)', ascending=False).head(2)
            excluded = set(df_3_0['Echipa']).union(set(df_0_3['Echipa']))
            df_adv = sim_res[~sim_res['Echipa'].isin(excluded)].sort_values('Calificare Total (%)', ascending=False).head(6)

            st.markdown("#### 🎯 HLTV Visual Pick'Em Bracket")
            render_hltv_pickem_bracket(df_3_0, df_adv, df_0_3)

            export_txt = (
                f"=== NEURAL MAJOR PICK ({stage_mode}) ===\n"
                f"3:0 -> {', '.join(df_3_0['Echipa'].tolist())}\n"
                f"3:1 / 3:2 -> {', '.join(df_adv['Echipa'].tolist())}\n"
                f"0:3 -> {', '.join(df_0_3['Echipa'].tolist())}"
            )
            st.download_button("📥 Export Pick'Em (.txt)", export_txt, file_name="major_pickem.txt")

            with st.expander("📊 Tabel Complet Probabilități (Toate cele 16 echipe)", expanded=True):
                st.dataframe(
                    sim_res.style.format({
                        '3-0 (%)': '{:.1f}%',
                        'Calificare Total (%)': '{:.1f}%',
                        '3-1 / 3-2 (%)': '{:.1f}%',
                        '0-3 (%)': '{:.1f}%'
                    }),
                    use_container_width=True
                )

    # =========================================================
    # SUB-TAB 2: MATCH-BY-MATCH & MAP ANALYZER
    # =========================================================
    with sub_match:
        st.markdown("### ⚔️ Analizor Meci Individual (Stage / Map Edge / H2H / Scor)")

        as_of_cutoff = st.text_input(
            "Dată limită snapshot-uri `team_stats.csv` (opțional, ex. `2026-06-02` pentru Blind Backtest)",
            value="",
            key="h2h_as_of_date"
        ).strip() or None

        team_stats_map = load_team_stats("data/team_stats.csv", as_of_date=as_of_cutoff)

        c1, c_mid, c2 = st.columns([2, 1.3, 2])

        with c1:
            st.subheader("🔵 TEAM A")
            team_a = st.text_input("Echipă A", value="Vitality")
            sa = team_stats_map.get(team_a.strip().lower())
            rank_a = st.number_input("HLTV Rank A", 1, 200, 1)
            default_wr_a = float(sa["top20_wr"]) if sa else 0.78
            wr_a = st.slider("3M Winrate A", 0.0, 1.0, round(default_wr_a, 2), 0.01)
            reg_a = st.selectbox("Context Turnee A", list(REGION_SOS_DEFAULT.keys()), index=0, key="reg_a")
            if sa:
                st.caption(f"✅ Stats A: Top20 WR `{sa['top20_wr']*100:.1f}%` ({sa['top20_maps']}/{sa['maps_played']} maps) | Eco K/D `{sa['eco_kd']}` (`{sa['eco_per_map']}` eco/map)")
            else:
                st.caption("⚠️ Team A nu e în `team_stats.csv` — folosește valorile manuale.")

        with c2:
            st.subheader("🟠 TEAM B")
            team_b = st.text_input("Echipă B", value="Falcons")
            sb = team_stats_map.get(team_b.strip().lower())
            rank_b = st.number_input("HLTV Rank B", 1, 200, 4)
            default_wr_b = float(sb["top20_wr"]) if sb else 0.64
            wr_b = st.slider("3M Winrate B", 0.0, 1.0, round(default_wr_b, 2), 0.01)
            reg_b = st.selectbox("Context Turnee B", list(REGION_SOS_DEFAULT.keys()), index=0, key="reg_b")
            if sb:
                st.caption(f"✅ Stats B: Top20 WR `{sb['top20_wr']*100:.1f}%` ({sb['top20_maps']}/{sb['maps_played']} maps) | Eco K/D `{sb['eco_kd']}` (`{sb['eco_per_map']}` eco/map)")
            else:
                st.caption("⚠️ Team B nu e în `team_stats.csv` — folosește valorile manuale.")

        auto_veto = round(compute_map_veto_edge(sa, sb), 2)

        with c_mid:
            st.subheader("⚙️ CONTEXT")
            stage_ctx = st.selectbox("Fază Turneu", ["Stage 1", "Stage 2", "Stage 3", "Playoffs (QF/SF)", "Grand Final"])
            default_fmt_idx = 2 if stage_ctx == "Grand Final" else (1 if stage_ctx in ["Stage 3", "Playoffs (QF/SF)"] else 0)
            match_fmt = st.selectbox("Format", ["BO1", "BO3", "BO5"], index=default_fmt_idx)
            map_edge = st.slider("Avantaj Hartă / Veto (-0.25 B .. +0.25 A)", -0.25, 0.25, auto_veto, 0.01)
            h2h_edge = st.slider("Avantaj H2H Recent (-0.20 B .. +0.20 A)", -0.20, 0.20, 0.00, 0.01)
            calc_btn = st.button("🔥 CALCULEAZĂ MECIUL", type="primary", use_container_width=True)

        if calc_btn:
            if predict_match_with_team_stats is not None:
                res = predict_match_with_team_stats(
                    model=model,
                    scaler=scaler,
                    team_a=team_a,
                    team_b=team_b,
                    rank_a=rank_a,
                    rank_b=rank_b,
                    fallback_wr_a=wr_a,
                    fallback_wr_b=wr_b,
                    match_format=match_fmt,
                    event_tier="Major",
                    sos_adj_a=REGION_SOS_DEFAULT[reg_a],
                    sos_adj_b=REGION_SOS_DEFAULT[reg_b],
                    as_of_date=as_of_cutoff,
                    manual_map_edge=map_edge,
                    h2h_edge_a=h2h_edge,
                )
            else:
                res = predict_match_context(
                    model, scaler, rank_a, rank_b, wr_a, wr_b,
                    match_format=match_fmt,
                    event_tier="Major",
                    sos_adj_a=REGION_SOS_DEFAULT[reg_a],
                    sos_adj_b=REGION_SOS_DEFAULT[reg_b],
                    map_edge_a=map_edge,
                    h2h_edge_a=h2h_edge
                )

            if "upset_risk_pct" not in res or "volatility_score" not in res:
                risk_info = assess_match_risk(
                    prob_a=res["prob_a"],
                    match_format=match_fmt,
                    p_map=res.get("p_map"),
                    eco_kd_a=sa.get("eco_kd") if sa else None,
                    eco_kd_b=sb.get("eco_kd") if sb else None,
                    eco_per_map_a=sa.get("eco_per_map") if sa else None,
                    eco_per_map_b=sb.get("eco_per_map") if sb else None,
                    reliability_a=sa.get("reliability", 0.7) if sa else 0.7,
                    reliability_b=sb.get("reliability", 0.7) if sb else 0.7,
                )
                res.update(risk_info)

            r1, r2, r3 = st.columns([2, 1.2, 2])
            with r1:
                st.markdown(f"""
                <div class="prob-box">
                    <div class="team-name">{team_a.upper()}</div>
                    <div class="prob-val-a">{res['prob_a']*100:.1f}%</div>
                    <div style="color:#8b949e;">Map Win Prob: {res['p_map']*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="prob-box" style="border-color:#de9b35;">
                    <div style="color:#8b949e; font-size:12px;">SCOR PROBABIL ({match_fmt})</div>
                    <div style="font-size:32px; font-weight:900; color:#e6edf3; margin:10px 0;">{res['predicted_score']}</div>
                    <div style="color:#de9b35; font-size:12px;">{stage_ctx}</div>
                </div>""", unsafe_allow_html=True)
            with r3:
                st.markdown(f"""
                <div class="prob-box">
                    <div class="team-name">{team_b.upper()}</div>
                    <div class="prob-val-b">{res['prob_b']*100:.1f}%</div>
                    <div style="color:#8b949e;">Map Win Prob: {(1-res['p_map'])*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("⚡ Upset Risk", f"{res['upset_risk_pct']:.1f}%")
            m2.metric("🌊 Volatility Score", f"{res['volatility_score']:.1f} / 100")
            m3.metric("🛡️ Trust Level", f"{res['badge']} — {res['trust_label']}")

    # =========================================================
    # SUB-TAB 3: MAJOR PLAYOFFS BRACKET (QF BO3 -> SF BO3 -> GF BO5)
    # =========================================================
    with sub_playoffs:
        st.markdown("### 👑 Major Playoffs Single-Elimination Bracket")
        st.caption("Introdu cele 8 echipe calificate în ordinea perechilor din Sferturi: **(1-2) QF1**, **(3-4) QF2**, **(5-6) QF3**, **(7-8) QF4**. QF & SF sunt BO3, Grand Final este BO5.")

        default_qf = pd.DataFrame({
            'slot': ['QF1 - Top', 'QF1 - Bot', 'QF2 - Top', 'QF2 - Bot', 'QF3 - Top', 'QF3 - Bot', 'QF4 - Top', 'QF4 - Bot'],
            'team': ['9z', 'FURIA', 'Aurora', 'BETBOOM', 'G2', 'Spirit', 'Falcons', 'Vitality'],
            'rank': [21, 5, 6, 17, 13, 3, 4, 1],
            'wr_3m': [0.64, 0.55, 0.56, 0.59, 0.56, 0.70, 0.63, 0.81]
        })

        edited_qf = st.data_editor(default_qf, num_rows="fixed", use_container_width=True, disabled=['slot'])

        if st.button("👑 SIMULEAZĂ PLAYOFFS (EXACT)", type="primary", use_container_width=True):
            summary_df, qf_list = simulate_playoffs_exact(edited_qf, model, scaler, event_tier="Major")

            st.markdown("#### Quarter-Finals (BO3)")
            q_cols = st.columns(4)
            for idx, m in enumerate(qf_list):
                with q_cols[idx]:
                    st.markdown(f"""
                    <div class="prob-box" style="padding:12px;">
                        <div style="color:#8b949e; font-size:12px;">{m['match']} (BO3)</div>
                        <div style="margin-top:6px;"><strong>{m['team_a']}</strong>: {m['prob_a']*100:.1f}%</div>
                        <div><strong>{m['team_b']}</strong>: {m['prob_b']*100:.1f}%</div>
                        <div style="color:#de9b35; font-size:12px; margin-top:6px;">Pick: {m['pick']}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("#### 🏆 Probabilități Avansare & Câștigare Major (Grand Final BO5)")
            st.dataframe(
                summary_df.style.format({
                    'Semifinals (%)': '{:.1f}%',
                    'Grand Final (%)': '{:.1f}%',
                    'Champion (%)': '{:.1f}%'
                }),
                use_container_width=True
            )