# Hub-ul principal Major: Pick'Em Bracket + Match-by-Match (Map Veto/Score/Rounds) + Playoffs
import streamlit as st
import numpy as np
import pandas as pd
from core.model_engine import load_predictor, predict_match_context, predict_match_with_team_stats
from core.swiss_sim import simulate_swiss_stage, simulate_swiss_match_by_match, REGION_SOS_DEFAULT
from core.playoff_sim import simulate_playoffs_exact
from core.team_stats_engine import (
    load_team_stats,
    get_latest_vrs_ranks,
    compute_map_veto_edge,
    assess_match_risk,
    compute_h2h_history,
)
from ui.styles import render_hltv_pickem_bracket


def _default_as_of_date() -> str:
    if st.session_state.get("blind_test_enabled", True):
        return str(st.session_state.get("team_stats_as_of") or st.session_state.get("cutoff_date") or "2026-06-02")
    return ""


def render_major_hub():
    model, scaler = load_predictor()
    if model is None:
        st.warning("⚠️ Modelul nu este încă antrenat! Mergi în tab-ul **🧪 DATA & MODEL LAB** și apasă **ANTRENEAZĂ REȚEAUA**.")
        return

    default_as_of = _default_as_of_date()

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

        c_cfg1, c_cfg2, c_cfg3, c_cfg4, c_cfg5 = st.columns([1.8, 1.3, 1.0, 1.0, 1.2])
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
            sim_engine_mode = st.selectbox(
                "Mod Simulare Pick'Em",
                [
                    "Monte Carlo Probabilist (V1)",
                    "Meci-cu-Meci & Buchholz (majors.im style)"
                ],
                index=0
            )
        with c_cfg3:
            n_sims = st.selectbox("Simulări MC", [2000, 5000, 10000], index=2)
        with c_cfg4:
            use_sos = st.checkbox("Regiune / SoS", value=True, help="Penalizează ușor winrate-ul umflat din regiuni/turnee mai accesibile.")
        with c_cfg5:
            use_tstats_swiss = st.checkbox("Team Stats", value=True, help="Integrează forma recentă, round diff, veto, H2H și volatilitatea.")

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

        m_opt1, m_opt2 = st.columns([1, 2])
        with m_opt1:
            mbm_seed = st.number_input("Seed Traseu Meci-cu-Meci", 1, 99999, 123, step=1)
        with m_opt2:
            mbm_det = st.checkbox("Alegere deterministă (mereu favoritul probabilității pe fiecare meci)", value=False)

        if st.button("🚀 GENEREAZĂ BRACKET PICK'EM", type="primary", use_container_width=True):
            sim_input = edited_df.copy()
            if not use_sos:
                sim_input['region'] = "Europe (Tier 1/1.5)"

            as_of_val = default_as_of or None

            if sim_engine_mode.startswith("Monte Carlo"):
                with st.spinner(f"Rulăm {n_sims} simulări oficiale Valve Major (VRS + Buchholz)..."):
                    sim_res, diag = simulate_swiss_stage(
                        sim_input,
                        model,
                        scaler,
                        n_sims=n_sims,
                        stage_mode=stage_mode,
                        is_major=True,
                        as_of_date=as_of_val,
                        team_stats_csv="data/team_stats.csv",
                        use_team_stats=use_tstats_swiss,
                        return_diagnostics=True,
                    )

                sim_view = sim_res.copy()
                base_3_0 = sim_view['3:0 Score'] if '3:0 Score' in sim_view.columns else sim_view['3-0 (%)']
                high_qual_penalty = np.where(
                    sim_view['Calificare Total (%)'] > 80.0,
                    (sim_view['Calificare Total (%)'] - 80.0) * 0.45,
                    0.0,
                )
                sim_view['Pick_3_0_Rank_Score'] = base_3_0 - high_qual_penalty

                df_3_0 = sim_view.sort_values('Pick_3_0_Rank_Score', ascending=False).head(2)
                df_0_3 = sim_view.sort_values('0-3 (%)', ascending=False).head(2)
                excluded = set(df_3_0['Echipa']).union(set(df_0_3['Echipa']))
                df_adv = (
                    sim_view[~sim_view['Echipa'].isin(excluded)]
                    .sort_values('Calificare Total (%)', ascending=False)
                    .head(6)
                )

                st.markdown("#### 🎯 HLTV Visual Pick'Em Bracket (Monte Carlo)")
                render_hltv_pickem_bracket(df_3_0, df_adv, df_0_3)

                export_txt = (
                    f"=== NEURAL MAJOR PICK MC ({stage_mode}) ===\n"
                    f"3:0 -> {', '.join(df_3_0['Echipa'].tolist())}\n"
                    f"3:1 / 3:2 -> {', '.join(df_adv['Echipa'].tolist())}\n"
                    f"0:3 -> {', '.join(df_0_3['Echipa'].tolist())}"
                )
                st.download_button("📥 Export Pick'Em (.txt)", export_txt, file_name="major_pickem_mc.txt")

                fmt_dict = {
                    '3-0 (%)': '{:.1f}%',
                    'Calificare Total (%)': '{:.1f}%',
                    '3-1 / 3-2 (%)': '{:.1f}%',
                    '0-3 (%)': '{:.1f}%',
                }
                if '3:0 Score' in sim_res.columns:
                    fmt_dict['3:0 Score'] = '{:.1f}'
                if 'Volatilitate (%)' in sim_res.columns:
                    fmt_dict['Volatilitate (%)'] = '{:.1f}%'

                with st.expander("📊 Tabel Complet Probabilități (Toate cele 16 echipe)", expanded=True):
                    st.dataframe(sim_res.style.format(fmt_dict), use_container_width=True)

                with st.expander("🔍 Diagnostic Monte Carlo (Dueluri Cheie & Echipe Pivot)", expanded=True):
                    st.caption("Aici vezi duelurile frecvente din simulări cu tensiune ridicată și echipele cu cea mai mare dispersie de rezultat.")
                    d1, d2 = st.columns([1.3, 1])
                    with d1:
                        st.markdown("**Dueluri Cheie / Frecvente în Simulări**")
                        st.dataframe(
                            diag["key_matchups"].style.format({
                                "Frecvență (%)": "{:.1f}%",
                                "Prob. A (%)": "{:.1f}%",
                                "Tensiune / Incertitudine (%)": "{:.1f}%",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
                    with d2:
                        st.markdown("**Echipe Pivot (Dispersie Mare)**")
                        st.dataframe(
                            diag["pivot_teams"].style.format({
                                "Calificare Total (%)": "{:.1f}%",
                                "3-1 / 3-2 (%)": "{:.1f}%",
                                "Volatilitate (%)": "{:.1f}%",
                                "Dispersie Bracket": "{:.1f}",
                            }),
                            use_container_width=True,
                            hide_index=True,
                        )
            else:
                with st.spinner("Simulăm bracket-ul complet meci cu meci și Buchholz (R1..R5)..."):
                    matches_df, standings_df = simulate_swiss_match_by_match(
                        sim_input,
                        model,
                        scaler,
                        stage_mode=stage_mode,
                        is_major=True,
                        as_of_date=as_of_val,
                        team_stats_csv="data/team_stats.csv",
                        use_team_stats=use_tstats_swiss,
                        seed=int(mbm_seed),
                        deterministic_picks=bool(mbm_det),
                    )

                adv_3_0 = standings_df[standings_df["Bilanț"] == "3-0"]["Echipa"].tolist()
                adv_3_1_2 = standings_df[standings_df["Bilanț"].isin(["3-1", "3-2"])]["Echipa"].tolist()
                elim_0_3 = standings_df[standings_df["Bilanț"] == "0-3"]["Echipa"].tolist()

                st.markdown("#### 🧭 Rezultat Clasare Finală din Traseul Meci-cu-Meci")
                k1, k2, k3 = st.columns(3)
                k1.success(f"**3:0 Advance ({len(adv_3_0)}):** " + (", ".join(adv_3_0) if adv_3_0 else "-"))
                k2.info(f"**3:1 / 3:2 Advance ({len(adv_3_1_2)}):** " + (", ".join(adv_3_1_2) if adv_3_1_2 else "-"))
                k3.error(f"**0:3 Eliminated ({len(elim_0_3)}):** " + (", ".join(elim_0_3) if elim_0_3 else "-"))

                export_mbm = (
                    f"=== NEURAL MAJOR PICK MATCH-BY-MATCH ({stage_mode}) ===\n"
                    f"3:0 -> {', '.join(adv_3_0)}\n"
                    f"3:1 / 3:2 -> {', '.join(adv_3_1_2)}\n"
                    f"0:3 -> {', '.join(elim_0_3)}"
                )
                st.download_button("📥 Export Clasare Meci-cu-Meci (.txt)", export_mbm, file_name="major_pickem_mbm.txt")

                with st.expander("🏁 Clasament Final pe Bilanț & Buchholz", expanded=True):
                    st.dataframe(standings_df, use_container_width=True, hide_index=True)

                with st.expander("⚔️ Traseul Complet al Meciurilor (Rundă cu Rundă)", expanded=True):
                    st.dataframe(matches_df, use_container_width=True, hide_index=True)

    # =========================================================
    # SUB-TAB 2: MATCH-BY-MATCH & MAP ANALYZER
    # =========================================================
    with sub_match:
        st.markdown("### ⚔️ Analizor Meci Individual (VRS Rank / Veto Simulat / H2H / Scor)")

        as_of_cutoff = st.text_input(
            "Dată limită snapshot-uri `team_stats.csv` & `matches.csv` (ex. `2026-06-02` pentru Blind Backtest)",
            value=default_as_of,
            key="h2h_as_of_date"
        ).strip() or None

        team_stats_map = load_team_stats("data/team_stats.csv", as_of_date=as_of_cutoff)
        vrs_rank_map = get_latest_vrs_ranks("data/matches.csv", as_of_date=as_of_cutoff)

        c1, c_mid, c2 = st.columns([2, 1.3, 2])

        with c1:
            st.subheader("🔵 TEAM A")
            team_a = st.text_input("Echipă A", value="Vitality", key="inp_team_a")
            key_a = team_a.strip().lower()
            sa = team_stats_map.get(key_a)
            auto_rank_a = int(vrs_rank_map.get(key_a, 1))
            rank_a = st.number_input("VRS Rank A", 1, 200, auto_rank_a, key=f"vrs_rank_a_{key_a}")
            auto_wr_a = float(sa["combined_2026_wr"]) if sa else 0.75
            wr_a = st.slider("Winrate (2026 / Recent 20m) A", 0.0, 1.0, round(auto_wr_a, 2), 0.01, key=f"wr_a_{key_a}")
            reg_a = st.selectbox("Context Turnee A", list(REGION_SOS_DEFAULT.keys()), index=0, key="reg_a")
            if sa:
                st.caption(
                    f"✅ Stats A: 2026/Rec WR `{sa['combined_2026_wr']*100:.1f}%` (Rec20 `{sa['recent_wr']*100:.1f}%`, "
                    f"`{sa['recent_avg_round_diff']:+.2f}` RD) | Top20 WR `{sa['top20_wr']*100:.1f}%` | Eff. R2 Break `{sa['effective_r2_break']*100:.1f}%`"
                )
            else:
                st.caption("⚠️ Team A nu e în `team_stats.csv` — folosește valorile manuale.")

        with c2:
            st.subheader("🟠 TEAM B")
            team_b = st.text_input("Echipă B", value="FUT", key="inp_team_b")
            key_b = team_b.strip().lower()
            sb = team_stats_map.get(key_b)
            auto_rank_b = int(vrs_rank_map.get(key_b, 8))
            rank_b = st.number_input("VRS Rank B", 1, 200, auto_rank_b, key=f"vrs_rank_b_{key_b}")
            auto_wr_b = float(sb["combined_2026_wr"]) if sb else 0.55
            wr_b = st.slider("Winrate (2026 / Recent 20m) B", 0.0, 1.0, round(auto_wr_b, 2), 0.01, key=f"wr_b_{key_b}")
            reg_b = st.selectbox("Context Turnee B", list(REGION_SOS_DEFAULT.keys()), index=0, key="reg_b")
            if sb:
                st.caption(
                    f"✅ Stats B: 2026/Rec WR `{sb['combined_2026_wr']*100:.1f}%` (Rec20 `{sb['recent_wr']*100:.1f}%`, "
                    f"`{sb['recent_avg_round_diff']:+.2f}` RD) | Top20 WR `{sb['top20_wr']*100:.1f}%` | Eff. R2 Break `{sb['effective_r2_break']*100:.1f}%`"
                )
            else:
                st.caption("⚠️ Team B nu e în `team_stats.csv` — folosește valorile manuale.")

        with c_mid:
            st.subheader("⚙️ CONTEXT")
            stage_ctx = st.selectbox("Fază Turneu", ["Stage 1", "Stage 2", "Stage 3", "Playoffs (QF/SF)", "Grand Final"])
            default_fmt_idx = 2 if stage_ctx == "Grand Final" else (1 if stage_ctx in ["Stage 3", "Playoffs (QF/SF)"] else 0)
            match_fmt = st.selectbox("Format", ["BO1", "BO3", "BO5"], index=default_fmt_idx)
            auto_veto = round(compute_map_veto_edge(sa, sb, match_format=match_fmt), 2)
            map_edge = st.slider(
                "Avantaj Hartă / Veto Simulat (-0.25 B .. +0.25 A)",
                -0.25,
                0.25,
                auto_veto,
                0.01,
                key=f"veto_{key_a}_{key_b}_{match_fmt}"
            )
            h2h_preview = compute_h2h_history(team_a, team_b, as_of_date=as_of_cutoff)
            auto_h2h = round(float(h2h_preview.get("h2h_edge", 0.0)), 2)
            h2h_edge = st.slider(
                "Avantaj H2H Recent (-0.20 B .. +0.20 A)",
                -0.20,
                0.20,
                auto_h2h,
                0.01,
                key=f"h2h_{key_a}_{key_b}_{as_of_cutoff}"
            )
            if h2h_preview.get("h2h_games", 0) > 0:
                st.caption(
                    f"H2H istoric (< {as_of_cutoff or 'all'}): {h2h_preview['h2h_wins_a']}/{h2h_preview['h2h_games']} "
                    f"pentru {team_a} (edge `{auto_h2h:+.2f}`)"
                )
            calc_btn = st.button("🔥 CALCULEAZĂ MECIUL", type="primary", use_container_width=True)

        if calc_btn:
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
                    r2_break_a=sa.get("effective_r2_break") if sa else None,
                    r2_break_b=sb.get("effective_r2_break") if sb else None,
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

            if (
                res.get("volatility_level") == "High"
                or float(res.get("volatility_score", 0.0)) >= 55.0
                or float(res.get("upset_risk_pct", 0.0)) >= 30.0
            ):
                st.warning(
                    f"⚠️ **ANUNȚ VOLATILITATE / RISC SURPRIZĂ:** Meci cu variație notabilă "
                    f"(Volatility Score: **{res['volatility_score']:.1f}/100**, Upset Risk: **{res['upset_risk_pct']:.1f}%**)."
                )

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("⚡ Upset Risk", f"{res['upset_risk_pct']:.1f}%")
            m2.metric("🌊 Volatility Score", f"{res['volatility_score']:.1f} / 100")
            m3.metric("📈 Calibrated Vol.", f"{res.get('volatility', 0.5)*100:.1f}% ({res.get('volatility_level', 'N/A')})")
            m4.metric("🛡️ Trust Level", f"{res['badge']} — {res['trust_label']}")

            if res.get("volatility_reasons"):
                st.caption("Factori volatilitate: " + " • ".join(res["volatility_reasons"]))

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
            as_of_val = default_as_of or None
            summary_df, qf_list = simulate_playoffs_exact(
                edited_qf,
                model,
                scaler,
                event_tier="Major",
                as_of_date=as_of_val,
                team_stats_csv="data/team_stats.csv",
                use_team_stats=True,
            )

            st.markdown("#### Quarter-Finals (BO3)")
            q_cols = st.columns(4)
            for idx, m in enumerate(qf_list):
                vol_lbl = m.get("volatility_level", "Medium")
                with q_cols[idx]:
                    st.markdown(f"""
                    <div class="prob-box" style="padding:12px;">
                        <div style="color:#8b949e; font-size:12px;">{m['match']} (BO3) • Vol: {vol_lbl}</div>
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