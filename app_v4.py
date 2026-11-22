import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st

from src.fantasy_optimizer import (
    fit_fantasy_model_from_past,
    load_fantasy_players,
    optimize_fantasy_lineups,
    score_fantasy_players,
)
from src.v4_inference import (
    ACTIVE_MAPS,
    build_singapore_teams_df,
    fit_v4_model,
    load_map_pool,
    load_matches_clean,
    pick_10_slots,
    predict_match_with_veto,
    simulate_swiss_16,
)

st.set_page_config(page_title="CS2 Predictor V4 + Veto + Fantasy — Singapore 2026", layout="wide")
st.title("CS2 Match, Swiss Pick'Em & HLTV Fantasy — V4 (Singapore 2026)")

with st.sidebar:
    st.header("Configurare")
    matches_path = st.text_input("Fișier meciuri", "data/matches.csv")
    stats_path = st.text_input("Fișier stats Singapore", "data/vrs_team_stats_singapore.csv")
    map_pool_path = st.text_input("Fișier Map Pool", "data/map_pool_singapore.csv")
    fantasy_path = st.text_input("Fișier jucători Fantasy", "data/fantasy_players_singapore.csv")
    past_fantasy_path = st.text_input("Fișier istoric Fantasy", "data/fantasy_past_events.csv")
    as_of_date = st.text_input("Data cutoff (include meciuri < data)", "2026-11-23")

    st.divider()
    st.subheader("Simulare Veto & Monte Carlo")
    map_weight = st.slider("Pondere Map Pool / Veto (0 = pur V4)", 0.0, 0.80, 0.35, 0.05)
    n_sims = st.select_slider("Simulări Monte Carlo", options=[2000, 5000, 10000, 20000], value=10000)

    st.divider()
    st.subheader("Reglaj alocare Pick'Em")
    overqual_cap = st.slider("Prag p_qual pentru penalizare 3-0 (%)", 60.0, 95.0, 85.0, 1.0)
    overqual_mult = st.slider("Intensitate penalizare 3-0", 0.0, 3.0, 1.2, 0.1)


def update_map_pool_csv(
    csv_path: str,
    team_a: str,
    team_b: str,
    map_results: list[tuple[str, str]],
) -> None:
    """Actualizează maps_played și map_winrate în map_pool_singapore.csv după hărțile jucate."""
    if not map_results:
        return
    mdf = pd.read_csv(csv_path)
    for map_name, map_winner in map_results:
        for team, won in [(team_a, map_winner == team_a), (team_b, map_winner == team_b)]:
            mask = (
                (mdf["team"].astype(str).str.strip().str.lower() == team.strip().lower())
                & (mdf["map"].astype(str).str.strip() == map_name)
            )
            if mask.any():
                idx = mdf[mask].index[0]
                old_cnt = int(mdf.loc[idx, "maps_played"])
                old_wr = float(mdf.loc[idx, "map_winrate"])
                old_wins = round(old_cnt * old_wr)
                new_cnt = old_cnt + 1
                new_wins = old_wins + (1 if won else 0)
                mdf.loc[idx, "maps_played"] = new_cnt
                mdf.loc[idx, "map_winrate"] = round(new_wins / new_cnt, 4)
    mdf.to_csv(csv_path, index=False)


try:
    matches_df = load_matches_clean(matches_path)
    train_df = matches_df[matches_df["date_dt"] < pd.to_datetime(as_of_date)].copy().reset_index(drop=True)
    pipe, coefs = fit_v4_model(train_df, stats_csv=stats_path)
    teams_df = build_singapore_teams_df(stats_path, train_df)
    map_pool = load_map_pool(map_pool_path, teams_df)
except Exception as exc:
    st.error(f"Eroare la încărcare / antrenare: {exc}")
    st.stop()

st.caption(
    f"Antrenat pe **{len(train_df)}** meciuri strict înainte de `{as_of_date}` | "
    f"Pondere Map Veto: **{map_weight:.2f}**"
)

sim_df, r1_df = simulate_swiss_16(
    pipe=pipe,
    teams_df=teams_df,
    stats_csv=stats_path,
    as_of_date=as_of_date,
    map_pool=map_pool,
    map_weight=map_weight,
    n_sims=n_sims,
    overqual_cap=overqual_cap,
    overqual_mult=overqual_mult,
)
picks = pick_10_slots(sim_df)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Round 1 & Veto H2H",
    "Swiss Pick'Em (10 Slots)",
    "Map Pool & Coeficienți",
    "Live Match Input & Retrain",
    "HLTV Fantasy",
])

with tab1:
    st.subheader("Round 1 Opening Matchups (BO1 + Simulated Veto)")
    st.dataframe(r1_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Simulator H2H & Veto (pentru Round 2 / Round 3 / Stage 2)")
    team_names = teams_df["team"].tolist()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        ta = st.selectbox("Echipa A (Seed mai mare / Ban 1)", team_names, index=0, key="h2h_ta")
    with c2:
        tb = st.selectbox("Echipa B", team_names, index=8, key="h2h_tb")
    with c3:
        fmt = st.selectbox("Format", ["BO1", "BO3"], index=0, key="h2h_fmt")

    if ta == tb:
        st.warning("Alege două echipe diferite.")
    else:
        ra_row = teams_df[teams_df["team"] == ta].iloc[0]
        rb_row = teams_df[teams_df["team"] == tb].iloc[0]
        res = predict_match_with_veto(
            pipe=pipe,
            team_a=ta,
            team_b=tb,
            rank_a=int(ra_row["vrs_rank"]),
            rank_b=int(rb_row["vrs_rank"]),
            wr3_a=float(ra_row["map_winrate"]),
            wr3_b=float(rb_row["map_winrate"]),
            match_format=fmt,
            stats_csv=stats_path,
            as_of_date=as_of_date,
            map_pool=map_pool,
            map_weight=map_weight,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric(
            f"Șansă Finală {ta} ({fmt})",
            f"{res['p_series'] * 100:.1f}%",
            f"{(res['p_series'] - res['p_base']) * 100:+.1f}% din Veto",
        )
        m2.metric(f"Șansă Finală {tb} ({fmt})", f"{(1.0 - res['p_series']) * 100:.1f}%")
        m3.metric(
            "Probabilitate Bază V4 (fără hărți)",
            f"{res['p_base'] * 100:.1f}% vs {(1.0 - res['p_base']) * 100:.1f}%",
        )

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.markdown("**Ordinea simulată de Ban / Pick (Veto)**")
            st.dataframe(pd.DataFrame(res["veto_steps"]), use_container_width=True, hide_index=True)
        with v_col2:
            st.markdown("**Probabilități pe hărțile jucate**")
            st.dataframe(pd.DataFrame(res["map_details"]), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Recomandare Pick'Em (10 sloturi — cu Map Veto integrat)")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("**3-0 (2 echipe)**")
        for t in picks["3-0"]:
            st.success(t)
    with k2:
        st.markdown("**3-1 / 3-2 (6 echipe)**")
        for t in picks["3-1 / 3-2"]:
            st.info(t)
    with k3:
        st.markdown("**0-3 (2 echipe)**")
        for t in picks["0-3"]:
            st.error(t)

    st.divider()
    st.subheader("Probabilități Monte Carlo Swiss")
    st.dataframe(sim_df, use_container_width=True, hide_index=True)

    st.subheader("Șanse calificare (Qual %)")
    chart_df = sim_df.sort_values("Qual %", ascending=True).set_index("Team")[["Qual %"]]
    st.bar_chart(chart_df, height=420)

with tab3:
    st.subheader("Matrice Map Pool (Winrate efectiv % după shrinkage)")
    mp_rows = []
    for t in team_names:
        row_d = {"Team": t}
        for m in ACTIVE_MAPS:
            cnt = map_pool[t][m]["maps_played"]
            eff = round(map_pool[t][m]["eff_wr"] * 100.0, 1)
            row_d[m] = f"{eff}% ({cnt})" if cnt > 0 else f"{eff}% (0)"
        mp_rows.append(row_d)
    st.dataframe(pd.DataFrame(mp_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Ponderi standardizate învățate de modelul V4")
    coef_df = pd.DataFrame({
        "Feature": list(coefs.keys()),
        "Coef": np.round(list(coefs.values()), 4),
    })
    st.dataframe(coef_df, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Adaugă rezultat live (după fiecare meci) & actualizează modelul + Map Pool")
    st.info(
        "Tip: Trimite screenshot-ul meciului în chat ca să-l verificăm împreună, "
        "sau bifează direct aici scorul și hărțile jucate — aplicația actualizează automat "
        "`matches.csv` și `map_pool_singapore.csv`."
    )

    col_d, col_s, col_f = st.columns(3)
    with col_d:
        m_date = st.date_input("Data meciului", value=dt.date(2026, 11, 23))
    with col_s:
        m_stage = st.selectbox("Etapa (Stage)", ["Stage 1", "Stage 2", "Stage 3", "Playoffs"], index=0)
    with col_f:
        m_fmt = st.selectbox("Format jucat", ["BO1", "BO3", "BO5"], index=0, key="live_fmt")

    col_a, col_b, col_w = st.columns([2, 2, 2])
    with col_a:
        in_ta = st.selectbox("Team A", team_names, index=0, key="live_ta")
    with col_b:
        in_tb = st.selectbox("Team B", team_names, index=8, key="live_tb")
    with col_w:
        in_winner = st.radio("Câștigător serie", [in_ta, in_tb], horizontal=True)

    st.markdown("**Hărțile jucate în acest meci (actualizează automat `map_pool_singapore.csv`)**")
    n_maps_played = 1 if m_fmt == "BO1" else st.radio("Număr hărți jucate", [2, 3], horizontal=True)
    map_results: list[tuple[str, str]] = []
    for idx_m in range(n_maps_played):
        mc1, mc2 = st.columns([2, 3])
        with mc1:
            sel_map = st.selectbox(f"Harta {idx_m + 1}", ACTIVE_MAPS, index=idx_m % len(ACTIVE_MAPS), key=f"map_{idx_m}")
        with mc2:
            sel_mw = st.radio(f"Câștigător Harta {idx_m + 1}", [in_ta, in_tb], horizontal=True, key=f"mw_{idx_m}")
        map_results.append((sel_map, sel_mw))

    if st.button("Salvează meciul & Reantrenează acum", type="primary"):
        if in_ta == in_tb:
            st.error("Team A și Team B trebuie să fie diferite.")
        else:
            ra_r = teams_df[teams_df["team"] == in_ta].iloc[0]
            rb_r = teams_df[teams_df["team"] == in_tb].iloc[0]
            new_row = pd.DataFrame([{
                "date": str(m_date),
                "event_name": "PGL Major Singapore 2026",
                "event_tier": "Major",
                "stage": m_stage,
                "format": m_fmt,
                "team_a": in_ta,
                "team_b": in_tb,
                "rank_a": int(ra_r["vrs_rank"]),
                "rank_b": int(rb_r["vrs_rank"]),
                "winrate_3m_a": float(ra_r["map_winrate"]),
                "winrate_3m_b": float(rb_r["map_winrate"]),
                "winner": 1 if in_winner == in_ta else 0,
            }])
            raw_m = pd.read_csv(matches_path)
            raw_m = pd.concat([raw_m, new_row], ignore_index=True)
            raw_m.to_csv(matches_path, index=False)

            update_map_pool_csv(map_pool_path, in_ta, in_tb, map_results)
            st.success(
                f"Salvat: {in_ta} vs {in_tb} ({m_fmt}, câștigător: {in_winner})! "
                f"Notă: dacă data meciului este {m_date}, setează 'Data cutoff' din stânga la {m_date + dt.timedelta(days=1)} ca să fie inclus la antrenare."
            )
            st.rerun()

    st.divider()
    st.subheader("Meciuri înregistrate deja din PGL Major Singapore 2026")
    sg_logged = matches_df[matches_df["event_name"].astype(str).str.contains("Singapore", case=False, na=False)]
    if sg_logged.empty:
        st.caption("Încă nu există meciuri salvate din Singapore.")
    else:
        st.dataframe(sg_logged.drop(columns=["date_dt"], errors="ignore"), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("HLTV Fantasy Optimizer (Buget $1,000,000 | 5 Jucători | Max 2 / Echipă)")
    learned_f_weights, n_past_rows = fit_fantasy_model_from_past(past_fantasy_path)
    st.caption(f"Ponderi Fantasy calibrate și pe **{n_past_rows}** intrări istorice din `{past_fantasy_path}`.")

    try:
        raw_players = load_fantasy_players(fantasy_path)
    except Exception as exc:
        st.warning(f"Nu am putut încărca `{fantasy_path}`: {exc}")
        raw_players = pd.DataFrame()

    if not raw_players.empty:
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            budget_val = st.number_input("Buget maxim ($)", min_value=500000, max_value=1500000, value=1000000, step=5000)
        with fc2:
            qual_w = st.slider("Bonus Qual % / 3-1 & 3-2", 0.0, 1.50, 0.70, 0.05)
        with fc3:
            team_pull = st.slider("Temperare rating după Qual %", 0.0, 1.0, 0.55, 0.05)
        with fc4:
            min_qual_filter = st.slider("Filtru minim Team Qual %", 0.0, 75.0, 0.0, 5.0)

        scored_players = score_fantasy_players(
            raw_players,
            sim_df,
            learned_weights=learned_f_weights,
            qual_bonus_weight=qual_w,
            team_pull_strength=team_pull,
        )
        filtered_players = scored_players[scored_players["Team Qual %"] >= min_qual_filter].reset_index(drop=True)
        lineups = optimize_fantasy_lineups(filtered_players, budget=int(budget_val), team_size=5, max_per_team=2)

        st.markdown("### Lineup-uri recomandate (5 jucători, max 2 / echipă)")
        l_tabs = st.tabs(list(lineups.keys()))
        for l_tab, (l_name, l_df) in zip(l_tabs, lineups.items()):
            with l_tab:
                if l_df.empty:
                    st.info("Nicio combinație validă de 5 jucători pentru filtrul/bugetul ales. Scade 'Filtru minim Team Qual %'.")
                else:
                    tot_cost = int(l_df["price"].sum())
                    tot_pts = round(float(l_df["Exp Pts"].sum()), 2)
                    avg_qual = round(float(l_df["Team Qual %"].mean()), 1)
                    rem_b = int(budget_val) - tot_cost
                    m_a, m_b, m_c, m_d = st.columns(4)
                    m_a.metric("Total Exp Pts", f"{tot_pts:.2f}")
                    m_b.metric("Cost Total", f"${tot_cost:,}")
                    m_c.metric("Buget Rămas", f"${rem_b:,}")
                    m_d.metric("Medie Team Qual %", f"{avg_qual:.1f}%")
                    st.dataframe(l_df, use_container_width=True, hide_index=True)

        with st.expander("Vezi ponderile învățate din istoricul Fantasy"):
            st.dataframe(
                pd.DataFrame([{"Feature": k, "Weight": v} for k, v in learned_f_weights.items()]),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()
        st.markdown("### Clasament complet jucători (Value & Expected Points)")
        team_filter = st.multiselect("Filtrează după echipă", options=sorted(filtered_players["team"].unique().tolist()))
        disp_df = filtered_players if not team_filter else filtered_players[filtered_players["team"].isin(team_filter)]
        st.dataframe(
            disp_df[[
                "player", "team", "price", "rating", "Adj Rating", "Team Qual %", "Team 3-0 %",
                "Team 3-1/3-2 %", "Exp Maps", "Rating Pts", "Win Pts", "Qual Bonus", "Stat Edge",
                "Exp Pts", "Pts / $100k",
            ]],
            use_container_width=True,
            hide_index=True,
        )