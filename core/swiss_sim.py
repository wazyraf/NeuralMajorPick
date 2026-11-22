import os
import numpy as np
import pandas as pd
from core.model_engine import compute_monotone_logit_from_inputs
from core.team_stats_engine import load_team_stats, compute_map_veto_edge, compute_h2h_history

REGION_SOS_DEFAULT = {
    "Europe (Tier 1/1.5)": 0.00,
    "Americas": -0.03,
    "Asia / Oceania": -0.06,
}


def _build_probability_matrices(
    teams_df: pd.DataFrame,
    model,
    stage_mode: str = "Stage 1 / Stage 2 (BO1 + Decisive BO3)",
    is_major: bool = True,
    as_of_date: str | None = None,
    team_stats_csv: str = "data/team_stats.csv",
    matches_csv: str = "data/matches.csv",
    use_team_stats: bool = True,
):
    n_teams = len(teams_df)
    teams = teams_df["team"].tolist()
    ranks = teams_df["rank"].tolist()
    raw_wrs = teams_df["wr_3m"].tolist()

    stats_db = load_team_stats(csv_path=team_stats_csv, as_of_date=as_of_date) if use_team_stats else {}

    effective_wrs = []
    form_scores = []
    for idx, w in enumerate(raw_wrs):
        sa = stats_db.get(str(teams[idx]).strip().lower()) if use_team_stats else None
        reg_adj = REGION_SOS_DEFAULT.get(teams_df["region"].iloc[idx], 0.0) if "region" in teams_df.columns else 0.0
        if sa:
            base_w = 0.55 * float(sa.get("combined_2026_wr", w)) + 0.45 * float(w)
            f_score = (
                0.50 * float(sa.get("recent_wr", base_w))
                + 0.30 * float(np.clip(sa.get("recent_avg_round_diff", 0.0) / 8.0, -0.5, 0.5))
                + 0.20 * float(sa.get("top20_wr", base_w))
            )
        else:
            base_w = float(w)
            f_score = base_w
        effective_wrs.append(float(np.clip(base_w + reg_adj, 0.05, 0.95)))
        form_scores.append(f_score)

    all_bo3 = "Stage 3" in stage_mode or "All BO3" in stage_mode
    event_tier = "Major" if is_major else "S-Tier LAN"
    temp = float(getattr(model, "learned_temp", 0.85)) if model is not None else 0.85

    prob_bo1 = np.zeros((n_teams, n_teams))
    prob_bo3 = np.zeros((n_teams, n_teams))
    vol_bo1 = np.full((n_teams, n_teams), 0.5)
    vol_bo3 = np.full((n_teams, n_teams), 0.4)

    for i in range(n_teams):
        for j in range(n_teams):
            if i == j:
                continue
            sa = stats_db.get(str(teams[i]).strip().lower()) if use_team_stats else None
            sb = stats_db.get(str(teams[j]).strip().lower()) if use_team_stats else None
            f_edge = float(np.clip(form_scores[i] - form_scores[j], -0.35, 0.35)) if (sa and sb) else 0.0
            m_edge_1 = compute_map_veto_edge(sa, sb, match_format="BO1") if (sa and sb) else 0.0
            m_edge_3 = compute_map_veto_edge(sa, sb, match_format="BO3") if (sa and sb) else 0.0
            h2h_edge = (
                compute_h2h_history(teams[i], teams[j], matches_csv=matches_csv, as_of_date=as_of_date)["h2h_edge"]
                if use_team_stats
                else 0.0
            )

            l1 = compute_monotone_logit_from_inputs(
                rank_a=ranks[i], rank_b=ranks[j],
                wr_a=effective_wrs[i], wr_b=effective_wrs[j],
                match_format="BO1", event_tier=event_tier,
                map_edge_a=m_edge_1, h2h_edge_a=h2h_edge, form_edge_a=f_edge, temp=temp,
            )
            l3 = compute_monotone_logit_from_inputs(
                rank_a=ranks[i], rank_b=ranks[j],
                wr_a=effective_wrs[i], wr_b=effective_wrs[j],
                match_format="BO3", event_tier=event_tier,
                map_edge_a=m_edge_3, h2h_edge_a=h2h_edge, form_edge_a=f_edge, temp=temp,
            )

            p1 = float(np.clip(1.0 / (1.0 + np.exp(-l1)), 0.08, 0.92))
            p3 = float(np.clip(1.0 / (1.0 + np.exp(-l3)), 0.08, 0.92))

            prob_bo1[i, j] = p1
            prob_bo3[i, j] = p3
            vol_bo1[i, j] = float(np.clip(0.55 - 0.35 * abs(p1 - 0.5) * 2.0, 0.15, 0.85))
            vol_bo3[i, j] = float(np.clip(0.42 - 0.30 * abs(p3 - 0.5) * 2.0, 0.12, 0.80))

    return {
        "n_teams": n_teams,
        "teams": teams,
        "ranks": ranks,
        "raw_wrs": raw_wrs,
        "all_bo3": all_bo3,
        "prob_bo1": prob_bo1,
        "prob_bo3": prob_bo3,
        "vol_bo1": vol_bo1,
        "vol_bo3": vol_bo3,
    }


def simulate_swiss_stage(
    teams_df: pd.DataFrame,
    model,
    scaler,
    n_sims: int = 10000,
    stage_mode: str = "Stage 1 / Stage 2 (BO1 + Decisive BO3)",
    is_major: bool = True,
    as_of_date: str | None = None,
    team_stats_csv: str = "data/team_stats.csv",
    matches_csv: str = "data/matches.csv",
    use_team_stats: bool = True,
    return_diagnostics: bool = False,
):
    n_teams = len(teams_df)
    if n_teams < 2:
        raise ValueError(f"Swiss stage necesită cel puțin 2 echipe, primite: {n_teams}")

    mats = _build_probability_matrices(
        teams_df=teams_df,
        model=model,
        stage_mode=stage_mode,
        is_major=is_major,
        as_of_date=as_of_date,
        team_stats_csv=team_stats_csv,
        matches_csv=matches_csv,
        use_team_stats=use_team_stats,
    )
    teams = mats["teams"]
    ranks = mats["ranks"]
    raw_wrs = mats["raw_wrs"]
    all_bo3 = mats["all_bo3"]
    prob_bo1 = mats["prob_bo1"]
    prob_bo3 = mats["prob_bo3"]
    vol_bo1 = mats["vol_bo1"]
    vol_bo3 = mats["vol_bo3"]

    counts_3_0 = np.zeros(n_teams)
    counts_qual = np.zeros(n_teams)
    counts_0_3 = np.zeros(n_teams)
    vol_accum = np.zeros(n_teams)
    matches_played_accum = np.zeros(n_teams)
    pair_appearances = {}

    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        wins = np.zeros(n_teams, dtype=int)
        losses = np.zeros(n_teams, dtype=int)
        opponents = [[] for _ in range(n_teams)]
        played = set()

        half = n_teams // 2
        p_mat_r1 = prob_bo3 if all_bo3 else prob_bo1
        v_mat_r1 = vol_bo3 if all_bo3 else vol_bo1
        for i in range(half):
            t1, t2 = i, i + half
            played.add((t1, t2))
            played.add((t2, t1))
            opponents[t1].append(t2)
            opponents[t2].append(t1)

            key_p = (min(t1, t2), max(t1, t2), "R1")
            pair_appearances[key_p] = pair_appearances.get(key_p, 0) + 1

            vol_accum[t1] += v_mat_r1[t1, t2]
            vol_accum[t2] += v_mat_r1[t2, t1]
            matches_played_accum[t1] += 1
            matches_played_accum[t2] += 1

            if rng.random() < p_mat_r1[t1, t2]:
                wins[t1] += 1
                losses[t2] += 1
            else:
                wins[t2] += 1
                losses[t1] += 1

        for round_num in range(2, 6):
            active = [i for i in range(n_teams) if wins[i] < 3 and losses[i] < 3]
            if not active:
                break

            buchholz = {
                i: sum(wins[opp] - losses[opp] for opp in opponents[i])
                for i in active
            }

            pools = {}
            for idx in active:
                pools.setdefault((wins[idx], losses[idx]), []).append(idx)

            for record, pool in pools.items():
                pool.sort(key=lambda idx: (-buchholz[idx], idx))
                w_rec, l_rec = record
                is_decisive = all_bo3 or (w_rec == 2 or l_rec == 2)
                p_mat = prob_bo3 if is_decisive else prob_bo1
                v_mat = vol_bo3 if is_decisive else vol_bo1

                unpaired = pool.copy()
                while len(unpaired) >= 2:
                    t1 = unpaired.pop(0)
                    opp_idx = len(unpaired) - 1
                    while opp_idx > 0 and (t1, unpaired[opp_idx]) in played:
                        opp_idx -= 1
                    t2 = unpaired.pop(opp_idx)

                    played.add((t1, t2))
                    played.add((t2, t1))
                    opponents[t1].append(t2)
                    opponents[t2].append(t1)

                    if is_decisive or round_num >= 4:
                        key_p = (min(t1, t2), max(t1, t2), f"{w_rec}-{l_rec}")
                        pair_appearances[key_p] = pair_appearances.get(key_p, 0) + 1

                    vol_accum[t1] += v_mat[t1, t2]
                    vol_accum[t2] += v_mat[t2, t1]
                    matches_played_accum[t1] += 1
                    matches_played_accum[t2] += 1

                    if rng.random() < p_mat[t1, t2]:
                        wins[t1] += 1
                        losses[t2] += 1
                    else:
                        wins[t2] += 1
                        losses[t1] += 1

        for i in range(n_teams):
            if wins[i] == 3 and losses[i] == 0:
                counts_3_0[i] += 1
                counts_qual[i] += 1
            elif wins[i] == 3:
                counts_qual[i] += 1
            elif losses[i] == 3 and wins[i] == 0:
                counts_0_3[i] += 1

    avg_vol = np.where(matches_played_accum > 0, vol_accum / np.maximum(matches_played_accum, 1), 0.5)
    p_3_0_pct = (counts_3_0 / n_sims) * 100
    p_qual_pct = (counts_qual / n_sims) * 100
    clean_3_0_score = p_3_0_pct * (0.72 + 0.28 * np.clip(p_3_0_pct / np.maximum(p_qual_pct, 1.0), 0.0, 1.0))

    res_df = pd.DataFrame({
        "VRS Seed": range(1, n_teams + 1),
        "Echipa": teams,
        "Rank": ranks,
        "WR 3M": [f"{w*100:.1f}%" for w in raw_wrs],
        "3-0 (%)": p_3_0_pct,
        "3:0 Score": clean_3_0_score,
        "Calificare Total (%)": p_qual_pct,
        "3-1 / 3-2 (%)": ((counts_qual - counts_3_0) / n_sims) * 100,
        "0-3 (%)": (counts_0_3 / n_sims) * 100,
        "Volatilitate (%)": avg_vol * 100,
    })

    if not return_diagnostics:
        return res_df

    diag_rows = []
    for (a_idx, b_idx, rec_lbl), cnt in sorted(pair_appearances.items(), key=lambda x: -x[1])[:10]:
        p_ab = prob_bo3[a_idx, b_idx] if (all_bo3 or ("-" in rec_lbl and ("2-" in rec_lbl or "-2" in rec_lbl))) else prob_bo1[a_idx, b_idx]
        closeness_pct = (1.0 - min(1.0, abs(p_ab - 0.5) * 2.0)) * 100.0
        diag_rows.append({
            "Duel": f"{teams[a_idx]} vs {teams[b_idx]}",
            "Scor / Rundă": rec_lbl,
            "Frecvență (%)": (cnt / n_sims) * 100.0,
            "Prob. A (%)": p_ab * 100.0,
            "Tensiune / Incertitudine (%)": closeness_pct,
        })

    pivot_df = res_df.copy()
    pivot_df["Dispersie Bracket"] = (
        np.minimum(pivot_df["Calificare Total (%)"], 100.0 - pivot_df["Calificare Total (%)"]) * 0.65
        + pivot_df["Volatilitate (%)"] * 0.35
    )
    pivot_df = pivot_df.sort_values("Dispersie Bracket", ascending=False)[
        ["Echipa", "VRS Seed", "Calificare Total (%)", "3-1 / 3-2 (%)", "Volatilitate (%)", "Dispersie Bracket"]
    ].head(6)

    return res_df, {
        "key_matchups": pd.DataFrame(diag_rows),
        "pivot_teams": pivot_df,
    }


def simulate_swiss_match_by_match(
    teams_df: pd.DataFrame,
    model,
    scaler,
    stage_mode: str = "Stage 1 / Stage 2 (BO1 + Decisive BO3)",
    is_major: bool = True,
    as_of_date: str | None = None,
    team_stats_csv: str = "data/team_stats.csv",
    matches_csv: str = "data/matches.csv",
    use_team_stats: bool = True,
    seed: int = 123,
    deterministic_picks: bool = False,
):
    """
    Simulează UN bracket complet rundă cu rundă (R1..R5) cu Buchholz + VRS Seed,
    returnând:
      - matches_df: fiecare meci simulat cu rundă, scor W-L înainte de meci, probabilitate, câștigător și marcaj de upset
      - standings_df: clasarea finală pe bilanț (3-0, 3-1, 3-2, 2-3, 1-3, 0-3) și Buchholz final
    """
    n_teams = len(teams_df)
    if n_teams < 2:
        raise ValueError(f"Swiss stage necesită cel puțin 2 echipe, primite: {n_teams}")

    mats = _build_probability_matrices(
        teams_df=teams_df,
        model=model,
        stage_mode=stage_mode,
        is_major=is_major,
        as_of_date=as_of_date,
        team_stats_csv=team_stats_csv,
        matches_csv=matches_csv,
        use_team_stats=use_team_stats,
    )
    teams = mats["teams"]
    ranks = mats["ranks"]
    raw_wrs = mats["raw_wrs"]
    all_bo3 = mats["all_bo3"]
    prob_bo1 = mats["prob_bo1"]
    prob_bo3 = mats["prob_bo3"]
    vol_bo1 = mats["vol_bo1"]
    vol_bo3 = mats["vol_bo3"]

    wins = np.zeros(n_teams, dtype=int)
    losses = np.zeros(n_teams, dtype=int)
    opponents = [[] for _ in range(n_teams)]
    played = set()
    match_records = []
    rng = np.random.default_rng(seed)

    def _play_pair(round_num: int, w_rec: int, l_rec: int, t1: int, t2: int, is_decisive: bool):
        fmt = "BO3" if is_decisive else "BO1"
        p12 = float(prob_bo3[t1, t2] if is_decisive else prob_bo1[t1, t2])
        v12 = float(vol_bo3[t1, t2] if is_decisive else vol_bo1[t1, t2])

        if deterministic_picks:
            t1_wins = p12 >= 0.5
        else:
            t1_wins = bool(rng.random() < p12)

        winner_idx = t1 if t1_wins else t2
        loser_idx = t2 if t1_wins else t1
        fav_idx = t1 if p12 >= 0.5 else t2
        is_upset = winner_idx != fav_idx
        win_prob = p12 if t1_wins else (1.0 - p12)

        played.add((t1, t2))
        played.add((t2, t1))
        opponents[t1].append(t2)
        opponents[t2].append(t1)

        wins[winner_idx] += 1
        losses[loser_idx] += 1

        match_records.append({
            "Rundă": f"R{round_num}",
            "Pool": f"{w_rec}-{l_rec}",
            "Format": fmt,
            "Echipa A": f"#{idx_seed(t1)} {teams[t1]}",
            "Echipa B": f"#{idx_seed(t2)} {teams[t2]}",
            "Prob. A (%)": round(p12 * 100.0, 1),
            "Câștigător": f"#{idx_seed(winner_idx)} {teams[winner_idx]} ({win_prob*100:.1f}%)",
            "Volatilitate (%)": round(v12 * 100.0, 1),
            "Upset": "⚡ DA" if is_upset else "-",
        })

    def idx_seed(k: int) -> int:
        return k + 1

    half = n_teams // 2
    for i in range(half):
        _play_pair(1, 0, 0, i, i + half, is_decisive=all_bo3)

    for round_num in range(2, 6):
        active = [i for i in range(n_teams) if wins[i] < 3 and losses[i] < 3]
        if not active:
            break

        buchholz = {i: sum(wins[opp] - losses[opp] for opp in opponents[i]) for i in active}
        pools = {}
        for idx in active:
            pools.setdefault((wins[idx], losses[idx]), []).append(idx)

        for record, pool in sorted(pools.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
            pool.sort(key=lambda idx: (-buchholz[idx], idx))
            w_rec, l_rec = record
            is_decisive = all_bo3 or (w_rec == 2 or l_rec == 2)

            unpaired = pool.copy()
            while len(unpaired) >= 2:
                t1 = unpaired.pop(0)
                opp_idx = len(unpaired) - 1
                while opp_idx > 0 and (t1, unpaired[opp_idx]) in played:
                    opp_idx -= 1
                t2 = unpaired.pop(opp_idx)
                _play_pair(round_num, w_rec, l_rec, t1, t2, is_decisive=is_decisive)

    final_buchholz = {i: sum(wins[opp] - losses[opp] for opp in opponents[i]) for i in range(n_teams)}
    standings_rows = []
    for i in range(n_teams):
        rec_str = f"{wins[i]}-{losses[i]}"
        if rec_str == "3-0":
            status_lbl = "🔥 3:0 Advance"
        elif rec_str in ("3-1", "3-2"):
            status_lbl = f"✅ {rec_str} Advance"
        elif rec_str == "0-3":
            status_lbl = "💀 0:3 Eliminated"
        else:
            status_lbl = f"❌ {rec_str} Eliminated"

        standings_rows.append({
            "VRS Seed": i + 1,
            "Echipa": teams[i],
            "Rank": ranks[i],
            "WR 3M": f"{raw_wrs[i]*100:.1f}%",
            "Bilanț": rec_str,
            "Buchholz": int(final_buchholz[i]),
            "Status": status_lbl,
        })

    order_map = {"3-0": 0, "3-1": 1, "3-2": 2, "2-3": 3, "1-3": 4, "0-3": 5}
    standings_df = (
        pd.DataFrame(standings_rows)
        .sort_values(by=["Bilanț", "Buchholz", "VRS Seed"], key=lambda col: col.map(order_map) if col.name == "Bilanț" else col, ascending=[True, False, True])
        .reset_index(drop=True)
    )

    return pd.DataFrame(match_records), standings_df