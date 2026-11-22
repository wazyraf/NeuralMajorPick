import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features_v4 import FEATURE_COLS_V4, build_features_v4, swap_features_v4

SINGAPORE_S1_SEEDS = [
    "PARIVISION", "Liquid", "Ninjas in Pyjamas", "FaZe",
    "9z", "Eternal Fire", "HEROIC", "M80",
    "100 Thieves", "DENDELE", "Wildcard", "Lynn Vision",
    "BC.Game", "Imperial", "FlyQuest", "Chinggis Warriors",
]

ACTIVE_MAPS = ["Anubis", "Cache", "Nuke", "Dust2", "Ancient", "Inferno", "Mirage"]


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def logit(p: float) -> float:
    p_c = float(np.clip(p, 0.01, 0.99))
    return float(np.log(p_c / (1.0 - p_c)))


def load_matches_clean(matches_csv: str) -> pd.DataFrame:
    df = pd.read_csv(matches_csv)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt", "winner"]).sort_values("date_dt", ascending=True).reset_index(drop=True)
    return df


def load_map_pool(map_pool_csv: str, teams_df: pd.DataFrame) -> dict[str, dict[str, dict]]:
    overall_wr_map = {
        str(r["team"]).strip().lower(): float(r["map_winrate"])
        for _, r in teams_df.iterrows()
    }
    raw: dict[str, dict[str, tuple[int, float]]] = {}
    if map_pool_csv and os.path.exists(map_pool_csv):
        mdf = pd.read_csv(map_pool_csv)
        for _, r in mdf.iterrows():
            tk = str(r["team"]).strip().lower()
            mp = str(r["map"]).strip()
            cnt = int(pd.to_numeric(r["maps_played"], errors="coerce") or 0)
            wr = float(pd.to_numeric(r["map_winrate"], errors="coerce") or 0.0)
            raw.setdefault(tk, {})[mp] = (cnt, wr)

    pool: dict[str, dict[str, dict]] = {}
    for _, r in teams_df.iterrows():
        t_name = str(r["team"])
        tk = t_name.strip().lower()
        base_wr = overall_wr_map.get(tk, 0.50)
        pool[t_name] = {}
        for m in ACTIVE_MAPS:
            cnt, wr = raw.get(tk, {}).get(m, (0, 0.0))
            if cnt > 0:
                eff_wr = (cnt * wr + 6.0 * base_wr) / (cnt + 6.0)
            else:
                eff_wr = max(0.18, base_wr - 0.14)
            pool[t_name][m] = {
                "maps_played": cnt,
                "raw_wr": wr if cnt > 0 else np.nan,
                "eff_wr": float(eff_wr),
            }
    return pool


def simulate_veto(
    team_a: str,
    team_b: str,
    match_format: str,
    map_pool: dict[str, dict[str, dict]],
) -> dict:
    rem = ACTIVE_MAPS.copy()
    steps: list[dict] = []

    def best_ban_for(chooser: str, opponent: str, candidates: list[str]) -> str:
        return min(
            candidates,
            key=lambda m: (
                map_pool[chooser][m]["eff_wr"] - 0.65 * map_pool[opponent][m]["eff_wr"],
                map_pool[chooser][m]["maps_played"],
            ),
        )

    def best_pick_for(chooser: str, opponent: str, candidates: list[str]) -> str:
        return max(
            candidates,
            key=lambda m: (
                map_pool[chooser][m]["eff_wr"] - 0.55 * map_pool[opponent][m]["eff_wr"],
                map_pool[chooser][m]["maps_played"],
            ),
        )

    fmt = match_format.strip().upper()
    if fmt == "BO1":
        sequence = [
            (team_a, team_b, "Ban"),
            (team_a, team_b, "Ban"),
            (team_b, team_a, "Ban"),
            (team_b, team_a, "Ban"),
            (team_b, team_a, "Ban"),
            (team_a, team_b, "Ban"),
        ]
        for chooser, opp, action in sequence:
            chosen = best_ban_for(chooser, opp, rem)
            rem.remove(chosen)
            steps.append({"Step": f"{chooser} {action}", "Map": chosen})
        decider = rem[0]
        steps.append({"Step": "Decider (BO1)", "Map": decider})
        return {"played_maps": [decider], "steps": steps}

    b1 = best_ban_for(team_a, team_b, rem); rem.remove(b1); steps.append({"Step": f"{team_a} Ban", "Map": b1})
    b2 = best_ban_for(team_b, team_a, rem); rem.remove(b2); steps.append({"Step": f"{team_b} Ban", "Map": b2})
    p1 = best_pick_for(team_a, team_b, rem); rem.remove(p1); steps.append({"Step": f"{team_a} Pick (Map 1)", "Map": p1})
    p2 = best_pick_for(team_b, team_a, rem); rem.remove(p2); steps.append({"Step": f"{team_b} Pick (Map 2)", "Map": p2})
    b3 = best_ban_for(team_a, team_b, rem); rem.remove(b3); steps.append({"Step": f"{team_a} Ban", "Map": b3})
    b4 = best_ban_for(team_b, team_a, rem); rem.remove(b4); steps.append({"Step": f"{team_b} Ban", "Map": b4})
    dec = rem[0]
    steps.append({"Step": "Decider (Map 3)", "Map": dec})
    return {"played_maps": [p1, p2, dec], "steps": steps}


def fit_v4_model(
    train_df: pd.DataFrame,
    stats_csv: str,
    reg_c: float = 0.35,
    decay: float = 0.003,
) -> tuple[Pipeline, dict[str, float]]:
    X_tr = build_features_v4(train_df, stats_csv=stats_csv, fixed_as_of=None)
    X_tr_sw = swap_features_v4(X_tr)

    y_tr = train_df["winner"].astype(int).values
    y_all = np.concatenate([y_tr, 1 - y_tr])
    X_all = pd.concat([X_tr, X_tr_sw], ignore_index=True)

    days_ago = (train_df["date_dt"].max() - train_df["date_dt"]).dt.days.clip(lower=0).values
    w_all = np.concatenate([np.exp(-decay * days_ago), np.exp(-decay * days_ago)])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=reg_c, max_iter=1000)),
    ])
    pipe.fit(X_all.values, y_all, clf__sample_weight=w_all)

    coefs = dict(zip(FEATURE_COLS_V4, pipe.named_steps["clf"].coef_[0]))
    return pipe, coefs


def predict_base_prob(
    pipe: Pipeline,
    team_a: str,
    team_b: str,
    rank_a: int,
    rank_b: int,
    wr3_a: float,
    wr3_b: float,
    match_format: str,
    stats_csv: str,
    as_of_date: str,
) -> float:
    one = pd.DataFrame([{
        "date": as_of_date,
        "event_name": "PGL Major Singapore 2026",
        "event_tier": "Major",
        "stage": "Stage 1",
        "format": match_format,
        "team_a": team_a,
        "team_b": team_b,
        "rank_a": rank_a,
        "rank_b": rank_b,
        "winrate_3m_a": wr3_a,
        "winrate_3m_b": wr3_b,
    }])
    x_row = build_features_v4(one, stats_csv=stats_csv, fixed_as_of=as_of_date)
    return float(np.clip(sigmoid(pipe.decision_function(x_row.values)[0]), 0.05, 0.95))


def predict_match_with_veto(
    pipe: Pipeline,
    team_a: str,
    team_b: str,
    rank_a: int,
    rank_b: int,
    wr3_a: float,
    wr3_b: float,
    match_format: str,
    stats_csv: str,
    as_of_date: str,
    map_pool: dict[str, dict[str, dict]],
    map_weight: float = 0.35,
) -> dict:
    fmt = match_format.strip().upper()
    p_base = predict_base_prob(pipe, team_a, team_b, rank_a, rank_b, wr3_a, wr3_b, fmt, stats_csv, as_of_date)
    veto = simulate_veto(team_a, team_b, fmt, map_pool)
    played = veto["played_maps"]

    map_details = []
    map_probs = []
    base_l = logit(p_base)
    for m in played:
        wa = map_pool[team_a][m]["eff_wr"]
        wb = map_pool[team_b][m]["eff_wr"]
        adj_l = base_l + map_weight * 4.0 * (wa - wb)
        pm = float(np.clip(sigmoid(adj_l), 0.05, 0.95))
        map_probs.append(pm)
        map_details.append({
            "Map": m,
            f"{team_a} Eff WR %": round(wa * 100.0, 1),
            f"{team_b} Eff WR %": round(wb * 100.0, 1),
            f"P({team_a}) %": round(pm * 100.0, 1),
        })

    if fmt == "BO1" or len(map_probs) == 1:
        p_series = map_probs[0]
    else:
        p1, p2, p3 = map_probs[0], map_probs[1], map_probs[2]
        p_series = (
            p1 * p2
            + p1 * (1.0 - p2) * p3
            + (1.0 - p1) * p2 * p3
        )

    return {
        "p_base": p_base,
        "p_series": float(np.clip(p_series, 0.05, 0.95)),
        "played_maps": played,
        "veto_steps": veto["steps"],
        "map_details": map_details,
    }


def build_singapore_teams_df(stats_csv: str, matches_df: pd.DataFrame) -> pd.DataFrame:
    stats_df = pd.read_csv(stats_csv)
    rows = []
    for idx, t in enumerate(SINGAPORE_S1_SEEDS, start=1):
        sub = stats_df[stats_df["team"].astype(str).str.strip().str.lower() == t.lower()]
        if not sub.empty:
            r0 = sub.iloc[-1]
            rows.append({
                "seed": idx,
                "team": t,
                "vrs_rank": int(r0["vrs_rank"]),
                "vrs_points": int(r0["vrs_points"]),
                "maps_played": int(r0["maps_played"]),
                "map_winrate": float(r0["map_winrate"]),
                "top20_maps": int(r0["top20_maps"]),
                "top20_winrate": float(r0["top20_winrate"]),
                "round_winrate": float(r0["round_winrate"]),
                "team_rating": float(r0["team_rating"]),
            })
        else:
            m_sub = matches_df[(matches_df["team_a"] == t) | (matches_df["team_b"] == t)]
            last_m = m_sub.iloc[-1]
            rk = int(last_m["rank_a"] if last_m["team_a"] == t else last_m["rank_b"])
            wr = float(last_m["winrate_3m_a"] if last_m["team_a"] == t else last_m["winrate_3m_b"])
            rows.append({
                "seed": idx,
                "team": t,
                "vrs_rank": rk,
                "vrs_points": 1300,
                "maps_played": 20,
                "map_winrate": wr,
                "top20_maps": 0,
                "top20_winrate": 0.0,
                "round_winrate": 0.50,
                "team_rating": 1.00,
            })
    return pd.DataFrame(rows)


def simulate_swiss_16(
    pipe: Pipeline,
    teams_df: pd.DataFrame,
    stats_csv: str,
    as_of_date: str,
    map_pool: dict[str, dict[str, dict]],
    map_weight: float = 0.35,
    n_sims: int = 10000,
    seed: int = 42,
    overqual_cap: float = 85.0,
    overqual_mult: float = 1.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(teams_df)
    teams = teams_df["team"].tolist()
    ranks = teams_df["vrs_rank"].tolist()
    wrs = teams_df["map_winrate"].tolist()

    p_bo1 = np.zeros((n, n))
    p_bo3 = np.zeros((n, n))
    r1_rows = []
    half = n // 2

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            res1 = predict_match_with_veto(
                pipe, teams[i], teams[j], ranks[i], ranks[j], wrs[i], wrs[j],
                "BO1", stats_csv, as_of_date, map_pool, map_weight
            )
            res3 = predict_match_with_veto(
                pipe, teams[i], teams[j], ranks[i], ranks[j], wrs[i], wrs[j],
                "BO3", stats_csv, as_of_date, map_pool, map_weight
            )
            p_bo1[i, j] = res1["p_series"]
            p_bo3[i, j] = res3["p_series"]

            if j == i + half and i < half:
                pa = round(res1["p_series"] * 100.0, 1)
                pb_base = round(res1["p_base"] * 100.0, 1)
                dec_map = res1["played_maps"][0]
                wa_m = round(map_pool[teams[i]][dec_map]["eff_wr"] * 100.0, 1)
                wb_m = round(map_pool[teams[j]][dec_map]["eff_wr"] * 100.0, 1)
                r1_rows.append({
                    "Match": f"#{i+1} {teams[i]} vs #{j+1} {teams[j]}",
                    "Veto Decider": dec_map,
                    "Map WR (A vs B)": f"{wa_m}% vs {wb_m}%",
                    "V4 Base %": f"{pb_base}%",
                    "Final P(A) %": pa,
                    "Final P(B) %": round(100.0 - pa, 1),
                    "Favorite": teams[i] if pa >= 50.0 else teams[j],
                })

    c_3_0 = np.zeros(n)
    c_qual = np.zeros(n)
    c_3_1_2 = np.zeros(n)
    c_0_3 = np.zeros(n)

    rng = np.random.default_rng(seed)
    for _ in range(n_sims):
        wins = np.zeros(n, dtype=int)
        losses = np.zeros(n, dtype=int)
        opponents: list[list[int]] = [[] for _ in range(n)]
        played: set[tuple[int, int]] = set()

        for i in range(half):
            t1, t2 = i, i + half
            played.update({(t1, t2), (t2, t1)})
            opponents[t1].append(t2)
            opponents[t2].append(t1)
            if rng.random() < p_bo1[t1, t2]:
                wins[t1] += 1
                losses[t2] += 1
            else:
                wins[t2] += 1
                losses[t1] += 1

        for _ in range(2, 6):
            active = [i for i in range(n) if wins[i] < 3 and losses[i] < 3]
            if not active:
                break
            buchholz = {i: sum(int(wins[o]) - int(losses[o]) for o in opponents[i]) for i in active}
            pools: dict[tuple[int, int], list[int]] = {}
            for idx in active:
                pools.setdefault((int(wins[idx]), int(losses[idx])), []).append(idx)

            for (w_rec, l_rec), pool in sorted(pools.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
                pool.sort(key=lambda idx: (-buchholz[idx], idx))
                is_decisive = (w_rec == 2 or l_rec == 2)
                mat = p_bo3 if is_decisive else p_bo1
                unpaired = pool.copy()
                while len(unpaired) >= 2:
                    t1 = unpaired.pop(0)
                    opp_pos = len(unpaired) - 1
                    while opp_pos > 0 and (t1, unpaired[opp_pos]) in played:
                        opp_pos -= 1
                    t2 = unpaired.pop(opp_pos)
                    played.update({(t1, t2), (t2, t1)})
                    opponents[t1].append(t2)
                    opponents[t2].append(t1)
                    if rng.random() < mat[t1, t2]:
                        wins[t1] += 1
                        losses[t2] += 1
                    else:
                        wins[t2] += 1
                        losses[t1] += 1

        for i in range(n):
            if wins[i] == 3 and losses[i] == 0:
                c_3_0[i] += 1
                c_qual[i] += 1
            elif wins[i] == 3:
                c_qual[i] += 1
                c_3_1_2[i] += 1
            elif losses[i] == 3 and wins[i] == 0:
                c_0_3[i] += 1

    p_3_0 = (c_3_0 / n_sims) * 100.0
    p_qual = (c_qual / n_sims) * 100.0
    p_3_1_2 = (c_3_1_2 / n_sims) * 100.0
    p_0_3 = (c_0_3 / n_sims) * 100.0

    overqual_penalty = np.where(p_qual > overqual_cap, (p_qual - overqual_cap) * overqual_mult, 0.0)
    score_3_0 = p_3_0 * (p_3_0 / np.maximum(p_3_1_2, 1.0)) - overqual_penalty

    sim_df = pd.DataFrame({
        "Seed": teams_df["seed"].values,
        "Team": teams,
        "VRS Rank": ranks,
        "Rating": teams_df["team_rating"].values,
        "Winrate 3M %": np.round(np.array(wrs) * 100.0, 1),
        "3-0 %": np.round(p_3_0, 1),
        "Score 3-0": np.round(score_3_0, 2),
        "3-1/3-2 %": np.round(p_3_1_2, 1),
        "0-3 %": np.round(p_0_3, 1),
        "Qual %": np.round(p_qual, 1),
    })
    return sim_df, pd.DataFrame(r1_rows)


def pick_10_slots(sim_df: pd.DataFrame) -> dict[str, list[str]]:
    p_3_0 = sim_df.sort_values("Score 3-0", ascending=False).head(2)["Team"].tolist()
    p_0_3 = sim_df.sort_values("0-3 %", ascending=False).head(2)["Team"].tolist()
    excluded = set(p_3_0).union(set(p_0_3))
    p_3_1_2 = (
        sim_df[~sim_df["Team"].isin(excluded)]
        .sort_values("3-1/3-2 %", ascending=False)
        .head(6)["Team"]
        .tolist()
    )
    return {"3-0": p_3_0, "3-1 / 3-2": p_3_1_2, "0-3": p_0_3}