import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features_v4 import build_features_v4, swap_features_v4


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def fit_v4_model(train_df: pd.DataFrame, stats_csv: str, reg_c: float = 0.35, decay: float = 0.003) -> Pipeline:
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
    return pipe


def match_prob_v4(
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
    return float(np.clip(sigmoid(pipe.decision_function(x_row.values))[0], 0.05, 0.95))


def simulate_swiss_16(
    pipe: Pipeline,
    teams_df: pd.DataFrame,
    stats_csv: str,
    as_of_date: str,
    n_sims: int = 10000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(teams_df)
    teams = teams_df["team"].tolist()
    ranks = teams_df["rank"].tolist()
    wrs = teams_df["wr_3m"].tolist()

    p_bo1 = np.zeros((n, n))
    p_bo3 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p_bo1[i, j] = match_prob_v4(pipe, teams[i], teams[j], ranks[i], ranks[j], wrs[i], wrs[j], "BO1", stats_csv, as_of_date)
            p_bo3[i, j] = match_prob_v4(pipe, teams[i], teams[j], ranks[i], ranks[j], wrs[i], wrs[j], "BO3", stats_csv, as_of_date)

    half = n // 2
    r1_rows = []
    for i in range(half):
        t1, t2 = i, i + half
        pa = p_bo1[t1, t2] * 100.0
        r1_rows.append({
            "match": f"#{i+1} {teams[t1]} vs #{i+1+half} {teams[t2]}",
            "format": "BO1",
            "fav": teams[t1] if pa >= 50.0 else teams[t2],
            "p_team_a": round(pa, 1),
            "p_team_b": round(100.0 - pa, 1),
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

    overqual_penalty = np.where(p_qual > 85.0, (p_qual - 85.0) * 1.2, 0.0)
    score_3_0 = p_3_0 * (p_3_0 / np.maximum(p_3_1_2, 1.0)) - overqual_penalty

    sim_df = pd.DataFrame({
        "seed": range(1, n + 1),
        "team": teams,
        "vrs_rank": ranks,
        "p_3_0": p_3_0,
        "score_3_0": score_3_0,
        "p_3_1_3_2": p_3_1_2,
        "p_qual": p_qual,
        "p_0_3": p_0_3,
    })
    return sim_df, pd.DataFrame(r1_rows)


def pick_10_slots(sim_df: pd.DataFrame) -> dict[str, list[str]]:
    p_3_0 = sim_df.sort_values("score_3_0", ascending=False).head(2)["team"].tolist()
    p_0_3 = sim_df.sort_values("p_0_3", ascending=False).head(2)["team"].tolist()
    excluded = set(p_3_0).union(set(p_0_3))
    p_3_1_2 = (
        sim_df[~sim_df["team"].isin(excluded)]
        .sort_values("p_3_1_3_2", ascending=False)
        .head(6)["team"]
        .tolist()
    )
    return {"3-0": p_3_0, "3-1 / 3-2": p_3_1_2, "0-3": p_0_3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", default="data/matches.csv")
    parser.add_argument("--stats", default="data/vrs_team_stats_singapore.csv")
    parser.add_argument("--as-of", default="2026-11-23")
    parser.add_argument("--sims", type=int, default=10000)
    args = parser.parse_args()

    df = pd.read_csv(args.matches)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt", "winner"]).sort_values("date_dt", ascending=True).reset_index(drop=True)

    # Folosim toate datele până la as_of (include Cologne + turneele de toamnă 2026)
    train_df = df[df["date_dt"] < pd.to_datetime(args.as_of)].copy().reset_index(drop=True)
    pipe = fit_v4_model(train_df, stats_csv=args.stats)

    # Ordinea oficială de seeding din HLTV Stage 1 (seeds 1..16)
    official_seed_order = [
        "PARIVISION", "Liquid", "Ninjas in Pyjamas", "FaZe",
        "9z", "Eternal Fire", "HEROIC", "M80",
        "100 Thieves", "DENDELE", "Wildcard", "Lynn Vision",
        "BC.Game", "Imperial", "FlyQuest", "Chinggis Warriors",
    ]
    stats_df = pd.read_csv(args.stats)
    rows = []
    for t in official_seed_order:
        sub = stats_df[stats_df["team"].astype(str).str.strip().str.lower() == t.lower()]
        r0 = sub.iloc[-1]
        rows.append({
            "team": t,
            "rank": int(r0["vrs_rank"]),
            "wr_3m": float(r0["map_winrate"]),
        })

    teams_df = pd.DataFrame(rows)
    sim_df, r1_df = simulate_swiss_16(pipe, teams_df, stats_csv=args.stats, as_of_date=args.as_of, n_sims=args.sims)
    picks = pick_10_slots(sim_df)

    print(f"=== PGL MAJOR SINGAPORE 2026 — STAGE 1 (trained on n={len(train_df)} matches) ===\n")
    print("--- ROUND 1 OPENING MATCHUPS (BO1) ---")
    print(r1_df.to_string(index=False))

    print("\n--- SWISS MONTE CARLO PROBABILITIES (10,000 SIMS) ---")
    print(sim_df[["seed", "team", "vrs_rank", "p_3_0", "p_3_1_3_2", "p_0_3", "p_qual"]].round(1).to_string(index=False))

    print("\n--- RECOMMENDED PICK'EM (10 SLOTS) ---")
    for k, v in picks.items():
        print(f"  {k:<10}: {v}")


if __name__ == "__main__":
    main()