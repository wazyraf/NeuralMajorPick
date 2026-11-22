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
        "event_name": "IEM Cologne Major 2026",
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
    n_sims: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
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

    c_3_0 = np.zeros(n)
    c_qual = np.zeros(n)
    c_3_1_2 = np.zeros(n)
    c_0_3 = np.zeros(n)

    rng = np.random.default_rng(seed)
    half = n // 2

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

    # Penalizăm ușor echipele cu p_qual > 85% (mai expuse la "3-0 curse") și favorizăm raportul p_3_0 / p_3_1_2
    overqual_penalty = np.where(p_qual > 85.0, (p_qual - 85.0) * 1.2, 0.0)
    score_3_0 = p_3_0 * (p_3_0 / np.maximum(p_3_1_2, 1.0)) - overqual_penalty

    return pd.DataFrame({
        "seed": range(1, n + 1),
        "team": teams,
        "vrs_rank": ranks,
        "wr_3m": wrs,
        "p_3_0": p_3_0,
        "score_3_0": score_3_0,
        "p_3_1_3_2": p_3_1_2,
        "p_qual": p_qual,
        "p_0_3": p_0_3,
    })


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


def actual_records_from_matches(df: pd.DataFrame, teams: list[str]) -> dict[str, str]:
    s1 = df[
        df["event_name"].astype(str).str.contains("Cologne", case=False, na=False)
        & df["stage"].astype(str).str.contains("Stage 1", case=False, na=False)
    ]
    w_map = {t: 0 for t in teams}
    l_map = {t: 0 for t in teams}
    for _, r in s1.iterrows():
        ta, tb = str(r["team_a"]), str(r["team_b"])
        win_a = int(r["winner"]) == 1
        if ta in w_map and tb in w_map:
            if win_a:
                w_map[ta] += 1
                l_map[tb] += 1
            else:
                w_map[tb] += 1
                l_map[ta] += 1
    return {t: f"{w_map[t]}-{l_map[t]}" for t in teams}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", default="data/matches.csv")
    parser.add_argument("--stats", default="data/vrs_team_stats_cologne.csv")
    parser.add_argument("--cutoff", default="2026-06-02")
    parser.add_argument("--sims", type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_csv(args.matches)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt", "winner"]).sort_values("date_dt", ascending=True).reset_index(drop=True)

    train_df = df[df["date_dt"] < pd.to_datetime(args.cutoff)].copy().reset_index(drop=True)
    pipe = fit_v4_model(train_df, stats_csv=args.stats)

    stats_df = pd.read_csv(args.stats)
    s1_teams_raw = [
        "Lynn Vision", "M80", "SINNERS", "FlyQuest", "MIBR", "THUNDER dOWNUNDER",
        "TYLOO", "B8", "HEROIC", "Sharks", "NRG", "GamerLegion",
        "Gaimin Gladiators", "BETBOOM", "Liquid", "BIG"
    ]
    rows = []
    for t in s1_teams_raw:
        sub = stats_df[stats_df["team"].astype(str).str.strip().str.lower() == t.lower()]
        if not sub.empty:
            r0 = sub.iloc[-1]
            rk = int(r0["vrs_rank"])
            wr = float(r0["map_winrate"])
        else:
            m_sub = df[(df["team_a"] == t) | (df["team_b"] == t)]
            last_m = m_sub.iloc[-1]
            rk = int(last_m["rank_a"] if last_m["team_a"] == t else last_m["rank_b"])
            wr = float(last_m["winrate_3m_a"] if last_m["team_a"] == t else last_m["winrate_3m_b"])
        rows.append({"team": t, "rank": rk, "wr_3m": wr})

    teams_df = pd.DataFrame(rows).sort_values("rank", ascending=True).reset_index(drop=True)
    sim_df = simulate_swiss_16(pipe, teams_df, stats_csv=args.stats, as_of_date=args.cutoff, n_sims=args.sims)
    picks = pick_10_slots(sim_df)
    actual_rec = actual_records_from_matches(df, teams_df["team"].tolist())

    sim_df["actual_rec"] = sim_df["team"].map(actual_rec)
    print(sim_df[["seed", "team", "vrs_rank", "p_3_0", "p_3_1_3_2", "p_0_3", "p_qual", "actual_rec"]].round(1).to_string(index=False))
    print("\nPREDICTED PICK'EM 10:", picks)

    actual_3_0 = {t for t, rec in actual_rec.items() if rec == "3-0"}
    actual_0_3 = {t for t, rec in actual_rec.items() if rec == "0-3"}
    actual_adv_any = {t for t, rec in actual_rec.items() if rec.startswith("3-")}
    actual_3_1_2 = actual_adv_any - actual_3_0

    hit_3_0 = len(set(picks["3-0"]) & actual_3_0)
    hit_3_1_2 = len(set(picks["3-1 / 3-2"]) & actual_3_1_2)
    hit_0_3 = len(set(picks["0-3"]) & actual_0_3)
    total_hits = hit_3_0 + hit_3_1_2 + hit_0_3

    print(f"ACTUAL 3-0   : {sorted(actual_3_0)} -> hits: {hit_3_0}/2")
    print(f"ACTUAL 3-1/32: {sorted(actual_3_1_2)} -> hits vs 6 slots: {hit_3_1_2}/6")
    print(f"ACTUAL 0-3   : {sorted(actual_0_3)} -> hits: {hit_0_3}/2")
    print(f"TOTAL SCORE  : {total_hits}/10")


if __name__ == "__main__":
    main()