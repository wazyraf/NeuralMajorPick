from itertools import combinations
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PAST_FEATURE_COLS = [
    "rating",
    "multikill_pct",
    "entry_pct",
    "support_pct",
    "clutch_pct",
    "awp",
    "dpr",
    "price_k",
]


def load_fantasy_players(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    num_cols = [
        "price", "rating", "ct_rating", "t_rating",
        "awp", "entry_pct", "clutch_pct", "support_pct", "multikill_pct", "dpr",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["player", "team", "price", "rating"]).reset_index(drop=True)


def fit_fantasy_model_from_past(past_csv: str) -> tuple[dict[str, float], int]:
    """
    Învață ponderi pe statisticile jucătorilor din istoricul Fantasy (`actual_points`).
    Ignoră automat rândurile fără rating valid (`rating > 0`).
    """
    default_weights = {
        "rating": 14.0,
        "multikill_pct": 0.65,
        "entry_pct": 0.35,
        "support_pct": -0.25,
        "clutch_pct": 1.20,
        "awp": 3.50,
        "dpr": -18.0,
        "price_k": 0.15,
    }
    if not past_csv or not os.path.exists(past_csv):
        return default_weights, 0

    pdf = pd.read_csv(past_csv)
    if "actual_points" not in pdf.columns or "rating" not in pdf.columns:
        return default_weights, 0

    pdf["actual_points"] = pd.to_numeric(pdf["actual_points"], errors="coerce")
    for c in ["rating", "multikill_pct", "entry_pct", "support_pct", "clutch_pct", "awp", "dpr", "price"]:
        if c in pdf.columns:
            pdf[c] = pd.to_numeric(pdf[c], errors="coerce")

    valid = pdf.dropna(subset=["actual_points", "rating"]).copy()
    valid = valid[valid["rating"] > 0.1].copy()
    if len(valid) < 8:
        return default_weights, len(valid)

    valid["pts_per_map"] = valid["actual_points"] / 7.0
    valid["price_k"] = valid["price"] / 1000.0

    X_raw = valid[["rating", "multikill_pct", "entry_pct", "support_pct", "clutch_pct", "awp", "dpr", "price_k"]].copy()
    for c in X_raw.columns:
        col_med = float(X_raw[c].median()) if X_raw[c].notna().any() else 0.0
        X_raw[c] = X_raw[c].fillna(col_med)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw.values)
    y = valid["pts_per_map"].to_numpy()

    ridge = Ridge(alpha=25.0)
    ridge.fit(X_sc, y)

    learned = {}
    scales = np.maximum(scaler.scale_, 1e-6)
    for feat, coef_sc, sc in zip(PAST_FEATURE_COLS, ridge.coef_, scales):
        learned[feat] = float(coef_sc / sc)

    blended = {}
    for k, base_v in default_weights.items():
        if k in learned:
            blended[k] = round(0.60 * learned[k] + 0.40 * base_v, 4)
        else:
            blended[k] = base_v

    return blended, len(valid)


def score_fantasy_players(
    players_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    learned_weights: dict[str, float] | None = None,
    qual_bonus_weight: float = 0.70,
    team_pull_strength: float = 0.55,
) -> pd.DataFrame:
    """Calculează punctele așteptate în Swiss punând greutate reală pe șansele echipei și volumul de hărți."""
    w_map = learned_weights or {
        "rating": 14.0,
        "multikill_pct": 0.65,
        "entry_pct": 0.35,
        "support_pct": -0.25,
        "clutch_pct": 1.20,
        "awp": 3.50,
        "dpr": -18.0,
        "price_k": 0.15,
    }

    s_lookup: dict[str, dict[str, float]] = {}
    for _, r in sim_df.iterrows():
        tk = str(r["Team"]).strip().lower()
        p30 = float(r["3-0 %"]) / 100.0
        p312 = float(r["3-1/3-2 %"]) / 100.0
        p03 = float(r["0-3 %"]) / 100.0
        pqual = float(r["Qual %"]) / 100.0
        prem = max(0.0, 1.0 - p30 - p312 - p03)

        exp_maps = 6.2 * p30 + 9.2 * p312 + 5.2 * p03 + 7.4 * prem
        exp_wins = 3.0 * (p30 + p312) + 2.0 * (prem * 0.55) + 1.0 * (prem * 0.45)

        s_lookup[tk] = {
            "p_3_0": float(r["3-0 %"]),
            "p_3_1_2": float(r["3-1/3-2 %"]),
            "p_0_3": float(r["0-3 %"]),
            "p_qual": float(r["Qual %"]),
            "team_rating": float(r.get("Rating", 1.0)),
            "exp_maps": exp_maps,
            "exp_wins": exp_wins,
        }

    out = players_df.copy()
    exp_maps_list = []
    exp_wins_list = []
    p30_list = []
    p312_list = []
    p03_list = []
    pqual_list = []
    team_rat_list = []

    for _, r in out.iterrows():
        st_info = s_lookup.get(
            str(r["team"]).strip().lower(),
            {
                "p_3_0": 10.0, "p_3_1_2": 40.0, "p_0_3": 12.0,
                "p_qual": 50.0, "team_rating": 1.0, "exp_maps": 7.4, "exp_wins": 1.6,
            },
        )
        exp_maps_list.append(st_info["exp_maps"])
        exp_wins_list.append(st_info["exp_wins"])
        p30_list.append(st_info["p_3_0"])
        p312_list.append(st_info["p_3_1_2"])
        p03_list.append(st_info["p_0_3"])
        pqual_list.append(st_info["p_qual"])
        team_rat_list.append(st_info["team_rating"])

    out["Team 3-0 %"] = np.round(p30_list, 1)
    out["Team 3-1/3-2 %"] = np.round(p312_list, 1)
    out["Team 0-3 %"] = np.round(p03_list, 1)
    out["Team Qual %"] = np.round(pqual_list, 1)
    out["Team Rating"] = np.round(team_rat_list, 2)
    out["Exp Maps"] = np.round(exp_maps_list, 2)
    out["Exp Wins"] = np.round(exp_wins_list, 2)

    qual_factor = np.clip(out["Team Qual %"] / 65.0, 0.50, 1.15)
    above_1 = np.maximum(0.0, out["rating"] - 1.00)
    below_1 = np.minimum(0.0, out["rating"] - 1.00)
    adj_rating = (
        1.00
        + above_1 * (1.0 - team_pull_strength * np.maximum(0.0, 1.0 - qual_factor))
        + below_1 * (1.0 + 0.30 * team_pull_strength * np.maximum(0.0, 1.0 - qual_factor))
    )
    out["Adj Rating"] = np.round(adj_rating, 3)

    rating_pts = out["Exp Maps"] * (out["Adj Rating"] - 1.00) * 45.0
    win_pts = out["Exp Wins"] * 8.0
    qual_pts = ((out["Team Qual %"] - 45.0) / 10.0) * (qual_bonus_weight * 7.0) + (out["Team 3-1/3-2 %"] * 0.05 * qual_bonus_weight)

    stat_per_map = (
        w_map.get("rating", 14.0) * (out["Adj Rating"] - 1.05)
        + w_map.get("multikill_pct", 0.65) * (out["multikill_pct"] - 16.5)
        + w_map.get("entry_pct", 0.35) * (out["entry_pct"] - 10.5)
        + w_map.get("support_pct", -0.25) * (out["support_pct"] - 21.5)
        + w_map.get("clutch_pct", 1.20) * (out["clutch_pct"] - 1.2)
        + w_map.get("awp", 3.50) * np.maximum(0.0, out["awp"] - 0.25)
        + w_map.get("dpr", -18.0) * (out["dpr"] - 0.64)
    )
    stat_pts = np.clip(stat_per_map * 0.35, -3.5, 5.5)

    out["Rating Pts"] = np.round(rating_pts, 2)
    out["Win Pts"] = np.round(win_pts, 2)
    out["Qual Bonus"] = np.round(qual_pts, 2)
    out["Stat Edge"] = np.round(stat_pts, 2)
    out["Exp Pts"] = np.round(rating_pts + win_pts + qual_pts + stat_pts, 2)
    out["Pts / $100k"] = np.round(out["Exp Pts"] / (out["price"] / 100000.0), 2)

    return out.sort_values("Exp Pts", ascending=False).reset_index(drop=True)


def _is_valid_lineup(combo: tuple[int, ...], prices: np.ndarray, teams: list[str], budget: int, max_per_team: int) -> bool:
    if int(prices[list(combo)].sum()) > budget:
        return False
    counts: dict[str, int] = {}
    for idx in combo:
        tm = teams[idx]
        c = counts.get(tm, 0) + 1
        if c > max_per_team:
            return False
        counts[tm] = c
    return True


def optimize_fantasy_lineups(
    scored_df: pd.DataFrame,
    budget: int = 1000000,
    team_size: int = 5,
    max_per_team: int = 2,
    pool_top_n: int = 36,
) -> dict[str, pd.DataFrame]:
    """Găsește lineup-uri optime de 5 jucători sub buget și limita max per echipă."""
    cols = ["player", "team", "price", "rating", "Adj Rating", "Team Qual %", "Team 3-0 %", "Exp Maps", "Exp Pts", "Pts / $100k"]
    if len(scored_df) < team_size:
        return {
            "Top Contenders (Qual & Form)": pd.DataFrame(columns=cols),
            "Best Value ($ per Pt)": pd.DataFrame(columns=cols),
            "3-0 & Fast Qual Push": pd.DataFrame(columns=cols),
        }

    cand = scored_df.sort_values("Exp Pts", ascending=False).head(min(pool_top_n, len(scored_df))).reset_index(drop=True)
    n = len(cand)
    prices = cand["price"].astype(int).to_numpy()
    teams = cand["team"].astype(str).tolist()
    exp_pts = cand["Exp Pts"].astype(float).to_numpy()
    val_pts = cand["Pts / $100k"].astype(float).to_numpy()
    qual_pct = cand["Team Qual %"].astype(float).to_numpy()
    p30_pct = cand["Team 3-0 %"].astype(float).to_numpy()

    best_max: tuple[float, tuple[int, ...]] | None = None
    best_val: tuple[float, tuple[int, ...]] | None = None
    best_up: tuple[float, tuple[int, ...]] | None = None

    for combo in combinations(range(n), team_size):
        if not _is_valid_lineup(combo, prices, teams, budget, max_per_team):
            continue
        idx_l = list(combo)
        tot_exp = float(exp_pts[idx_l].sum())
        tot_qual = float(qual_pct[idx_l].sum())
        tot_30 = float(p30_pct[idx_l].sum())
        uniq_teams = len({teams[i] for i in idx_l})

        contender_score = tot_exp + 0.45 * tot_qual
        value_score = tot_exp + 2.2 * float(val_pts[idx_l].sum()) + 0.50 * tot_qual + 0.5 * uniq_teams
        push_score = tot_exp + 0.30 * tot_30 + 0.30 * tot_qual

        if best_max is None or contender_score > best_max[0]:
            best_max = (contender_score, combo)
        if best_val is None or value_score > best_val[0]:
            best_val = (value_score, combo)
        if best_up is None or push_score > best_up[0]:
            best_up = (push_score, combo)

    result: dict[str, pd.DataFrame] = {}
    for label, item in [
        ("Top Contenders (Qual & Form)", best_max),
        ("Best Value ($ per Pt)", best_val),
        ("3-0 & Fast Qual Push", best_up),
    ]:
        if item is not None:
            sub = cand.iloc[list(item[1])][cols].sort_values("Exp Pts", ascending=False).reset_index(drop=True)
            result[label] = sub
        else:
            result[label] = pd.DataFrame(columns=cols)
    return result