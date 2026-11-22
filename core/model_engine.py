import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from core.team_stats_engine import (
    load_team_stats,
    compute_map_veto_edge,
    assess_match_risk,
    compute_h2h_history,
)

FEATURE_COLS = [
    "rank_diff",
    "log_rank_ratio",
    "wr_diff",
    "is_bo1",
    "is_major",
]


class CS2MatchNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


def _shrink_by_sample(value: float | None, prior: float, n_maps: float | None, full_sample: float = 30.0) -> float:
    if value is None or np.isnan(value):
        return prior
    n = max(0.0, float(n_maps or 0.0))
    w = min(1.0, n / full_sample)
    return float(w * value + (1.0 - w) * prior)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=df.index)
    features["rank_diff"] = df["rank_b"] - df["rank_a"]
    features["log_rank_ratio"] = np.log1p(df["rank_b"]) - np.log1p(df["rank_a"])
    features["wr_diff"] = df["winrate_3m_a"] - df["winrate_3m_b"]
    features["is_bo1"] = (df["format"] == "BO1").astype(float)
    features["is_major"] = (df["event_tier"] == "Major").astype(float)
    return features[FEATURE_COLS]


def _fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    best_t = 0.85
    best_loss = float("inf")
    y_f = y_true.astype(np.float64)
    for t in np.linspace(0.55, 1.15, 61):
        p = 1.0 / (1.0 + np.exp(-np.clip(logits * t, -15.0, 15.0)))
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        loss = -np.mean(y_f * np.log(p) + (1.0 - y_f) * np.log(1.0 - p))
        if loss < best_loss:
            best_loss = loss
            best_t = float(t)
    return round(best_t, 4)


def load_predictor(model_path="models/cs2_net.pt", scaler_path="models/scaler.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        if checkpoint.get("input_dim") != len(FEATURE_COLS):
            return None, None
        model = CS2MatchNet(len(FEATURE_COLS))
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model.learned_temp = float(checkpoint.get("temperature", 0.85))
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        if getattr(scaler, "n_features_in_", len(FEATURE_COLS)) != len(FEATURE_COLS):
            return None, None
        return model, scaler
    except Exception:
        return None, None


def get_model_metadata(model_path="models/cs2_net.pt"):
    if not os.path.exists(model_path):
        return None
    try:
        checkpoint = torch.load(model_path, weights_only=True)
        if checkpoint.get("input_dim") != len(FEATURE_COLS):
            return None
        return {
            "cutoff_date": checkpoint.get("cutoff_date", "Full Dataset"),
            "n_matches": checkpoint.get("n_matches", "?"),
            "max_train_date": checkpoint.get("max_train_date", "?"),
            "temperature": checkpoint.get("temperature", 0.85),
        }
    except Exception:
        return None


def _compute_volatility(
    prob_a: float,
    match_format: str,
    rank_a: int,
    rank_b: int,
    sa: dict | None,
    sb: dict | None,
    risk_dict: dict,
) -> tuple[float, str, list[str]]:
    reasons: list[str] = []
    closeness = 1.0 - min(1.0, abs(prob_a - 0.5) * 2.0)
    fmt_vol = 0.20 if match_format == "BO1" else (0.07 if match_format == "BO3" else 0.02)
    if match_format == "BO1":
        reasons.append("Format BO1 (variație ridicată)")

    rank_gap = abs(int(rank_a) - int(rank_b))
    rank_vol = max(0.0, 1.0 - min(rank_gap, 40) / 40.0) * 0.12
    if rank_gap <= 8:
        reasons.append(f"Diferență mică de VRS rank ({rank_gap})")

    form_vol = 0.0
    r2_vol = 0.0
    sample_vol = 0.0

    if sa and sb:
        rec_wr_a = sa.get("recent_wr", sa.get("map_wr", 0.5))
        rec_wr_b = sb.get("recent_wr", sb.get("map_wr", 0.5))
        rd_a = sa.get("recent_avg_round_diff", 0.0)
        rd_b = sb.get("recent_avg_round_diff", 0.0)

        if abs(rec_wr_a - rec_wr_b) < 0.08 and abs(rd_a - rd_b) < 1.0:
            form_vol += 0.07
            reasons.append("Formă recentă (20 hărți) apropiată")

        brk_a = _shrink_by_sample(sa.get("big_r2_break"), 0.20, sa.get("big_maps"), 30.0)
        brk_b = _shrink_by_sample(sb.get("big_r2_break"), 0.20, sb.get("big_maps"), 30.0)
        max_brk = max(brk_a, brk_b)
        if max_brk >= 0.28:
            r2_vol += min(0.08, (max_brk - 0.20) * 0.40)
            reasons.append(f"Round 2 break ridicat ({max_brk:.1%})")

        min_big = min(float(sa.get("big_maps", 30) or 30), float(sb.get("big_maps", 30) or 30))
        if min_big < 20:
            sample_vol += ((20.0 - min_big) / 20.0) * 0.08
            reasons.append(f"Eșantion redus la Big events ({int(min_big)} hărți)")

    base_upset = float(risk_dict.get("upset_risk_pct", 25.0)) / 100.0 if isinstance(risk_dict, dict) else 0.25
    vol_score = float(
        np.clip(0.35 * closeness + fmt_vol + rank_vol + form_vol + r2_vol + sample_vol + 0.16 * base_upset, 0.05, 0.92)
    )

    if vol_score >= 0.62:
        vol_level = "High"
    elif vol_score >= 0.38:
        vol_level = "Medium"
    else:
        vol_level = "Low"

    return round(vol_score, 4), vol_level, reasons


def _format_score(prob_a: float, match_format: str) -> str:
    prob_b = 1.0 - prob_a
    if match_format == "BO1":
        if prob_a >= 0.5:
            loser_rounds = int(np.clip(round(13 * (prob_b / prob_a) ** 0.75), 4, 11))
            return f"13 - {loser_rounds}"
        loser_rounds = int(np.clip(round(13 * (prob_a / prob_b) ** 0.75), 4, 11))
        return f"{loser_rounds} - 13"
    if match_format == "BO3":
        if prob_a >= 0.66:
            return "2 - 0"
        if prob_a >= 0.50:
            return "2 - 1"
        if prob_a >= 0.34:
            return "1 - 2"
        return "0 - 2"
    if prob_a >= 0.76:
        return "3 - 0"
    if prob_a >= 0.60:
        return "3 - 1"
    if prob_a >= 0.50:
        return "3 - 2"
    if prob_a >= 0.40:
        return "2 - 3"
    if prob_a >= 0.24:
        return "1 - 3"
    return "0 - 3"


def compute_monotone_logit_from_inputs(
    rank_a: int,
    rank_b: int,
    wr_a: float,
    wr_b: float,
    match_format: str = "BO3",
    event_tier: str = "Major",
    map_edge_a: float = 0.0,
    h2h_edge_a: float = 0.0,
    form_edge_a: float = 0.0,
    temp: float = 0.85,
) -> float:
    """
    Calculează un logit cu direcție garantată:
      - rank mai mic la A (rank_a < rank_b) -> contribuție pozitivă pentru A
      - winrate mai mare la A (wr_a > wr_b) -> contribuție pozitivă pentru A
      - form_edge_a, map_edge_a, h2h_edge_a -> contribuții directe și moderate
    """
    ra = max(1.0, float(rank_a))
    rb = max(1.0, float(rank_b))
    log_rank_term = (np.log1p(rb) - np.log1p(ra)) * 0.75
    wr_term = (float(wr_a) - float(wr_b)) * 2.40
    ctx_term = float(form_edge_a) * 1.30 + float(map_edge_a) * 1.10 + float(h2h_edge_a) * 0.90
    bo1_damp = 0.82 if match_format == "BO1" else 1.0
    major_damp = 0.90 if str(event_tier).strip().lower() == "major" else 1.0

    logit = (log_rank_term + wr_term + ctx_term) * bo1_damp * major_damp * float(temp)
    return float(np.clip(logit, -2.6, 2.6))


def predict_match_context(
    model,
    scaler,
    rank_a: int,
    rank_b: int,
    wr_a: float,
    wr_b: float,
    match_format: str = "BO3",
    event_tier: str = "Major",
    sos_adj_a: float = 0.0,
    sos_adj_b: float = 0.0,
    map_edge_a: float = 0.0,
    h2h_edge_a: float = 0.0,
    form_edge_a: float = 0.0,
):
    eff_wr_a = float(np.clip(wr_a + sos_adj_a, 0.05, 0.95))
    eff_wr_b = float(np.clip(wr_b + sos_adj_b, 0.05, 0.95))
    temp = float(getattr(model, "learned_temp", 0.85)) if model is not None else 0.85

    logit = compute_monotone_logit_from_inputs(
        rank_a=rank_a,
        rank_b=rank_b,
        wr_a=eff_wr_a,
        wr_b=eff_wr_b,
        match_format=match_format,
        event_tier=event_tier,
        map_edge_a=map_edge_a,
        h2h_edge_a=h2h_edge_a,
        form_edge_a=form_edge_a,
        temp=temp,
    )

    prob_a = float(1.0 / (1.0 + np.exp(-logit)))

    if match_format == "BO5":
        roots = np.roots([-2.0, 3.0, 0.0, -prob_a])
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-5 and 0.0 <= r.real <= 1.0]
        p_map = real_roots[0] if real_roots else prob_a
        q = 1.0 - p_map
        prob_a = float((p_map ** 3) * (1.0 + 3.0 * q + 6.0 * (q ** 2)))
    else:
        p_map = prob_a

    prob_a = float(np.clip(prob_a, 0.06, 0.94))
    prob_b = 1.0 - prob_a

    return {
        "prob_a": prob_a,
        "prob_b": prob_b,
        "p_map": float(np.clip(p_map, 0.06, 0.94)),
        "predicted_score": _format_score(prob_a, match_format),
    }


def predict_match_with_team_stats(
    model,
    scaler,
    team_a: str,
    team_b: str,
    rank_a: int,
    rank_b: int,
    fallback_wr_a: float,
    fallback_wr_b: float,
    match_format: str = "BO3",
    event_tier: str = "Major",
    sos_adj_a: float = 0.0,
    sos_adj_b: float = 0.0,
    as_of_date: str | None = None,
    selected_maps: list[str] | None = None,
    manual_map_edge: float | None = None,
    h2h_edge_a: float | None = None,
    team_stats_csv: str = "data/team_stats.csv",
    matches_csv: str = "data/matches.csv",
):
    stats_db = load_team_stats(csv_path=team_stats_csv, as_of_date=as_of_date)
    sa = stats_db.get(team_a.strip().lower())
    sb = stats_db.get(team_b.strip().lower())

    wr_a = float(sa["combined_2026_wr"]) if sa else float(fallback_wr_a)
    wr_b = float(sb["combined_2026_wr"]) if sb else float(fallback_wr_b)

    auto_map_edge = compute_map_veto_edge(sa, sb, selected_maps=selected_maps, match_format=match_format)
    total_map_edge = float(np.clip(manual_map_edge if manual_map_edge is not None else auto_map_edge, -0.25, 0.25))

    h2h_info = compute_h2h_history(
        team_a=team_a,
        team_b=team_b,
        matches_csv=matches_csv,
        as_of_date=as_of_date,
    )
    eff_h2h_edge = float(np.clip(h2h_edge_a if h2h_edge_a is not None else h2h_info["h2h_edge"], -0.20, 0.20))

    form_edge_a = 0.0
    if sa and sb:
        rec_diff = float(sa.get("recent_wr", wr_a) - sb.get("recent_wr", wr_b))
        rd_gap = float((sa.get("recent_avg_round_diff", 0.0) - sb.get("recent_avg_round_diff", 0.0)) / 10.0)
        t20_diff = float(sa.get("top20_wr", wr_a) - sb.get("top20_wr", wr_b))
        form_edge_a = float(np.clip(0.45 * rec_diff + 0.35 * rd_gap + 0.20 * t20_diff, -0.40, 0.40))

    res = predict_match_context(
        model=model,
        scaler=scaler,
        rank_a=rank_a,
        rank_b=rank_b,
        wr_a=wr_a,
        wr_b=wr_b,
        match_format=match_format,
        event_tier=event_tier,
        sos_adj_a=sos_adj_a,
        sos_adj_b=sos_adj_b,
        map_edge_a=total_map_edge,
        h2h_edge_a=eff_h2h_edge,
        form_edge_a=form_edge_a,
    )

    prob_a_cal = res["prob_a"]
    rel_a = sa.get("reliability", 0.75) if sa else 0.75
    rel_b = sb.get("reliability", 0.75) if sb else 0.75

    risk = assess_match_risk(
        prob_a=prob_a_cal,
        match_format=match_format,
        p_map=res["p_map"],
        eco_kd_a=sa.get("eco_kd") if sa else None,
        eco_kd_b=sb.get("eco_kd") if sb else None,
        eco_per_map_a=sa.get("eco_per_map") if sa else None,
        eco_per_map_b=sb.get("eco_per_map") if sb else None,
        reliability_a=rel_a,
        reliability_b=rel_b,
        r2_break_a=sa.get("effective_r2_break") if sa else None,
        r2_break_b=sb.get("effective_r2_break") if sb else None,
        h2h_volatility_bump=h2h_info["h2h_volatility_bump"],
    )

    vol_score, vol_level, vol_reasons = _compute_volatility(
        prob_a=prob_a_cal,
        match_format=match_format,
        rank_a=rank_a,
        rank_b=rank_b,
        sa=sa,
        sb=sb,
        risk_dict=risk if isinstance(risk, dict) else {},
    )
    if h2h_info["h2h_volatility_bump"] > 0.02:
        vol_score = float(np.clip(vol_score + h2h_info["h2h_volatility_bump"], 0.05, 0.95))
        vol_reasons.append(f"Istoric H2H disputat ({h2h_info['h2h_wins_a']}/{h2h_info['h2h_games']} victorii A)")

    return {
        **res,
        **(risk if isinstance(risk, dict) else {}),
        "volatility": round(vol_score, 4),
        "volatility_level": vol_level,
        "volatility_reasons": vol_reasons,
        "h2h_info": h2h_info,
        "used_wr_a": round(wr_a, 4),
        "used_wr_b": round(wr_b, 4),
        "auto_map_edge": round(auto_map_edge, 4),
        "total_map_edge": round(total_map_edge, 4),
        "stats_found_a": sa is not None,
        "stats_found_b": sb is not None,
    }


def train_model(
    csv_path="data/matches.csv",
    team_stats_csv="data/team_stats.csv",
    decay_lambda=0.003,
    epochs=200,
    cutoff_date=None,
):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date", ascending=True).reset_index(drop=True)

    if cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date)
        df = df[df["date"] < cutoff_ts].copy().reset_index(drop=True)
        if len(df) < 10:
            raise ValueError(f"Prea puține meciuri ({len(df)}) înainte de data {cutoff_date}!")

    base_features = prepare_features(df)

    swapped_features = pd.DataFrame(index=base_features.index)
    swapped_features["rank_diff"] = -base_features["rank_diff"]
    swapped_features["log_rank_ratio"] = -base_features["log_rank_ratio"]
    swapped_features["wr_diff"] = -base_features["wr_diff"]
    swapped_features["is_bo1"] = base_features["is_bo1"]
    swapped_features["is_major"] = base_features["is_major"]

    max_date = df["date"].max()
    days_ago = (max_date - df["date"]).dt.days
    weights = np.exp(-decay_lambda * days_ago).values

    n_total = len(df)
    split_idx = max(10, int(n_total * 0.85))

    X_train_df = pd.concat([base_features.iloc[:split_idx], swapped_features.iloc[:split_idx]], ignore_index=True)
    y_train = np.concatenate([df["winner"].values[:split_idx], 1 - df["winner"].values[:split_idx]]).astype(np.float32)
    w_train = np.concatenate([weights[:split_idx], weights[:split_idx]]).astype(np.float32)

    X_cal_df = pd.concat([base_features.iloc[split_idx:], swapped_features.iloc[split_idx:]], ignore_index=True)
    y_cal = np.concatenate([df["winner"].values[split_idx:], 1 - df["winner"].values[split_idx:]]).astype(np.float32)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df[FEATURE_COLS].values).astype(np.float32)

    dataset = TensorDataset(
        torch.tensor(X_train_scaled),
        torch.tensor(y_train).unsqueeze(1),
        torch.tensor(w_train).unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    torch.manual_seed(42)
    model = CS2MatchNet(input_dim=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    model.train()
    for _ in range(epochs):
        for bx, by, bw in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = (criterion(out, by) * bw).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    if len(X_cal_df) >= 10:
        X_cal_scaled = scaler.transform(X_cal_df[FEATURE_COLS].values).astype(np.float32)
        with torch.no_grad():
            cal_logits = model(torch.tensor(X_cal_scaled)).numpy().flatten()
        learned_temp = _fit_temperature(cal_logits, y_cal)
    else:
        learned_temp = 0.85

    X_all_df = pd.concat([base_features, swapped_features], ignore_index=True)
    y_all = np.concatenate([df["winner"].values, 1 - df["winner"].values]).astype(np.float32)
    w_all = np.concatenate([weights, weights]).astype(np.float32)

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_df[FEATURE_COLS].values).astype(np.float32)
    full_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_all_scaled),
            torch.tensor(y_all).unsqueeze(1),
            torch.tensor(w_all).unsqueeze(1),
        ),
        batch_size=32,
        shuffle=True,
    )

    torch.manual_seed(42)
    model = CS2MatchNet(input_dim=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        for bx, by, bw in full_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = (criterion(out, by) * bw).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    model.learned_temp = learned_temp
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(X_all_scaled)) * learned_temp).numpy().flatten()
        acc = ((preds >= 0.5) == y_all).mean()

    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "input_dim": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "temperature": learned_temp,
        "cutoff_date": str(cutoff_date) if cutoff_date is not None else "Full Dataset",
        "n_matches": int(len(df)),
        "max_train_date": str(max_date.date()),
    }, "models/cs2_net.pt")

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return float(acc), int(len(df))


def evaluate_holdout(test_csv="data/test_hong_kong.csv", cutoff_date=None, team_stats_csv="data/team_stats.csv"):
    if not os.path.exists(test_csv) or not os.path.exists("models/cs2_net.pt"):
        return None

    df_test = pd.read_csv(test_csv)
    if cutoff_date is not None and "date" in df_test.columns:
        df_test["date_dt"] = pd.to_datetime(df_test["date"])
        df_test = df_test[df_test["date_dt"] >= pd.to_datetime(cutoff_date)].copy()
        if len(df_test) == 0:
            return None

    model, scaler = load_predictor()
    if model is None:
        return None

    stats_db = load_team_stats(csv_path=team_stats_csv, as_of_date=cutoff_date)
    hist_df = pd.read_csv("data/matches.csv") if os.path.exists("data/matches.csv") else None
    if hist_df is not None and cutoff_date is not None and "date" in hist_df.columns:
        hist_df = hist_df[pd.to_datetime(hist_df["date"]) < pd.to_datetime(cutoff_date)].copy()

    probs_list = []
    for _, r in df_test.iterrows():
        out = predict_match_with_team_stats(
            model=model,
            scaler=scaler,
            team_a=str(r.get("team_a", "")),
            team_b=str(r.get("team_b", "")),
            rank_a=int(r.get("rank_a", 20)),
            rank_b=int(r.get("rank_b", 20)),
            fallback_wr_a=float(r.get("winrate_3m_a", 0.5)),
            fallback_wr_b=float(r.get("winrate_3m_b", 0.5)),
            match_format=str(r.get("format", "BO3")),
            event_tier=str(r.get("event_tier", "Major")),
            as_of_date=cutoff_date,
            team_stats_csv=team_stats_csv,
        )
        probs_list.append(float(out["prob_a"]))

    probs_a = np.array(probs_list, dtype=float)
    df_res = df_test.copy()
    df_res["prob_a"] = probs_a
    df_res["predicted_winner"] = np.where(probs_a >= 0.5, df_res["team_a"], df_res["team_b"])
    df_res["actual_winner"] = np.where(df_res["winner"] == 1, df_res["team_a"], df_res["team_b"])
    df_res["correct"] = (probs_a >= 0.5).astype(int) == df_res["winner"]

    return {
        "accuracy": float(df_res["correct"].mean()),
        "brier_score": float(np.mean((probs_a - df_res["winner"]) ** 2)),
        "total_matches": len(df_res),
        "correct_count": int(df_res["correct"].sum()),
        "details": df_res,
    }