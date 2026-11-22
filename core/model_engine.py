import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

from core.team_stats_engine import load_team_stats, compute_map_veto_edge, assess_match_risk


class CS2MatchNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()
    features['rank_diff'] = df['rank_b'] - df['rank_a']
    features['log_rank_ratio'] = np.log1p(df['rank_b']) - np.log1p(df['rank_a'])
    features['wr_diff'] = df['winrate_3m_a'] - df['winrate_3m_b']
    features['wr_a'] = df['winrate_3m_a']
    features['wr_b'] = df['winrate_3m_b']
    features['is_bo1'] = (df['format'] == 'BO1').astype(float)
    features['is_major'] = (df['event_tier'] == 'Major').astype(float)
    return features


def load_predictor(model_path="models/cs2_net.pt", scaler_path="models/scaler.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    checkpoint = torch.load(model_path, weights_only=True)
    model = CS2MatchNet(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def get_model_metadata(model_path="models/cs2_net.pt"):
    if not os.path.exists(model_path):
        return None
    checkpoint = torch.load(model_path, weights_only=True)
    return {
        'cutoff_date': checkpoint.get('cutoff_date', 'Full Dataset'),
        'n_matches': checkpoint.get('n_matches', '?'),
        'max_train_date': checkpoint.get('max_train_date', '?')
    }


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
):
    base_fmt = "BO1" if match_format == "BO1" else "BO3"
    sample = pd.DataFrame([{
        'rank_a': rank_a,
        'rank_b': rank_b,
        'winrate_3m_a': float(np.clip(wr_a + sos_adj_a, 0.05, 0.95)),
        'winrate_3m_b': float(np.clip(wr_b + sos_adj_b, 0.05, 0.95)),
        'format': base_fmt,
        'event_tier': event_tier
    }])
    X_scaled = scaler.transform(prepare_features(sample).values).astype(np.float32)
    with torch.no_grad():
        logit = model(torch.tensor(X_scaled)).item()

    logit_adj = logit + (map_edge_a * 1.4) + (h2h_edge_a * 0.9)
    prob_a = float(1.0 / (1.0 + np.exp(-logit_adj)))

    if match_format == "BO5":
        roots = np.roots([-2.0, 3.0, 0.0, -prob_a])
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-5 and 0.0 <= r.real <= 1.0]
        p_map = real_roots[0] if real_roots else prob_a
        q = 1.0 - p_map
        prob_a = float((p_map ** 3) * (1.0 + 3.0 * q + 6.0 * (q ** 2)))
    else:
        p_map = prob_a

    prob_a = float(np.clip(prob_a, 0.01, 0.99))
    prob_b = 1.0 - prob_a

    if match_format == "BO1":
        if prob_a >= 0.5:
            loser_rounds = int(np.clip(round(13 * (prob_b / prob_a) ** 0.75), 3, 11))
            score_str = f"13 - {loser_rounds}"
        else:
            loser_rounds = int(np.clip(round(13 * (prob_a / prob_b) ** 0.75), 3, 11))
            score_str = f"{loser_rounds} - 13"
    elif match_format == "BO3":
        if prob_a >= 0.68:
            score_str = "2 - 0"
        elif prob_a >= 0.50:
            score_str = "2 - 1"
        elif prob_a >= 0.32:
            score_str = "1 - 2"
        else:
            score_str = "0 - 2"
    else:
        if prob_a >= 0.78:
            score_str = "3 - 0"
        elif prob_a >= 0.62:
            score_str = "3 - 1"
        elif prob_a >= 0.50:
            score_str = "3 - 2"
        elif prob_a >= 0.38:
            score_str = "2 - 3"
        elif prob_a >= 0.22:
            score_str = "1 - 3"
        else:
            score_str = "0 - 3"

    return {
        'prob_a': prob_a,
        'prob_b': prob_b,
        'p_map': float(np.clip(p_map, 0.01, 0.99)),
        'predicted_score': score_str
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
    manual_map_edge: float = 0.0,
    h2h_edge_a: float = 0.0,
    team_stats_csv: str = "data/team_stats.csv",
):
    """
    Combină predicția neurală de bază cu statisticile recente din team_stats.csv
    (filtrate cronologic până la as_of_date pentru a evita data leakage) și calculează
    riscul de surpriză (upset_risk) + volatilitatea meciului.
    """
    stats_db = load_team_stats(csv_path=team_stats_csv, as_of_date=as_of_date)
    sa = stats_db.get(team_a.strip().lower())
    sb = stats_db.get(team_b.strip().lower())

    wr_a = sa["top20_wr"] if (sa and sa.get("top20_maps", 0) >= 5) else (sa["map_wr"] if sa else fallback_wr_a)
    wr_b = sb["top20_wr"] if (sb and sb.get("top20_maps", 0) >= 5) else (sb["map_wr"] if sb else fallback_wr_b)

    sos_delta_a = float(np.clip(sa["sos_delta"] * 0.35, -0.06, 0.06)) if sa else 0.0
    sos_delta_b = float(np.clip(sb["sos_delta"] * 0.35, -0.06, 0.06)) if sb else 0.0

    eco_a = sa.get("eco_kd") if sa else None
    eco_b = sb.get("eco_kd") if sb else None
    eco_pm_a = sa.get("eco_per_map") if sa else None
    eco_pm_b = sb.get("eco_per_map") if sb else None

    eco_edge_logit = 0.0
    if eco_a is not None and eco_b is not None:
        eco_edge_logit = float(np.clip((eco_a - eco_b) * 0.45, -0.20, 0.20))

    auto_map_edge = compute_map_veto_edge(sa, sb, selected_maps=selected_maps)
    total_map_edge = float(np.clip(manual_map_edge + auto_map_edge, -0.25, 0.25))

    real_wr_edge = float(np.clip((wr_a - wr_b) * 2.5, -0.65, 0.65)) if (sa and sb) else 0.0

    base_res = predict_match_context(
        model=model,
        scaler=scaler,
        rank_a=rank_a,
        rank_b=rank_b,
        wr_a=wr_a,
        wr_b=wr_b,
        match_format=match_format,
        event_tier=event_tier,
        sos_adj_a=sos_adj_a + sos_delta_a,
        sos_adj_b=sos_adj_b + sos_delta_b,
        map_edge_a=float(np.clip(total_map_edge + real_wr_edge * 0.35, -0.25, 0.25)),
        h2h_edge_a=float(np.clip(h2h_edge_a + (eco_edge_logit * 0.5) + (real_wr_edge * 0.15), -0.20, 0.20)),
    )

    rel_a = sa.get("reliability", 0.75) if sa else 0.75
    rel_b = sb.get("reliability", 0.75) if sb else 0.75

    risk = assess_match_risk(
        prob_a=base_res["prob_a"],
        match_format=match_format,
        p_map=base_res["p_map"],
        eco_kd_a=eco_a,
        eco_kd_b=eco_b,
        eco_per_map_a=eco_pm_a,
        eco_per_map_b=eco_pm_b,
        reliability_a=rel_a,
        reliability_b=rel_b,
    )

    return {
        **base_res,
        **risk,
        "used_wr_a": round(wr_a, 4),
        "used_wr_b": round(wr_b, 4),
        "auto_map_edge": round(auto_map_edge, 4),
        "total_map_edge": round(total_map_edge, 4),
        "stats_found_a": sa is not None,
        "stats_found_b": sb is not None,
    }

def train_model(csv_path="data/matches.csv", decay_lambda=0.003, epochs=250, cutoff_date=None):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])

    if cutoff_date is not None:
        cutoff_ts = pd.to_datetime(cutoff_date)
        df = df[df['date'] < cutoff_ts].copy()
        if len(df) < 10:
            raise ValueError(f"Prea puține meciuri ({len(df)}) înainte de data {cutoff_date}!")

    max_date = df['date'].max()
    days_ago = (max_date - df['date']).dt.days
    weights = np.exp(-decay_lambda * days_ago).values

    df_swapped = df.copy()
    df_swapped['team_a'], df_swapped['team_b'] = df['team_b'], df['team_a']
    df_swapped['rank_a'], df_swapped['rank_b'] = df['rank_b'], df['rank_a']
    df_swapped['winrate_3m_a'], df_swapped['winrate_3m_b'] = df['winrate_3m_b'], df['winrate_3m_a']
    df_swapped['winner'] = 1 - df['winner']

    df_full = pd.concat([df, df_swapped], ignore_index=True)
    weights_full = np.concatenate([weights, weights])

    X_df = prepare_features(df_full)
    y = df_full['winner'].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values).astype(np.float32)

    X_tensor = torch.tensor(X_scaled)
    y_tensor = torch.tensor(y).unsqueeze(1)
    w_tensor = torch.tensor(weights_full.astype(np.float32)).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor, w_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    torch.manual_seed(42)
    model = CS2MatchNet(input_dim=X_scaled.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    model.train()
    for _ in range(epochs):
        for bx, by, bw in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = (criterion(out, by) * bw).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(X_tensor)).numpy().flatten()
        acc = ((preds >= 0.5) == y).mean()

    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'input_dim': X_scaled.shape[1],
        'feature_cols': list(X_df.columns),
        'cutoff_date': str(cutoff_date) if cutoff_date is not None else "Full Dataset",
        'n_matches': int(len(df)),
        'max_train_date': str(max_date.date())
    }, "models/cs2_net.pt")

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return acc, len(df)


def evaluate_holdout(test_csv="data/test_hong_kong.csv"):
    if not os.path.exists(test_csv) or not os.path.exists("models/cs2_net.pt"):
        return None

    df_test = pd.read_csv(test_csv)
    model, scaler = load_predictor()
    if model is None:
        return None

    X_test = prepare_features(df_test)
    X_scaled = scaler.transform(X_test.values).astype(np.float32)

    with torch.no_grad():
        probs_a = torch.sigmoid(model(torch.tensor(X_scaled))).numpy().flatten()

    df_res = df_test.copy()
    df_res['prob_a'] = probs_a
    df_res['predicted_winner'] = np.where(probs_a >= 0.5, df_res['team_a'], df_res['team_b'])
    df_res['actual_winner'] = np.where(df_res['winner'] == 1, df_res['team_a'], df_res['team_b'])
    df_res['correct'] = (probs_a >= 0.5).astype(int) == df_res['winner']

    return {
        'accuracy': float(df_res['correct'].mean()),
        'brier_score': float(np.mean((probs_a - df_res['winner']) ** 2)),
        'total_matches': len(df_res),
        'correct_count': int(df_res['correct'].sum()),
        'details': df_res
    }