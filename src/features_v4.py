import os
import numpy as np
import pandas as pd

FEATURE_COLS_V4 = [
    "diff_vrs_points_norm",
    "diff_wr_3m",
    "diff_shrunk_map_wr",
    "diff_shrunk_top20_wr",
    "diff_round_winrate",
    "diff_team_rating",
    "is_bo1",
    "is_major",
]


def _shrink_rate(rate: float, n_maps: float, prior: float = 0.50, k: float = 15.0) -> float:
    """Empirical Bayes shrinkage spre un prior neutru în funcție de volumul de hărți."""
    if np.isnan(rate):
        return prior
    n = max(0.0, float(n_maps))
    return float((n * rate + k * prior) / (n + k))


def load_vrs_stats_map(stats_csv: str | None, before_date: pd.Timestamp) -> dict[str, dict]:
    if not stats_csv or not os.path.exists(stats_csv):
        return {}

    df = pd.read_csv(stats_csv)
    if df.empty or "team" not in df.columns:
        return {}

    df["team_key"] = df["team"].astype(str).str.strip().str.lower()
    if "snapshot_date" in df.columns:
        df["snap_dt"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        df = df[df["snap_dt"].notna() & (df["snap_dt"] < before_date)].copy()
        df = df.sort_values("snap_dt", ascending=True)

    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = row["team_key"]
        if not key:
            continue

        vrs_pts = pd.to_numeric(row.get("vrs_points", np.nan), errors="coerce")
        vrs_rnk = pd.to_numeric(row.get("vrs_rank", np.nan), errors="coerce")
        maps_p = pd.to_numeric(row.get("maps_played", 0.0), errors="coerce")
        map_wr = pd.to_numeric(row.get("map_winrate", np.nan), errors="coerce")
        t20_maps = pd.to_numeric(row.get("top20_maps", 0.0), errors="coerce")
        t20_wr = pd.to_numeric(row.get("top20_winrate", np.nan), errors="coerce")
        r_wr = pd.to_numeric(row.get("round_winrate", np.nan), errors="coerce")
        t_rat = pd.to_numeric(row.get("team_rating", np.nan), errors="coerce")

        out[key] = {
            "vrs_points": float(vrs_pts) if pd.notna(vrs_pts) else np.nan,
            "vrs_rank": float(vrs_rnk) if pd.notna(vrs_rnk) else np.nan,
            "maps_played": float(maps_p) if pd.notna(maps_p) else 0.0,
            "map_winrate": float(map_wr) if pd.notna(map_wr) else np.nan,
            "top20_maps": float(t20_maps) if pd.notna(t20_maps) else 0.0,
            "top20_winrate": float(t20_wr) if pd.notna(t20_wr) else np.nan,
            "round_winrate": float(r_wr) if pd.notna(r_wr) else np.nan,
            "team_rating": float(t_rat) if pd.notna(t_rat) else np.nan,
        }
    return out


def build_features_v4(
    df: pd.DataFrame,
    stats_csv: str | None = "data/vrs_team_stats_cologne.csv",
    fixed_as_of: str | None = None,
) -> pd.DataFrame:
    rows = []
    has_date = "date" in df.columns
    dates = pd.to_datetime(df["date"], errors="coerce") if has_date else pd.Series([pd.NaT] * len(df))
    cache: dict[str, dict[str, dict]] = {}

    for idx, r in df.iterrows():
        if fixed_as_of:
            cutoff_ts = pd.to_datetime(fixed_as_of)
            c_key = str(cutoff_ts.date())
        elif pd.notna(dates.iloc[idx]):
            cutoff_ts = dates.iloc[idx]
            c_key = str(cutoff_ts.date())
        else:
            cutoff_ts = pd.Timestamp.max
            c_key = "MAX"

        if c_key not in cache:
            cache[c_key] = load_vrs_stats_map(stats_csv, cutoff_ts)
        s_map = cache[c_key]

        ta = str(r.get("team_a", "")).strip().lower()
        tb = str(r.get("team_b", "")).strip().lower()
        sa = s_map.get(ta, {})
        sb = s_map.get(tb, {})

        ra = max(1.0, float(sa.get("vrs_rank") if pd.notna(sa.get("vrs_rank")) else r.get("rank_a", 50)))
        rb = max(1.0, float(sb.get("vrs_rank") if pd.notna(sb.get("vrs_rank")) else r.get("rank_b", 50)))

        pts_a = sa.get("vrs_points", np.nan)
        pts_b = sb.get("vrs_points", np.nan)
        if pd.notna(pts_a) and pd.notna(pts_b):
            diff_vrs_pts = float((float(pts_a) - float(pts_b)) / 400.0)
        else:
            diff_vrs_pts = float((np.log1p(rb) - np.log1p(ra)) * 0.55)

        wr3_a = float(r.get("winrate_3m_a", 0.5))
        wr3_b = float(r.get("winrate_3m_b", 0.5))

        s_map_a = _shrink_rate(sa.get("map_winrate", wr3_a), sa.get("maps_played", 20.0), prior=0.50, k=15.0)
        s_map_b = _shrink_rate(sb.get("map_winrate", wr3_b), sb.get("maps_played", 20.0), prior=0.50, k=15.0)

        s_t20_a = _shrink_rate(sa.get("top20_winrate", s_map_a), sa.get("top20_maps", 0.0), prior=0.45, k=8.0)
        s_t20_b = _shrink_rate(sb.get("top20_winrate", s_map_b), sb.get("top20_maps", 0.0), prior=0.45, k=8.0)

        rwr_a = float(sa.get("round_winrate")) if pd.notna(sa.get("round_winrate")) else 0.50
        rwr_b = float(sb.get("round_winrate")) if pd.notna(sb.get("round_winrate")) else 0.50

        rat_a = float(sa.get("team_rating")) if pd.notna(sa.get("team_rating")) else 1.00
        rat_b = float(sb.get("team_rating")) if pd.notna(sb.get("team_rating")) else 1.00

        fmt = str(r.get("format", "BO3")).strip().upper()
        tier = str(r.get("event_tier", "")).strip().lower()

        rows.append({
            "diff_vrs_points_norm": diff_vrs_pts,
            "diff_wr_3m": float(wr3_a - wr3_b),
            "diff_shrunk_map_wr": float(s_map_a - s_map_b),
            "diff_shrunk_top20_wr": float(s_t20_a - s_t20_b),
            "diff_round_winrate": float(rwr_a - rwr_b),
            "diff_team_rating": float(rat_a - rat_b),
            "is_bo1": 1.0 if fmt == "BO1" else 0.0,
            "is_major": 1.0 if tier == "major" else 0.0,
        })

    return pd.DataFrame(rows, index=df.index)[FEATURE_COLS_V4]


def swap_features_v4(X: pd.DataFrame) -> pd.DataFrame:
    sw = X.copy()
    diff_cols = [
        "diff_vrs_points_norm",
        "diff_wr_3m",
        "diff_shrunk_map_wr",
        "diff_shrunk_top20_wr",
        "diff_round_winrate",
        "diff_team_rating",
    ]
    for c in diff_cols:
        sw[c] = -sw[c]
    return sw[FEATURE_COLS_V4]