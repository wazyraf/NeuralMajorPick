import json
import os
import numpy as np
import pandas as pd


DEFAULT_TEAM_STATS_PATH = "data/team_stats.csv"


def load_team_stats(
    csv_path: str = DEFAULT_TEAM_STATS_PATH,
    as_of_date: str | None = None,
) -> dict[str, dict]:
    """
    Încarcă statisticile per echipă din data/team_stats.csv.
    Dacă `as_of_date` este setat (ex. '2026-06-02' la Blind Backtest),
    păstrează doar snapshot-urile cu snapshot_date < as_of_date
    și reține cel mai recent snapshot disponibil pentru fiecare echipă.
    """
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path)
    if df.empty or "team" not in df.columns:
        return {}

    df["team_clean"] = df["team"].astype(str).str.strip()
    df = df[df["team_clean"] != ""].copy()
    if df.empty:
        return {}

    if "snapshot_date" in df.columns:
        df["snap_dt"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
        if as_of_date is not None and str(as_of_date).strip() != "":
            cutoff_ts = pd.to_datetime(as_of_date)
            df = df[df["snap_dt"].isna() | (df["snap_dt"] < cutoff_ts)].copy()
        df = df.sort_values(["snap_dt"], ascending=True, na_position="first")

    if df.empty:
        return {}

    stats_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        team = row["team_clean"]

        maps_played = float(pd.to_numeric(row.get("maps_played", 0), errors="coerce") or 0.0)
        map_wins = float(pd.to_numeric(row.get("map_wins", 0), errors="coerce") or 0.0)
        map_draws = float(pd.to_numeric(row.get("map_draws", 0), errors="coerce") or 0.0)
        map_losses = float(pd.to_numeric(row.get("map_losses", 0), errors="coerce") or 0.0)

        denom_maps = maps_played if maps_played > 0 else (map_wins + map_draws + map_losses)
        map_wr = (map_wins / denom_maps) if denom_maps > 0 else 0.5

        top20_maps = float(pd.to_numeric(row.get("top20_maps", 0), errors="coerce") or 0.0)
        top20_wins = float(pd.to_numeric(row.get("top20_wins", 0), errors="coerce") or 0.0)
        top20_wr_raw = pd.to_numeric(row.get("top20_wr", np.nan), errors="coerce")

        if pd.notna(top20_wr_raw) and float(top20_wr_raw) > 0:
            val_top20 = float(top20_wr_raw)
            top20_wr = val_top20 / 100.0 if val_top20 > 1.0 else val_top20
        elif top20_maps > 0:
            top20_wr = top20_wins / top20_maps
        else:
            top20_wr = map_wr

        eco_kd = float(pd.to_numeric(row.get("eco_kd", 0.35), errors="coerce") or 0.35)
        eco_rounds = int(float(pd.to_numeric(row.get("eco_rounds", 0), errors="coerce") or 0))
        kd_ratio = float(pd.to_numeric(row.get("kd_ratio", 1.0), errors="coerce") or 1.0)
        top20_kd = float(pd.to_numeric(row.get("top20_kd", kd_ratio), errors="coerce") or kd_ratio)
        rounds_played = int(float(pd.to_numeric(row.get("rounds_played", 0), errors="coerce") or 0))

        # 1) Drop-ul / ponderea hărților jucate contra Top 20 față de totalul hărților (0.0 .. 1.0)
        top20_share = (top20_maps / denom_maps) if (denom_maps > 0 and top20_maps > 0) else 0.5
        # 2) Densitatea rundelor de eco per hartă jucată
        eco_per_map = (eco_rounds / denom_maps) if denom_maps > 0 else 1.0

        # Fiabilitatea eșantionului (0.35 .. 1.0): combină volumul total de hărți
        # cu acoperirea contra adversarilor de top (top20_share)
        vol_score = min(denom_maps / 80.0, 1.0)
        sos_coverage = min(top20_share / 0.70, 1.0)
        reliability = float(np.clip(0.35 + 0.40 * vol_score + 0.25 * sos_coverage, 0.35, 1.0))

        map_pool: dict[str, float] = {}
        raw_pool = row.get("map_pool_json", "")
        if pd.notna(raw_pool) and str(raw_pool).strip():
            try:
                parsed = json.loads(str(raw_pool))
                if isinstance(parsed, dict):
                    for map_name, wr_val in parsed.items():
                        wr_num = float(wr_val)
                        map_pool[str(map_name).strip()] = wr_num / 100.0 if wr_num > 1.0 else wr_num
            except json.JSONDecodeError:
                map_pool = {}

        stats_map[team.lower()] = {
            "team": team,
            "snapshot_date": str(row.get("snapshot_date", "")),
            "time_window": str(row.get("time_window", "")),
            "maps_played": int(denom_maps),
            "map_wins": int(map_wins),
            "map_draws": int(map_draws),
            "map_losses": int(map_losses),
            "rounds_played": rounds_played,
            "map_wr": round(float(np.clip(map_wr, 0.0, 1.0)), 4),
            "top20_maps": int(top20_maps),
            "top20_share": round(float(np.clip(top20_share, 0.0, 1.0)), 3),
            "top20_wr": round(float(np.clip(top20_wr, 0.0, 1.0)), 4),
            "sos_delta": round(float(np.clip(top20_wr - map_wr, -0.5, 0.5)), 4),
            "kd_ratio": round(kd_ratio, 3),
            "top20_kd": round(top20_kd, 3),
            "eco_rounds": eco_rounds,
            "eco_per_map": round(float(np.clip(eco_per_map, 0.0, 5.0)), 3),
            "eco_kd": round(eco_kd, 3),
            "reliability": round(reliability, 3),
            "map_pool": map_pool,
        }

    return stats_map


def compute_map_veto_edge(
    stats_a: dict | None,
    stats_b: dict | None,
    selected_maps: list[str] | None = None,
) -> float:
    """
    Calculează avantajul estimat pe hartă / veto (-0.20 .. +0.20 din perspectiva Team A)
    folosind winrate-urile pe hărțile comune / selectate din `map_pool`.
    """
    if not stats_a or not stats_b:
        return 0.0

    pool_a = stats_a.get("map_pool") or {}
    pool_b = stats_b.get("map_pool") or {}
    if not pool_a or not pool_b:
        return 0.0

    common_maps = [m for m in pool_a.keys() if m in pool_b]
    if selected_maps:
        selected_set = {m.strip().lower() for m in selected_maps if str(m).strip()}
        filtered = [m for m in common_maps if m.lower() in selected_set]
        if filtered:
            common_maps = filtered

    if not common_maps:
        return 0.0

    diffs = [pool_a[m] - pool_b[m] for m in common_maps]
    return float(np.clip(np.mean(diffs) * 0.5, -0.20, 0.20))


def assess_match_risk(
    prob_a: float,
    match_format: str = "BO3",
    p_map: float | None = None,
    eco_kd_a: float | None = None,
    eco_kd_b: float | None = None,
    eco_per_map_a: float | None = None,
    eco_per_map_b: float | None = None,
    reliability_a: float = 1.0,
    reliability_b: float = 1.0,
) -> dict:
    """
    Estimează riscul de surpriză (`upset_risk_pct`), volatilitatea meciului (`volatility_score`)
    și nivelul de încredere (`trust_label`, `badge`).
    """
    p_a_clamped = float(np.clip(prob_a, 0.01, 0.99))
    fav_prob = max(p_a_clamped, 1.0 - p_a_clamped)
    base_upset = (1.0 - fav_prob) * 100.0

    # Format BO1 are varianță mai mare decât BO3 / BO5
    fmt_mult = {"BO1": 1.15, "BO3": 1.0, "BO5": 0.90}.get(match_format, 1.0)

    # Apropierea de 50/50 pe hartă (meci strâns, ex. 13-11 / 2-1) crește volatilitatea
    effective_p_map = float(np.clip(p_map if p_map is not None else fav_prob, 0.01, 0.99))
    map_closeness = 1.0 - min(abs(effective_p_map - 0.5) * 2.0, 1.0)

    # Impactul rundelor de eco (eco_kd + densitatea eco_per_map) asupra volatilității
    eco_vol = 0.0
    if eco_kd_a is not None and eco_kd_b is not None:
        eco_edge = float(eco_kd_a) - float(eco_kd_b)
        if (p_a_clamped >= 0.5 and eco_edge < -0.10) or (p_a_clamped < 0.5 and eco_edge > 0.10):
            eco_vol += 4.0
        if min(float(eco_kd_a), float(eco_kd_b)) < 0.30:
            eco_vol += 2.5

    if eco_per_map_a is not None and eco_per_map_b is not None:
        avg_eco_per_map = (float(eco_per_map_a) + float(eco_per_map_b)) / 2.0
        if avg_eco_per_map > 1.10:
            eco_vol += min((avg_eco_per_map - 1.10) * 6.0, 3.5)

    avg_rel = (float(reliability_a) + float(reliability_b)) / 2.0
    sample_penalty = (1.0 - np.clip(avg_rel, 0.1, 1.0)) * 10.0

    upset_risk = float(np.clip(base_upset * fmt_mult + eco_vol + sample_penalty, 1.0, 49.5))
    volatility = float(np.clip(map_closeness * 65.0 + eco_vol * 4.0 + sample_penalty * 2.0, 5.0, 100.0))

    if upset_risk < 25.0 and volatility < 45.0:
        trust_label = "LOW RISK (HIGH TRUST)"
        badge = "🟢 SAFE"
    elif upset_risk <= 38.0 and volatility < 70.0:
        trust_label = "MODERATE RISK"
        badge = "🟡 MODERATE"
    else:
        trust_label = "HIGH UPSET RISK"
        badge = "🔴 VOLATILE"

    return {
        "upset_risk_pct": round(upset_risk, 1),
        "volatility_score": round(volatility, 1),
        "trust_label": trust_label,
        "badge": badge,
    }