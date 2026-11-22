import json
import os
import numpy as np
import pandas as pd


DEFAULT_TEAM_STATS_PATH = "data/team_stats.csv"
DEFAULT_MATCHES_PATH = "data/matches.csv"


def _parse_json_dict(raw_val, is_rate: bool = False) -> dict:
    if pd.isna(raw_val) or not str(raw_val).strip():
        return {}
    try:
        parsed = json.loads(str(raw_val))
        if not isinstance(parsed, dict):
            return {}
        out = {}
        for k, v in parsed.items():
            key = str(k).strip()
            num = float(v)
            if is_rate:
                out[key] = float(np.clip(num / 100.0 if num > 1.0 else num, 0.0, 1.0))
            else:
                out[key] = num
        return out
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


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

        recent_maps = int(float(pd.to_numeric(row.get("recent_maps", 0), errors="coerce") or 0))
        recent_wins = int(float(pd.to_numeric(row.get("recent_wins", 0), errors="coerce") or 0))
        recent_wr_raw = pd.to_numeric(row.get("recent_wr", np.nan), errors="coerce")
        if pd.notna(recent_wr_raw) and float(recent_wr_raw) >= 0:
            val_rec = float(recent_wr_raw)
            recent_wr = val_rec / 100.0 if val_rec > 1.0 else val_rec
        elif recent_maps > 0:
            recent_wr = recent_wins / recent_maps
        else:
            recent_wr = map_wr

        recent_avg_round_diff = float(
            pd.to_numeric(row.get("recent_avg_round_diff", 0.0), errors="coerce") or 0.0
        )

        combined_2026_wr = float(np.clip(0.55 * recent_wr + 0.45 * map_wr, 0.05, 0.95))

        big_maps = int(float(pd.to_numeric(row.get("big_maps", 0), errors="coerce") or 0))
        big_pistol_raw = pd.to_numeric(row.get("big_pistol_wr", np.nan), errors="coerce")
        big_r2_conv_raw = pd.to_numeric(row.get("big_r2_conv", np.nan), errors="coerce")
        big_r2_break_raw = pd.to_numeric(row.get("big_r2_break", np.nan), errors="coerce")

        big_pistol_wr = (
            float(np.clip(float(big_pistol_raw) / 100.0 if float(big_pistol_raw) > 1.0 else float(big_pistol_raw), 0.0, 1.0))
            if pd.notna(big_pistol_raw)
            else None
        )
        big_r2_conv = (
            float(np.clip(float(big_r2_conv_raw) / 100.0 if float(big_r2_conv_raw) > 1.0 else float(big_r2_conv_raw), 0.0, 1.0))
            if pd.notna(big_r2_conv_raw)
            else None
        )
        big_r2_break = (
            float(np.clip(float(big_r2_break_raw) / 100.0 if float(big_r2_break_raw) > 1.0 else float(big_r2_break_raw), 0.0, 1.0))
            if pd.notna(big_r2_break_raw)
            else None
        )

        w_big = float(np.clip(big_maps / 30.0, 0.0, 1.0)) if big_maps > 0 else 0.0
        eff_pistol_wr = (w_big * big_pistol_wr + (1.0 - w_big) * 0.50) if big_pistol_wr is not None else 0.50
        eff_r2_conv = (w_big * big_r2_conv + (1.0 - w_big) * 0.80) if big_r2_conv is not None else 0.80
        eff_r2_break = (w_big * big_r2_break + (1.0 - w_big) * 0.20) if big_r2_break is not None else 0.20

        top20_share = (top20_maps / denom_maps) if (denom_maps > 0 and top20_maps > 0) else 0.5
        eco_per_map = (eco_rounds / denom_maps) if denom_maps > 0 else 1.0

        vol_score = min(denom_maps / 80.0, 1.0)
        sos_coverage = min(top20_share / 0.70, 1.0)
        big_cov = min(big_maps / 25.0, 1.0) if big_maps > 0 else 0.5
        reliability = float(
            np.clip(0.32 + 0.34 * vol_score + 0.20 * sos_coverage + 0.14 * big_cov, 0.35, 1.0)
        )

        map_pool = _parse_json_dict(row.get("map_pool_json", ""), is_rate=True)
        recent_map_counts = _parse_json_dict(row.get("recent_map_counts_json", ""), is_rate=False)

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
            "combined_2026_wr": round(combined_2026_wr, 4),
            "top20_maps": int(top20_maps),
            "top20_share": round(float(np.clip(top20_share, 0.0, 1.0)), 3),
            "top20_wr": round(float(np.clip(top20_wr, 0.0, 1.0)), 4),
            "sos_delta": round(float(np.clip(top20_wr - map_wr, -0.5, 0.5)), 4),
            "kd_ratio": round(kd_ratio, 3),
            "top20_kd": round(top20_kd, 3),
            "eco_rounds": eco_rounds,
            "eco_per_map": round(float(np.clip(eco_per_map, 0.0, 5.0)), 3),
            "eco_kd": round(eco_kd, 3),
            "recent_maps": recent_maps,
            "recent_wins": recent_wins,
            "recent_wr": round(float(np.clip(recent_wr, 0.0, 1.0)), 4),
            "recent_avg_round_diff": round(recent_avg_round_diff, 3),
            "big_maps": big_maps,
            "big_pistol_wr": round(big_pistol_wr, 4) if big_pistol_wr is not None else None,
            "big_r2_conv": round(big_r2_conv, 4) if big_r2_conv is not None else None,
            "big_r2_break": round(big_r2_break, 4) if big_r2_break is not None else None,
            "effective_pistol_wr": round(eff_pistol_wr, 4),
            "effective_r2_conv": round(eff_r2_conv, 4),
            "effective_r2_break": round(eff_r2_break, 4),
            "reliability": round(reliability, 3),
            "map_pool": map_pool,
            "recent_map_counts": recent_map_counts,
        }

    return stats_map


def get_latest_vrs_ranks(
    matches_csv: str = DEFAULT_MATCHES_PATH,
    as_of_date: str | None = None,
) -> dict[str, int]:
    """
    Extrage ultimul VRS rank cunoscut per echipă din matches.csv (strict < as_of_date)
    pentru pre-completarea VRS Rank-ului fără data leakage.
    """
    if not os.path.exists(matches_csv):
        return {}

    df = pd.read_csv(matches_csv)
    if df.empty or "team_a" not in df.columns or "team_b" not in df.columns:
        return {}

    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        if as_of_date is not None and str(as_of_date).strip() != "":
            df = df[df["date_dt"].isna() | (df["date_dt"] < pd.to_datetime(as_of_date))].copy()
        df = df.sort_values("date_dt", ascending=True, na_position="first")

    ranks: dict[str, int] = {}
    for _, r in df.iterrows():
        ta = str(r.get("team_a", "")).strip()
        tb = str(r.get("team_b", "")).strip()
        ra = pd.to_numeric(r.get("rank_a", np.nan), errors="coerce")
        rb = pd.to_numeric(r.get("rank_b", np.nan), errors="coerce")
        if ta and pd.notna(ra) and int(ra) > 0:
            ranks[ta.lower()] = int(ra)
        if tb and pd.notna(rb) and int(rb) > 0:
            ranks[tb.lower()] = int(rb)
    return ranks


def compute_h2h_history(
    team_a: str,
    team_b: str,
    matches_csv: str = DEFAULT_MATCHES_PATH,
    as_of_date: str | None = None,
    decay_lambda: float = 0.004,
) -> dict:
    """
    Calculează istoricul direct H2H între team_a și team_b din matches.csv
    (strict < as_of_date) cu time-decay și shrinkage spre 0.50.
    """
    ta_key = str(team_a).strip().lower()
    tb_key = str(team_b).strip().lower()
    if not ta_key or not tb_key or ta_key == tb_key or not os.path.exists(matches_csv):
        return {
            "h2h_games": 0,
            "h2h_wins_a": 0,
            "h2h_wr_a": 0.5,
            "h2h_edge": 0.0,
            "h2h_volatility_bump": 0.0,
        }

    df = pd.read_csv(matches_csv)
    if df.empty:
        return {
            "h2h_games": 0,
            "h2h_wins_a": 0,
            "h2h_wr_a": 0.5,
            "h2h_edge": 0.0,
            "h2h_volatility_bump": 0.0,
        }

    if "date" in df.columns:
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        if as_of_date is not None and str(as_of_date).strip() != "":
            df = df[df["date_dt"].isna() | (df["date_dt"] < pd.to_datetime(as_of_date))].copy()

    a_col = df["team_a"].astype(str).str.strip().str.lower()
    b_col = df["team_b"].astype(str).str.strip().str.lower()

    mask = ((a_col == ta_key) & (b_col == tb_key)) | ((a_col == tb_key) & (b_col == ta_key))
    sub = df[mask].copy()
    if sub.empty:
        return {
            "h2h_games": 0,
            "h2h_wins_a": 0,
            "h2h_wr_a": 0.5,
            "h2h_edge": 0.0,
            "h2h_volatility_bump": 0.0,
        }

    ref_date = sub["date_dt"].max() if "date_dt" in sub.columns and sub["date_dt"].notna().any() else pd.Timestamp.now()
    weights = []
    outcomes_a = []
    wins_a_count = 0

    for _, row in sub.iterrows():
        raw_winner = int(pd.to_numeric(row.get("winner", 1), errors="coerce") or 0)
        row_a = str(row.get("team_a", "")).strip().lower()
        a_won = (1.0 if raw_winner == 1 else 0.0) if row_a == ta_key else (1.0 if raw_winner == 0 else 0.0)

        if a_won >= 0.5:
            wins_a_count += 1

        days_ago = max(0.0, float((ref_date - row["date_dt"]).days)) if ("date_dt" in sub.columns and pd.notna(row["date_dt"])) else 30.0
        weights.append(float(np.exp(-decay_lambda * days_ago)))
        outcomes_a.append(a_won)

    w_arr = np.array(weights, dtype=float)
    o_arr = np.array(outcomes_a, dtype=float)
    n_games = len(sub)
    eff_n = float(np.sum(w_arr))
    weighted_wr = float(np.sum(w_arr * o_arr) / eff_n) if eff_n > 0 else 0.5

    shrink_w = eff_n / (eff_n + 2.5)
    shrunk_wr = float(shrink_w * weighted_wr + (1.0 - shrink_w) * 0.5)

    h2h_edge = float(np.clip((shrunk_wr - 0.5) * 0.36, -0.12, 0.12))

    balance_vol = 1.0 - min(1.0, abs(weighted_wr - 0.5) * 2.0)
    h2h_vol_bump = float(np.clip(balance_vol * 0.12 * min(n_games / 3.0, 1.0), 0.0, 0.14))

    return {
        "h2h_games": int(n_games),
        "h2h_wins_a": int(wins_a_count),
        "h2h_wr_a": round(shrunk_wr, 4),
        "h2h_edge": round(h2h_edge, 4),
        "h2h_volatility_bump": round(h2h_vol_bump, 4),
    }


def simulate_map_veto_edge(
    stats_a: dict | None,
    stats_b: dict | None,
    match_format: str = "BO3",
    selected_maps: list[str] | None = None,
) -> float:
    """
    Simulează un veto de hărți în spate între stats_a și stats_b (din perspectiva Team A)
    folosind `map_pool` și frecvența recentă din `recent_map_counts`.
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

    rc_a = stats_a.get("recent_map_counts") or {}
    rc_b = stats_b.get("recent_map_counts") or {}

    map_scores = []
    for m in common_maps:
        wa = float(pool_a[m])
        wb = float(pool_b[m])
        fa = float(rc_a.get(m, 1.0))
        fb = float(rc_b.get(m, 1.0))
        conf = max(0.35, min(1.0, ((fa + fb) / 2.0) / 4.0))
        diff = (wa - wb) * conf
        map_scores.append((m, diff, wa, wb))

    sorted_by_a = sorted(map_scores, key=lambda item: item[1], reverse=True)
    sorted_by_b = sorted(map_scores, key=lambda item: item[1], reverse=False)

    if match_format == "BO1":
        rem = [item[1] for item in map_scores]
        while len(rem) > 1:
            rem.pop(int(np.argmin(rem)))
            if len(rem) <= 1:
                break
            rem.pop(int(np.argmax(rem)))
        veto_val = rem[0] if rem else float(np.mean([x[1] for x in map_scores]))
    elif match_format == "BO5":
        top_a = sorted_by_a[0][1] if len(sorted_by_a) >= 1 else 0.0
        top_b = sorted_by_b[0][1] if len(sorted_by_b) >= 1 else 0.0
        mid = float(np.mean([x[1] for x in sorted_by_a]))
        veto_val = 0.45 * top_a + 0.45 * top_b + 0.10 * mid
    else:
        pick_a = sorted_by_a[0][1] if len(sorted_by_a) >= 1 else 0.0
        pick_b = sorted_by_b[0][1] if len(sorted_by_b) >= 1 else 0.0
        decider = sorted_by_a[len(sorted_by_a) // 2][1] if sorted_by_a else 0.0
        veto_val = 0.40 * pick_a + 0.40 * pick_b + 0.20 * decider

    return float(np.clip(veto_val * 0.45, -0.20, 0.20))


def compute_map_veto_edge(
    stats_a: dict | None,
    stats_b: dict | None,
    selected_maps: list[str] | None = None,
    match_format: str = "BO3",
) -> float:
    """Wrapper păstrat pentru compatibilitate, bazat pe simularea de veto."""
    return simulate_map_veto_edge(
        stats_a=stats_a,
        stats_b=stats_b,
        match_format=match_format,
        selected_maps=selected_maps,
    )


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
    r2_break_a: float | None = None,
    r2_break_b: float | None = None,
    h2h_volatility_bump: float = 0.0,
) -> dict:
    """
    Estimează riscul de surpriză (`upset_risk_pct`), volatilitatea meciului (`volatility_score`)
    și nivelul de încredere (`trust_label`, `badge`).
    """
    p_a_clamped = float(np.clip(prob_a, 0.01, 0.99))
    fav_prob = max(p_a_clamped, 1.0 - p_a_clamped)
    base_upset = (1.0 - fav_prob) * 100.0

    fmt_mult = {"BO1": 1.15, "BO3": 1.0, "BO5": 0.90}.get(match_format, 1.0)

    effective_p_map = float(np.clip(p_map if p_map is not None else fav_prob, 0.01, 0.99))
    map_closeness = 1.0 - min(abs(effective_p_map - 0.5) * 2.0, 1.0)

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

    r2_vol = 0.0
    brk_vals = [float(x) for x in (r2_break_a, r2_break_b) if x is not None and not np.isnan(float(x))]
    if brk_vals:
        max_brk = max(brk_vals)
        if max_brk > 0.25:
            r2_vol = min((max_brk - 0.25) * 22.0, 4.0)

    h2h_vol_pct = float(np.clip(h2h_volatility_bump * 45.0, 0.0, 7.0))

    avg_rel = (float(reliability_a) + float(reliability_b)) / 2.0
    sample_penalty = (1.0 - np.clip(avg_rel, 0.1, 1.0)) * 10.0

    upset_risk = float(np.clip(base_upset * fmt_mult + eco_vol + r2_vol + h2h_vol_pct + sample_penalty, 1.0, 49.5))
    volatility = float(
        np.clip(map_closeness * 65.0 + (eco_vol + r2_vol + h2h_vol_pct) * 4.0 + sample_penalty * 2.0, 5.0, 100.0)
    )

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