# Simulator oficial Valve Major (Runda 1: 1v9..8v16, Buchholz pe VRS seed, BO1/BO3 dinamic pe stage)
import numpy as np
import pandas as pd
import torch
from core.model_engine import prepare_features

REGION_SOS_DEFAULT = {
    "Europe (Tier 1/1.5)": 0.00,
    "Americas": -0.03,
    "Asia / Oceania": -0.06,
}


def simulate_swiss_stage(
    teams_df: pd.DataFrame,
    model,
    scaler,
    n_sims: int = 10000,
    stage_mode: str = "Stage 1 / Stage 2 (BO1 + Decisive BO3)",
    is_major: bool = True,
) -> pd.DataFrame:
    """
    Simulează o fază Swiss de 16 echipe conform regulamentului oficial Valve Major:
    - Runda 1: împerechere oficială Top 8 vs Bottom 8 (1 vs 9, 2 vs 10, ..., 8 vs 16).
    - Rundele 2-5: grupare pe scor (W-L), sortare după Buchholz Score (suma diferențelor W-L ale adversarilor),
      apoi VRS Seed (ordinea rândurilor 1..16) și împerechere high vs low fără rematch.
    - Format meciuri:
        * Stage 1 / Stage 2: BO1 în meciurile non-decisive (0-0, 1-0, 0-1, 1-1),
          BO3 în meciurile de calificare/eliminare (2-0, 0-2, 2-1, 1-2, 2-2).
        * Stage 3 / All-BO3: toate meciurile sunt BO3.
    """
    n_teams = len(teams_df)
    teams = teams_df['team'].tolist()
    ranks = teams_df['rank'].tolist()
    raw_wrs = teams_df['wr_3m'].tolist()

    # Aplicăm ajustarea Strength of Schedule dacă coloana 'region' este prezentă
    if 'region' in teams_df.columns:
        adj_wrs = [
            float(np.clip(w + REGION_SOS_DEFAULT.get(reg, 0.0), 0.05, 0.95))
            for w, reg in zip(raw_wrs, teams_df['region'].tolist())
        ]
    else:
        adj_wrs = [float(w) for w in raw_wrs]

    all_bo3 = "Stage 3" in stage_mode or "All BO3" in stage_mode
    event_tier = "Major" if is_major else "S-Tier LAN"

    # Pre-calculăm matricele de probabilități BO1 și BO3
    prob_bo1 = np.zeros((n_teams, n_teams))
    prob_bo3 = np.zeros((n_teams, n_teams))

    pairs = []
    for i in range(n_teams):
        for j in range(n_teams):
            if i != j:
                pairs.append((i, j, 'BO1'))
                pairs.append((i, j, 'BO3'))

    batch_df = pd.DataFrame([
        {
            'rank_a': ranks[i],
            'rank_b': ranks[j],
            'winrate_3m_a': adj_wrs[i],
            'winrate_3m_b': adj_wrs[j],
            'format': fmt,
            'event_tier': event_tier
        }
        for i, j, fmt in pairs
    ])

    X_scaled = scaler.transform(prepare_features(batch_df).values).astype(np.float32)
    with torch.no_grad():
        preds = torch.sigmoid(model(torch.tensor(X_scaled))).numpy().flatten()

    for idx, (i, j, fmt) in enumerate(pairs):
        if fmt == 'BO1':
            prob_bo1[i, j] = preds[idx]
        else:
            prob_bo3[i, j] = preds[idx]

    counts_3_0 = np.zeros(n_teams)
    counts_qual = np.zeros(n_teams)
    counts_0_3 = np.zeros(n_teams)

    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        wins = np.zeros(n_teams, dtype=int)
        losses = np.zeros(n_teams, dtype=int)
        opponents = [[] for _ in range(n_teams)]
        played = set()

        # --- RUNDA 1: Împerechere oficială Valve Major 1 vs 9, 2 vs 10, ..., 8 vs 16 ---
        half = n_teams // 2
        p_mat_r1 = prob_bo3 if all_bo3 else prob_bo1
        for i in range(half):
            t1, t2 = i, i + half
            played.add((t1, t2))
            played.add((t2, t1))
            opponents[t1].append(t2)
            opponents[t2].append(t1)
            if rng.random() < p_mat_r1[t1, t2]:
                wins[t1] += 1
                losses[t2] += 1
            else:
                wins[t2] += 1
                losses[t1] += 1

        # --- RUNDELE 2 - 5: Buchholz Tie-Breaker + VRS Seed ---
        for _round_num in range(2, 6):
            active = [i for i in range(n_teams) if wins[i] < 3 and losses[i] < 3]
            if not active:
                break

            # Calculăm scorul Buchholz curent: sum(wins[opp] - losses[opp])
            buchholz = {
                i: sum(wins[opp] - losses[opp] for opp in opponents[i])
                for i in active
            }

            pools = {}
            for idx in active:
                pools.setdefault((wins[idx], losses[idx]), []).append(idx)

            for record, pool in pools.items():
                # Sortăm descrescător după Buchholz, apoi crescător după VRS Seed (idx)
                pool.sort(key=lambda idx: (-buchholz[idx], idx))
                w_rec, l_rec = record
                is_decisive = all_bo3 or (w_rec == 2 or l_rec == 2)
                p_mat = prob_bo3 if is_decisive else prob_bo1

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

    res_df = pd.DataFrame({
        'VRS Seed': range(1, n_teams + 1),
        'Echipa': teams,
        'Rank': ranks,
        'WR 3M': [f"{w*100:.1f}%" for w in raw_wrs],
        '3-0 (%)': (counts_3_0 / n_sims) * 100,
        'Calificare Total (%)': (counts_qual / n_sims) * 100,
        '3-1 / 3-2 (%)': ((counts_qual - counts_3_0) / n_sims) * 100,
        '0-3 (%)': (counts_0_3 / n_sims) * 100
    })
    return res_df