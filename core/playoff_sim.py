# Simulator determinist Single-Elimination (QF BO3 -> SF BO3 -> Final BO5)
import numpy as np
import pandas as pd
from core.model_engine import predict_match_context


def simulate_playoffs_exact(qf_teams_df: pd.DataFrame, model, scaler, event_tier: str = "Major"):
    """
    Calculează analitic exact probabilitățile pentru un bracket Single-Elimination de 8 echipe:
    - Rândurile 0-1: Quarter-Final 1 (BO3)
    - Rândurile 2-3: Quarter-Final 2 (BO3)
    - Rândurile 4-5: Quarter-Final 3 (BO3)
    - Rândurile 6-7: Quarter-Final 4 (BO3)
    - Semifinals: BO3 (câștigător QF1 vs QF2, câștigător QF3 vs QF4)
    - Grand Final: BO5
    """
    n = 8
    teams = qf_teams_df['team'].tolist()
    ranks = qf_teams_df['rank'].tolist()
    wrs = qf_teams_df['wr_3m'].tolist()

    p_bo3 = np.zeros((n, n))
    p_bo5 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                res3 = predict_match_context(model, scaler, ranks[i], ranks[j], wrs[i], wrs[j], "BO3", event_tier)
                res5 = predict_match_context(model, scaler, ranks[i], ranks[j], wrs[i], wrs[j], "BO5", event_tier)
                p_bo3[i, j] = res3['prob_a']
                p_bo5[i, j] = res5['prob_a']

    # Probabilități Quarter-Finals (4 meciuri BO3)
    qf_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    prob_reach_sf = np.zeros(n)
    qf_matchups = []

    for m_idx, (a, b) in enumerate(qf_pairs):
        pa = p_bo3[a, b]
        pb = 1.0 - pa
        prob_reach_sf[a] = pa
        prob_reach_sf[b] = pb
        qf_matchups.append({
            'match': f"QF{m_idx + 1}",
            'team_a': teams[a], 'prob_a': pa,
            'team_b': teams[b], 'prob_b': pb,
            'pick': teams[a] if pa >= pb else teams[b]
        })

    # Probabilități Semi-Finals (BO3)
    prob_reach_gf = np.zeros(n)
    # SF1: {0,1} vs {2,3}
    for a in [0, 1]:
        prob_reach_gf[a] = prob_reach_sf[a] * sum(prob_reach_sf[b] * p_bo3[a, b] for b in [2, 3])
    for b in [2, 3]:
        prob_reach_gf[b] = prob_reach_sf[b] * sum(prob_reach_sf[a] * p_bo3[b, a] for a in [0, 1])

    # SF2: {4,5} vs {6,7}
    for a in [4, 5]:
        prob_reach_gf[a] = prob_reach_sf[a] * sum(prob_reach_sf[b] * p_bo3[a, b] for b in [6, 7])
    for b in [6, 7]:
        prob_reach_gf[b] = prob_reach_sf[b] * sum(prob_reach_sf[a] * p_bo3[b, a] for a in [4, 5])

    # Probabilități Grand Final (BO5) -> Champion
    prob_champ = np.zeros(n)
    for a in [0, 1, 2, 3]:
        prob_champ[a] = prob_reach_gf[a] * sum(prob_reach_gf[b] * p_bo5[a, b] for b in [4, 5, 6, 7])
    for b in [4, 5, 6, 7]:
        prob_champ[b] = prob_reach_gf[b] * sum(prob_reach_gf[a] * p_bo5[b, a] for a in [0, 1, 2, 3])

    summary_df = pd.DataFrame({
        'Echipa': teams,
        'Rank': ranks,
        'WR 3M': [f"{w*100:.1f}%" for w in wrs],
        'Semifinals (%)': prob_reach_sf * 100,
        'Grand Final (%)': prob_reach_gf * 100,
        'Champion (%)': prob_champ * 100
    }).sort_values('Champion (%)', ascending=False).reset_index(drop=True)

    return summary_df, qf_matchups