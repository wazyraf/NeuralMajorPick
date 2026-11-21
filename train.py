import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
import os

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

def prepare_features(df):
    features = pd.DataFrame()
    features['rank_diff'] = df['rank_b'] - df['rank_a']
    features['log_rank_ratio'] = np.log1p(df['rank_b']) - np.log1p(df['rank_a'])
    features['wr_diff'] = df['winrate_3m_a'] - df['winrate_3m_b']
    features['wr_a'] = df['winrate_3m_a']
    features['wr_b'] = df['winrate_3m_b']
    features['is_bo1'] = (df['format'] == 'BO1').astype(float)
    features['is_major'] = (df['event_tier'] == 'Major').astype(float)
    return features

def train_model(csv_path="data/matches.csv", decay_lambda=0.003, epochs=200):
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Exponential Time Decay: w = exp(-lambda * zile_trecute)
    max_date = df['date'].max()
    days_ago = (max_date - df['date']).dt.days
    weights = np.exp(-decay_lambda * days_ago).values
    
    # Symmetric Data Augmentation
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
        'feature_cols': list(X_df.columns)
    }, "models/cs2_net.pt")
    
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
        
    return acc, len(df)

def evaluate_holdout(test_csv="data/test_hong_kong.csv"):
    if not os.path.exists(test_csv) or not os.path.exists("models/cs2_net.pt"):
        return None
    
    df_test = pd.read_csv(test_csv)
    checkpoint = torch.load("models/cs2_net.pt", weights_only=True)
    model = CS2MatchNet(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    X_test = prepare_features(df_test)
    X_scaled = scaler.transform(X_test.values).astype(np.float32)
    
    with torch.no_grad():
        probs_a = torch.sigmoid(model(torch.tensor(X_scaled))).numpy().flatten()
        
    df_res = df_test.copy()
    df_res['prob_a'] = probs_a
    df_res['predicted_winner'] = np.where(probs_a >= 0.5, df_res['team_a'], df_res['team_b'])
    df_res['actual_winner'] = np.where(df_res['winner'] == 1, df_res['team_a'], df_res['team_b'])
    df_res['correct'] = (probs_a >= 0.5).astype(int) == df_res['winner']
    df_res['confidence'] = np.maximum(probs_a, 1.0 - probs_a)
    
    accuracy = df_res['correct'].mean()
    brier_score = np.mean((probs_a - df_res['winner']) ** 2)
    
    return {
        'accuracy': accuracy,
        'brier_score': brier_score,
        'total_matches': len(df_res),
        'correct_count': int(df_res['correct'].sum()),
        'details': df_res
    }

def simulate_swiss_pickem(teams_df, model, scaler, n_sims=5000, is_major=True):
    """
    Simulează un Swiss Stage de 16 echipe (format Major CS2: BO1 în meciurile non-decisive, BO3 la 2-0 / 0-2 / 2-1 / 1-2 / 2-2).
    teams_df trebuie să conțină: ['team', 'rank', 'wr_3m'] ordonate după seed (1..16).
    """
    n_teams = len(teams_df)
    teams = teams_df['team'].tolist()
    ranks = teams_df['rank'].tolist()
    wrs = teams_df['wr_3m'].tolist()
    
    # Pre-calculăm matricea de probabilități pentru BO1 și BO3 între toate perechile
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
            'rank_a': ranks[i], 'rank_b': ranks[j],
            'winrate_3m_a': wrs[i], 'winrate_3m_b': wrs[j],
            'format': fmt, 'event_tier': 'Major' if is_major else 'S-Tier LAN'
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
        played = set()
        
        for round_num in range(1, 6):
            active = [i for i in range(n_teams) if wins[i] < 3 and losses[i] < 3]
            if not active:
                break
            # Grupăm după (wins, losses) și împerechem high seed vs low seed
            pools = {}
            for idx in active:
                pools.setdefault((wins[idx], losses[idx]), []).append(idx)
                
            for record, pool in pools.items():
                pool.sort() # Seed initial ca tie-breaker simplificat
                w_rec, l_rec = record
                is_decisive = (w_rec == 2 or l_rec == 2)
                p_mat = prob_bo3 if is_decisive else prob_bo1
                
                unpaired = pool.copy()
                while len(unpaired) >= 2:
                    t1 = unpaired.pop(0)
                    # Căutăm cel mai jos seed pe care nu l-a întâlnit încă
                    opp_idx = len(unpaired) - 1
                    while opp_idx > 0 and (t1, unpaired[opp_idx]) in played:
                        opp_idx -= 1
                    t2 = unpaired.pop(opp_idx)
                    played.add((t1, t2))
                    played.add((t2, t1))
                    
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
        'Seed': range(1, n_teams + 1),
        'Echipa': teams,
        'Rank': ranks,
        'WR 3M': [f"{w*100:.1f}%" for w in wrs],
        '3-0 (%)': (counts_3_0 / n_sims) * 100,
        'Calificare Total (%)': (counts_qual / n_sims) * 100,
        '3-1 / 3-2 (%)': ((counts_qual - counts_3_0) / n_sims) * 100,
        '0-3 (%)': (counts_0_3 / n_sims) * 100
    })
    return res_df

if __name__ == "__main__":
    acc, n_matches = train_model()
    print(f"[SUCCES] Model antrenat pe {n_matches} meciuri (x2 simetric). Acuratete interna: {acc*100:.1f}%")
    holdout = evaluate_holdout()
    if holdout:
        print(f"[HOLDOUT TEST] BLAST Hong Kong: {holdout['correct_count']}/{holdout['total_matches']} ({holdout['accuracy']*100:.1f}%) | Brier: {holdout['brier_score']:.4f}")