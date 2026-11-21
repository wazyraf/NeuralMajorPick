import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import os
from train import CS2MatchNet, prepare_features, train_model, evaluate_holdout, simulate_swiss_pickem

st.set_page_config(page_title="CS2 Neural HUD // Pick'Em Engine", page_icon="🎯", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e6edf3; }
    .hud-header {
        background: linear-gradient(90deg, #1b222c 0%, #2a1f14 100%);
        border-left: 5px solid #de9b35;
        padding: 15px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    .prob-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .team-name { font-size: 24px; font-weight: 800; letter-spacing: 1px; }
    .prob-val-a { font-size: 42px; font-weight: 900; color: #58a6ff; }
    .prob-val-b { font-size: 42px; font-weight: 900; color: #de9b35; }
    .tier-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hud-header">
    <h2 style="margin:0; color:#de9b35;">🎯 CS2 NEURAL PREDICTOR // MAJOR PICK'EM HUD</h2>
    <p style="margin:0; color:#8b949e; font-size:14px;">PyTorch MLP • Exponential Time Decay • Monte Carlo Swiss Simulator • Hold-Out Validation</p>
</div>
""", unsafe_allow_html=True)

def load_predictor():
    if not os.path.exists("models/cs2_net.pt"):
        return None, None
    checkpoint = torch.load("models/cs2_net.pt", weights_only=True)
    model = CS2MatchNet(checkpoint['input_dim'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

tab1, tab2, tab3 = st.tabs([
    "⚔️ MATCH PREDICTOR",
    "🏆 PICK'EM SWISS SIMULATOR",
    "🧪 BACKTEST & RETRAIN ENGINE"
])

# --- TAB 1: MATCH PREDICTOR ---
with tab1:
    model, scaler = load_predictor()
    if model is None:
        st.warning("⚠️ Modelul nu este încă antrenat! Mergi în tab-ul **BACKTEST & RETRAIN ENGINE** și apasă **ANTRENEAZĂ REȚEAUA**.")
    else:
        col_a, col_mid, col_b = st.columns([2, 1, 2])
        
        with col_a:
            st.subheader("🔵 TEAM A (CT / Left)")
            team_a = st.text_input("Nume Echipă A", value="MOUZ")
            rank_a = st.number_input("HLTV Rank A", min_value=1, max_value=200, value=3)
            wr_a = st.slider("Winrate ultimele 3 luni (A)", 0.0, 1.0, 0.65, 0.01)

        with col_mid:
            st.subheader("⚙️ MATCH CONTEXT")
            match_format = st.selectbox("Format", ["BO1", "BO3", "BO5"], index=1)
            event_tier = st.selectbox("Event Tier", ["Major", "S-Tier LAN", "Tier-2 LAN"], index=0)
            predict_btn = st.button("🔥 CALCULATE ODDS", use_container_width=True, type="primary")

        with col_b:
            st.subheader("🟠 TEAM B (T / Right)")
            team_b = st.text_input("Nume Echipă B", value="FURIA")
            rank_b = st.number_input("HLTV Rank B", min_value=1, max_value=200, value=6)
            wr_b = st.slider("Winrate ultimele 3 luni (B)", 0.0, 1.0, 0.56, 0.01)

        if predict_btn:
            sample = pd.DataFrame([{
                'rank_a': rank_a, 'rank_b': rank_b,
                'winrate_3m_a': wr_a, 'winrate_3m_b': wr_b,
                'format': match_format, 'event_tier': event_tier
            }])
            X_scaled = scaler.transform(prepare_features(sample).values).astype(np.float32)
            
            with torch.no_grad():
                prob_a = torch.sigmoid(model(torch.tensor(X_scaled))).item()
            prob_b = 1.0 - prob_a
            max_p = max(prob_a, prob_b)
            fav_team = team_a if prob_a >= prob_b else team_b

            if max_p >= 0.72:
                tier_txt, tier_bg = f"🔒 LOCK ({fav_team})", "#1f6feb"
            elif max_p >= 0.60:
                tier_txt, tier_bg = f"⚖️ SOLID LEAN ({fav_team})", "#238636"
            else:
                tier_txt, tier_bg = f"🎲 TOSS-UP / UPSET ALERT ({match_format})", "#9e6a03"

            st.markdown("---")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"""
                <div class="prob-box">
                    <div class="team-name">{team_a.upper()}</div>
                    <div class="prob-val-a">{prob_a*100:.1f}%</div>
                    <div style="color:#8b949e;">HLTV Rank #{rank_a} • 3M WR {wr_a*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with r2:
                st.markdown(f"""
                <div class="prob-box">
                    <div class="team-name">{team_b.upper()}</div>
                    <div class="prob-val-b">{prob_b*100:.1f}%</div>
                    <div style="color:#8b949e;">HLTV Rank #{rank_b} • 3M WR {wr_b*100:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"<div style='text-align:center;'><span class='tier-badge' style='background:{tier_bg}; color:white;'>{tier_txt}</span></div>", unsafe_allow_html=True)
            st.write("")
            st.progress(prob_a)

# --- TAB 2: PICK'EM SWISS SIMULATOR ---
with tab2:
    st.subheader("🏆 Simulator Pick'Em Swiss Stage (Monte Carlo)")
    st.caption("Editează direct în tabel cele 16 echipe (în ordinea seed-urilor 1–16), HLTV Rank-ul și Winrate-ul lor pe ultimele 3 luni (ex: 0.62).")
    
    model, scaler = load_predictor()
    if model is None:
        st.warning("⚠️ Antrenează mai întâi rețeaua în tab-ul 3!")
    else:
        default_teams = pd.DataFrame({
            'team': [
                'Liquid', 'GamerLegion', '3DMAX', 'Astralis',
                'paiN', 'MIBR', 'B8', 'Imperial',
                'M80', 'Legacy', 'TYLOO', 'Lynn Vision',
                'FlyQuest', 'NRG', 'Wildcard', 'Rare Atom'
            ],
            'rank': [12, 14, 15, 16, 18, 19, 21, 23, 25, 26, 28, 30, 32, 34, 38, 45],
            'wr_3m': [0.58, 0.55, 0.56, 0.57, 0.59, 0.54, 0.53, 0.60, 0.57, 0.61, 0.64, 0.66, 0.52, 0.55, 0.51, 0.49]
        })
        
        edited_teams = st.data_editor(default_teams, num_rows="fixed", use_container_width=True)
        
        c_sim1, c_sim2 = st.columns([1, 2])
        with c_sim1:
            n_sims = st.selectbox("Număr simulări Monte Carlo", [2000, 5000, 10000], index=1)
        with c_sim2:
            run_sim = st.button("🚀 RULEAZĂ SIMULAREA PICK'EM", type="primary", use_container_width=True)
            
        if run_sim:
            with st.spinner(f"Simulăm {n_sims} turnee Swiss în PyTorch..."):
                sim_res = simulate_swiss_pickem(edited_teams, model, scaler, n_sims=n_sims)
            
            # Recomandări automate Pick'Em
            top_3_0 = sim_res.sort_values('3-0 (%)', ascending=False).head(2)['Echipa'].tolist()
            top_0_3 = sim_res.sort_values('0-3 (%)', ascending=False).head(2)['Echipa'].tolist()
            # Pentru Advance (3-1 / 3-2) evităm echipele puse deja la 3-0!
            adv_candidates = sim_res[~sim_res['Echipa'].isin(top_3_0)].sort_values('Calificare Total (%)', ascending=False).head(6)['Echipa'].tolist()
            
            st.success("🎯 **RECOMANDARE OPTIMALĂ PICK'EM (Anti-3-0 Curse):**")
            p1, p2, p3 = st.columns(3)
            p1.metric("🔥 3-0 Picks (2 echipe)", ", ".join(top_3_0))
            p2.metric("✅ Advance Picks (6 echipe)", ", ".join(adv_candidates))
            p3.metric("💀 0-3 Picks (2 echipe)", ", ".join(top_0_3))
            
            st.dataframe(
                sim_res.style.format({
                    '3-0 (%)': '{:.1f}%',
                    'Calificare Total (%)': '{:.1f}%',
                    '3-1 / 3-2 (%)': '{:.1f}%',
                    '0-3 (%)': '{:.1f}%'
                }),
                use_container_width=True
            )

# --- TAB 3: BACKTEST & RETRAIN ---
with tab3:
    st.subheader("🧠 Antrenare Rețea & Validare Out-of-Time")
    
    if os.path.exists("data/matches.csv"):
        df_curr = pd.read_csv("data/matches.csv")
        c_m1, c_m2, c_m3 = st.columns(3)
        c_m1.metric("Total Meciuri Antrenare", len(df_curr))
        c_m2.metric("Turnee Incluse", df_curr['event_name'].nunique())
        c_m3.metric("Ultima Dată din Dataset", str(df_curr['date'].max()))
    
    c1, c2 = st.columns(2)
    with c1:
        decay_val = st.slider("Time Decay (λ) — pondere meta recent", 0.000, 0.010, 0.003, 0.001, format="%.3f")
    with c2:
        epochs_val = st.number_input("Epoci antrenare", min_value=50, max_value=1000, value=250, step=50)
        
    if st.button("🧠 ANTRENEAZĂ PE CELE 576 MECIURI", type="primary"):
        with st.spinner("Antrenăm rețeaua neuronală..."):
            acc, n = train_model(decay_lambda=decay_val, epochs=epochs_val)
        st.success(f"✅ Model antrenat pe {n} meciuri (x2 simetric = {n*2} exemple)! Acuratețe internă: **{acc*100:.1f}%**")

    st.markdown("---")
    st.subheader("🧪 Blind Backtest: BLAST Hong Kong (Hold-Out Test)")
    
    if os.path.exists("data/test_hong_kong.csv"):
        holdout = evaluate_holdout("data/test_hong_kong.csv")
        if holdout:
            b1, b2, b3 = st.columns(3)
            b1.metric("Acuratețe BLAST Hong Kong", f"{holdout['accuracy']*100:.1f}% ({holdout['correct_count']}/{holdout['total_matches']})")
            b2.metric("Brier Score (Calibrare)", f"{holdout['brier_score']:.4f}", help="Mai mic = mai bine (0.25 = random, <0.20 = foarte bun în esports)")
            b3.metric("Meciuri Testate (Nevăzute)", holdout['total_matches'])
            
            disp = holdout['details'][['date', 'stage', 'format', 'team_a', 'team_b', 'prob_a', 'predicted_winner', 'actual_winner', 'correct']].copy()
            disp['prob_a'] = (disp['prob_a'] * 100).round(1).astype(str) + '%'
            st.dataframe(disp, use_container_width=True)
    else:
        st.info("ℹ️ Creează fișierul `data/test_hong_kong.csv` cu meciurile din BLAST Hong Kong pentru a vedea aici scorul exact de backtesting!")