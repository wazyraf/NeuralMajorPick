import os
import streamlit as st
import pandas as pd
from core.model_engine import train_model, evaluate_holdout, get_model_metadata
from core.team_stats_engine import load_team_stats

def render_data_lab():
    st.markdown("### 🧪 Dataset Manager, Training & Out-of-Time Validation")
    csv_path = "data/matches.csv"

    meta = get_model_metadata()
    if meta:
        st.info(
            f"📡 **Model activ încărcat:** Antrenat pe **{meta['n_matches']} meciuri** | "
            f"Mod: **{meta['cutoff_date']}** | Ultimul meci din fereastra de antrenare: **{meta['max_train_date']}**"
        )

    if os.path.exists(csv_path):
        df_curr = pd.read_csv(csv_path)
        df_curr['date_dt'] = pd.to_datetime(df_curr['date'])
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Meciuri în CSV", len(df_curr))
        c2.metric("Turnee Unice", df_curr['event_name'].nunique() if 'event_name' in df_curr.columns else "-")
        c3.metric("Interval Date CSV", f"{df_curr['date_dt'].min().date()} → {df_curr['date_dt'].max().date()}")
    else:
        df_curr = None
        st.error("Fișierul `data/matches.csv` nu a fost găsit!")

    with st.expander("➕ Adaugă Meci Nou Direct în Dataset (`data/matches.csv`)", expanded=False):
        with st.form("add_match_form"):
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                m_date = st.date_input("Data meciului")
                m_event = st.text_input("Nume Turneu", value="StarLadder Budapest Major 2026")
                m_tier = st.selectbox("Event Tier", ["Major", "S-Tier LAN", "Tier-2 LAN"])
            with fc2:
                m_team_a = st.text_input("Team A", value="Vitality")
                m_rank_a = st.number_input("Rank A", 1, 200, 1)
                m_wr_a = st.number_input("3M Winrate A", 0.0, 1.0, 0.80, 0.01)
            with fc3:
                m_team_b = st.text_input("Team B", value="Spirit")
                m_rank_b = st.number_input("Rank B", 1, 200, 3)
                m_wr_b = st.number_input("3M Winrate B", 0.0, 1.0, 0.70, 0.01)

            fc4, fc5 = st.columns(2)
            with fc4:
                m_fmt = st.selectbox("Format", ["BO1", "BO3", "BO5"], index=1)
            with fc5:
                m_winner = st.selectbox("Câștigător", ["Team A (1)", "Team B (0)"])

            submitted = st.form_submit_button("💾 Salvează Meciul în CSV", type="primary")
            if submitted and df_curr is not None:
                new_row = {
                    'date': str(m_date),
                    'event_name': m_event,
                    'event_tier': m_tier,
                    'format': m_fmt,
                    'team_a': m_team_a,
                    'team_b': m_team_b,
                    'rank_a': int(m_rank_a),
                    'rank_b': int(m_rank_b),
                    'winrate_3m_a': float(m_wr_a),
                    'winrate_3m_b': float(m_wr_b),
                    'winner': 1 if "Team A" in m_winner else 0
                }
                cols_orig = [c for c in df_curr.columns if c != 'date_dt']
                filtered_row = {k: v for k, v in new_row.items() if k in cols_orig}
                df_updated = pd.concat([df_curr[cols_orig], pd.DataFrame([filtered_row])], ignore_index=True)
                df_updated.to_csv(csv_path, index=False)
                st.success(f"✅ Meciul **{m_team_a} vs {m_team_b}** a fost salvat!")
                st.rerun()

    st.markdown("---")
    st.subheader("🧠 Configurare Antrenare & Filtru Temporal (Blind Backtest)")

    use_cutoff = st.checkbox(
        "🔒 Mod Blind Backtest (Antrenează STRICT înainte de o dată limită, fără să atingi CSV-ul)",
        value=False
    )

    selected_cutoff = None
    filtered_count = len(df_curr) if df_curr is not None else 0

    if use_cutoff and df_curr is not None:
        min_d = df_curr['date_dt'].min().date()
        max_d = df_curr['date_dt'].max().date()
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            cutoff_input = st.date_input(
                "Data limită (exclusivă: date < cutoff)",
                value=max_d,
                min_value=min_d,
                max_value=max_d
            )
            selected_cutoff = str(cutoff_input)
        with col_d2:
            filtered_count = int((df_curr['date_dt'] < pd.to_datetime(selected_cutoff)).sum())
            st.warning(
                f"⏳ Vor fi folosite **{filtered_count} din {len(df_curr)} meciuri** "
                f"(toate meciurile jucate strict înainte de **{selected_cutoff}**)."
            )

    stats_preview = load_team_stats(csv_path="data/team_stats.csv", as_of_date=selected_cutoff)
    if stats_preview:
        st.caption(
            f"📊 **Team Stats (`data/team_stats.csv`):** {len(stats_preview)} echipe active "
            + (f"(filtrate `< {selected_cutoff}`)" if selected_cutoff else "(toate snapshot-urile)")
        )
    else:
        st.caption("📊 **Team Stats (`data/team_stats.csv`):** încă gol sau fără snapshot-uri înainte de data limită.")
        
    t1, t2 = st.columns(2)
    with t1:
        decay_val = st.slider("Exponential Time Decay (λ)", 0.000, 0.010, 0.003, 0.001, format="%.3f")
    with t2:
        epochs_val = st.number_input("Epoci antrenare", 50, 1000, 250, 50)

    btn_label = (
        f"🔒 ANTRENEAZĂ BLIND ({filtered_count} MECIURI < {selected_cutoff})"
        if use_cutoff else
        f"🧠 ANTRENEAZĂ PE TOT DATASETUL ({filtered_count} MECIURI)"
    )

    if st.button(btn_label, type="primary", use_container_width=True):
        try:
            with st.spinner("Antrenăm NeuralMajorNet..."):
                acc, n = train_model(
                    csv_path=csv_path,
                    decay_lambda=decay_val,
                    epochs=epochs_val,
                    cutoff_date=selected_cutoff
                )
            st.success(f"✅ Antrenare completă pe **{n} meciuri** ({n*2} exemple simetrice) — Acuratețe internă: **{acc*100:.1f}%**")
            st.rerun()
        except ValueError as err:
            st.error(f"❌ {err}")

    st.markdown("---")
    st.subheader("🧪 Hold-Out Validation Benchmark")
    if os.path.exists("data/test_hong_kong.csv"):
        holdout = evaluate_holdout("data/test_hong_kong.csv")
        if holdout:
            b1, b2, b3 = st.columns(3)
            b1.metric("Acuratețe Hold-Out", f"{holdout['accuracy']*100:.1f}% ({holdout['correct_count']}/{holdout['total_matches']})")
            b2.metric("Brier Score", f"{holdout['brier_score']:.4f}")
            b3.metric("Meciuri Test", holdout['total_matches'])