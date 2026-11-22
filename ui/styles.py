# CSS custom HLTV HUD + Bracket Pick'Em (3:0, 3:1/3:2, 0:3 + LOCK/VALUE/RISKY)
import streamlit as st


def inject_hud_css():
    st.markdown("""
    <style>
        .stApp { background-color: #0e1217; color: #e6edf3; }
        .hud-header {
            background: linear-gradient(90deg, #1b222c 0%, #242c37 60%, #2c2215 100%);
            border-left: 5px solid #de9b35;
            padding: 16px 22px;
            border-radius: 6px;
            margin-bottom: 18px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        }
        .prob-box {
            background-color: #161c24;
            border: 1px solid #2d3643;
            border-radius: 8px;
            padding: 18px;
            text-align: center;
        }
        .team-name { font-size: 22px; font-weight: 800; letter-spacing: 0.8px; }
        .prob-val-a { font-size: 38px; font-weight: 900; color: #58a6ff; }
        .prob-val-b { font-size: 38px; font-weight: 900; color: #de9b35; }

        /* HLTV Pick'Em Bracket Zones */
        .pickem-zone-green {
            background: linear-gradient(180deg, rgba(35,134,54,0.18) 0%, rgba(22,28,36,0.95) 100%);
            border: 1px solid #2ea043;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .pickem-zone-red {
            background: linear-gradient(180deg, rgba(218,54,51,0.18) 0%, rgba(22,28,36,0.95) 100%);
            border: 1px solid #f85149;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }
        .pickem-title {
            font-size: 13px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
            color: #c9d1d9;
        }
        .pick-slot {
            background-color: #1c2430;
            border: 1px solid #344050;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .badge-lock { background: #1f6feb; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 800; }
        .badge-value { background: #238636; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 800; }
        .badge-risky { background: #9e6a03; color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 800; }
    </style>
    """, unsafe_allow_html=True)


def get_confidence_badge(prob_pct: float, category: str) -> str:
    if category == "3-0":
        if prob_pct >= 42:
            return '<span class="badge-lock">LOCK</span>'
        elif prob_pct >= 24:
            return '<span class="badge-value">VALUE</span>'
        return '<span class="badge-risky">RISKY</span>'
    elif category == "0-3":
        if prob_pct >= 38:
            return '<span class="badge-lock">LOCK</span>'
        elif prob_pct >= 22:
            return '<span class="badge-value">VALUE</span>'
        return '<span class="badge-risky">RISKY</span>'
    else:  # 3-1 / 3-2 Advance
        if prob_pct >= 52:
            return '<span class="badge-lock">LOCK</span>'
        elif prob_pct >= 42:
            return '<span class="badge-value">VALUE</span>'
        return '<span class="badge-risky">RISKY</span>'


def render_hltv_pickem_bracket(df_3_0, df_adv, df_0_3):
    col_left, col_right = st.columns([1, 1.35])

    with col_left:
        slots_3_0 = "".join(
            f'<div class="pick-slot"><span><strong>#{int(r["VRS Seed"])} {r["Echipa"]}</strong></span><span>{r["3-0 (%)"]:.1f}% {get_confidence_badge(r["3-0 (%)"], "3-0")}</span></div>'
            for _, r in df_3_0.iterrows()
        )
        st.markdown(
            f'<div class="pickem-zone-green"><div class="pickem-title">🔥 3:0 ADVANCE UNDEFEATED (2 SLOTS)</div>{slots_3_0}</div>',
            unsafe_allow_html=True
        )

        slots_0_3 = "".join(
            f'<div class="pick-slot"><span><strong>#{int(r["VRS Seed"])} {r["Echipa"]}</strong></span><span>{r["0-3 (%)"]:.1f}% {get_confidence_badge(r["0-3 (%)"], "0-3")}</span></div>'
            for _, r in df_0_3.iterrows()
        )
        st.markdown(
            f'<div class="pickem-zone-red"><div class="pickem-title">💀 0:3 ELIMINATED WINLESS (2 SLOTS)</div>{slots_0_3}</div>',
            unsafe_allow_html=True
        )

    with col_right:
        slots_adv = "".join(
            f'<div class="pick-slot"><span><strong>#{int(r["VRS Seed"])} {r["Echipa"]}</strong> <span style="color:#8b949e; font-size:12px;">(Qual: {r["Calificare Total (%)"]:.1f}%)</span></span><span>{r["3-1 / 3-2 (%)"]:.1f}% {get_confidence_badge(r["3-1 / 3-2 (%)"], "ADV")}</span></div>'
            for _, r in df_adv.iterrows()
        )
        st.markdown(
            f'<div class="pickem-zone-green"><div class="pickem-title">✅ 3:1 OR 3:2 ADVANCE (6 SLOTS — ANTI 3-0 CURSE)</div>{slots_adv}</div>',
            unsafe_allow_html=True
        )