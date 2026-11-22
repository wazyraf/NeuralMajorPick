import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features_v4 import FEATURE_COLS_V4, build_features_v4, swap_features_v4


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", default="data/matches.csv")
    parser.add_argument("--stats", default="data/vrs_team_stats_cologne.csv")
    parser.add_argument("--cutoff", default="2026-06-02")
    parser.add_argument("--eval-event", default="Cologne")
    parser.add_argument("--eval-min-date", default="2026-06-02")
    parser.add_argument("--reg-c", type=float, default=0.35)
    parser.add_argument("--decay", type=float, default=0.003)
    args = parser.parse_args()

    df = pd.read_csv(args.matches)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt", "winner"]).sort_values("date_dt", ascending=True).reset_index(drop=True)

    train_df = df[df["date_dt"] < pd.to_datetime(args.cutoff)].copy().reset_index(drop=True)
    X_tr = build_features_v4(train_df, stats_csv=args.stats, fixed_as_of=None)
    X_tr_sw = swap_features_v4(X_tr)

    y_tr = train_df["winner"].astype(int).values
    y_tr_all = np.concatenate([y_tr, 1 - y_tr])
    X_tr_all = pd.concat([X_tr, X_tr_sw], ignore_index=True)

    days_ago = (train_df["date_dt"].max() - train_df["date_dt"]).dt.days.clip(lower=0).values
    w_all = np.concatenate([np.exp(-args.decay * days_ago), np.exp(-args.decay * days_ago)])

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=args.reg_c, max_iter=1000)),
    ])
    pipe.fit(X_tr_all.values, y_tr_all, clf__sample_weight=w_all)

    mask = df["event_name"].astype(str).str.contains(args.eval_event, case=False, na=False)
    if args.eval_min_date:
        mask = mask & (df["date_dt"] >= pd.to_datetime(args.eval_min_date))
    ev_df = df[mask].copy().reset_index(drop=True)

    probs = []
    for _, r in ev_df.iterrows():
        one_row = pd.DataFrame([r.to_dict()])
        as_of = str(pd.to_datetime(r["date"]).date())
        x_row = build_features_v4(one_row, stats_csv=args.stats, fixed_as_of=as_of)
        p_a = float(np.clip(sigmoid(pipe.decision_function(x_row.values))[0], 0.03, 0.97))
        probs.append(p_a)

    p_arr = np.array(probs)
    y_arr = ev_df["winner"].astype(int).values
    pred_bin = (p_arr >= 0.5).astype(int)

    acc = float((pred_bin == y_arr).mean())
    brier = float(np.mean((p_arr - y_arr) ** 2))

    print(f"V4 BLIND TEST ({args.eval_event}, n={len(ev_df)}): Acc={acc*100:.2f}% ({int((pred_bin==y_arr).sum())}/{len(ev_df)}) | Brier={brier:.4f}")
    for stage_name, grp in ev_df.assign(p=p_arr, ok=(pred_bin == y_arr)).groupby("stage", dropna=False):
        print(f"  - {stage_name}: {grp['ok'].mean()*100:.1f}% ({int(grp['ok'].sum())}/{len(grp)}) | Brier={np.mean((grp['p']-grp['winner'])**2):.4f}")


if __name__ == "__main__":
    main()