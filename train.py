import argparse
import os
import pandas as pd

from core.model_engine import (
    CS2MatchNet,
    prepare_features,
    load_predictor,
    get_model_metadata,
    train_model as _core_train_model,
    evaluate_holdout as _core_evaluate_holdout,
)


def train_model(
    csv_path: str = "data/matches.csv",
    decay_lambda: float = 0.003,
    epochs: int = 250,
    cutoff_date: str | None = "2026-06-02",
):
    """Wrapper peste core.model_engine.train_model cu cutoff implicit pentru blind test."""
    return _core_train_model(
        csv_path=csv_path,
        decay_lambda=decay_lambda,
        epochs=epochs,
        cutoff_date=cutoff_date,
    )


def evaluate_holdout(
    test_csv: str = "data/matches.csv",
    cutoff_date: str | None = "2026-06-02",
    event_filter: str | None = "IEM Cologne Major 2026",
):
    """
    Evaluează modelul pe un set holdout/blind test (ex. meciurile din IEM Cologne Major 2026
    începând cu cutoff_date='2026-06-02').
    """
    if not os.path.exists(test_csv):
        return None

    df_test = pd.read_csv(test_csv)
    if "date" in df_test.columns and cutoff_date is not None:
        df_test = df_test[pd.to_datetime(df_test["date"]) >= pd.to_datetime(cutoff_date)].copy()
    if event_filter and "event_name" in df_test.columns:
        df_test = df_test[df_test["event_name"] == event_filter].copy()

    if df_test.empty:
        return None

    filtered_path = "models/_blind_eval_temp.csv"
    os.makedirs("models", exist_ok=True)
    df_test.to_csv(filtered_path, index=False)
    try:
        return _core_evaluate_holdout(test_csv=filtered_path, cutoff_date=cutoff_date)
    finally:
        if os.path.exists(filtered_path):
            os.remove(filtered_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antrenare și Blind Backtest CS2MatchNet")
    parser.add_argument("--matches-csv", default="data/matches.csv")
    parser.add_argument("--cutoff-date", default="2026-06-02", help="Data limită (ex. 2026-06-02 pentru Cologne)")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--decay", type=float, default=0.003)
    parser.add_argument("--no-cutoff", action="store_true", help="Antrenează pe tot dataset-ul fără cutoff")
    parser.add_argument("--event", default="IEM Cologne Major 2026")
    args = parser.parse_args()

    effective_cutoff = None if args.no_cutoff else args.cutoff_date
    acc, n_matches = train_model(
        csv_path=args.matches_csv,
        decay_lambda=args.decay,
        epochs=args.epochs,
        cutoff_date=effective_cutoff,
    )
    meta = get_model_metadata() or {}
    print(
        f"[SUCCES] Model antrenat pe {n_matches} meciuri (x2 simetric) | "
        f"Cutoff: {meta.get('cutoff_date', effective_cutoff)} | "
        f"Max train date: {meta.get('max_train_date', '?')} | "
        f"Acuratete interna: {acc*100:.1f}%"
    )

    holdout = evaluate_holdout(
        test_csv=args.matches_csv,
        cutoff_date=args.cutoff_date,
        event_filter=args.event,
    )
    if holdout:
        print(
            f"[BLIND BACKTEST - {args.event}] "
            f"{holdout['correct_count']}/{holdout['total_matches']} ({holdout['accuracy']*100:.1f}%) | "
            f"Brier: {holdout['brier_score']:.4f}"
        )