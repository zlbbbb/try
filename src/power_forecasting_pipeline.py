from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class PipelineConfig:
    date_col: str
    target_col: str
    horizon: int = 6
    top_k: int = 5
    robust_radius: float = 1.0
    test_splits: int = 5
    robust_radius_candidates: Tuple[float, ...] = (0.01, 0.1, 1.0, 3.0, 10.0)
    scenario_delta: float = 0.02
    gra_rho: float = 0.5
    iqr_multiplier: float = 1.5
    interval_alpha: float = 0.2


def load_data(path: str, date_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")
    df[date_col] = pd.to_datetime(df[date_col])
    return df.sort_values(date_col).reset_index(drop=True)


def preprocess(df: pd.DataFrame, date_col: str, target_col: str, iqr_multiplier: float) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [c for c in out.columns if c != date_col]
    out[numeric_cols] = out[numeric_cols].apply(pd.to_numeric, errors="coerce")
    out[numeric_cols] = out[numeric_cols].ffill().bfill()

    for col in [c for c in numeric_cols if c != target_col]:
        q1, q3 = out[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low = q1 - iqr_multiplier * iqr
        high = q3 + iqr_multiplier * iqr
        out[col] = out[col].clip(lower=low, upper=high)
    return out


def grey_relation_scores(df: pd.DataFrame, target_col: str, feature_cols: List[str], rho: float = 0.5) -> pd.Series:
    target = df[target_col].to_numpy(dtype=float)
    target_norm = (target - target.min()) / (target.max() - target.min() + 1e-12)

    scores = {}
    for col in feature_cols:
        x = df[col].to_numpy(dtype=float)
        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
        diff = np.abs(target_norm - x_norm)
        d_min = diff.min()
        d_max = diff.max() + 1e-12
        rel = (d_min + rho * d_max) / (diff + rho * d_max)
        scores[col] = float(rel.mean())
    return pd.Series(scores).sort_values(ascending=False)


def make_supervised(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].copy()
    X = df[feature_cols].copy()
    return X, y


def scale_with_train(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mean = train.mean(axis=0)
    std = train.std(axis=0).replace(0, 1.0)
    return (train - mean) / std, (test - mean) / std


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def rolling_backtest(
    X: pd.DataFrame,
    y: pd.Series,
    robust_radius: float,
    splits: int,
) -> Dict[str, Dict[str, float]]:
    tscv = TimeSeriesSplit(n_splits=splits)
    baseline_hist_scores = []
    baseline_lr_scores = []
    dro_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train_s, X_test_s = scale_with_train(X_train, X_test)

        hist_pred = np.repeat(y_train.iloc[-1], len(y_test))
        baseline_hist_scores.append(metrics(y_test.to_numpy(), hist_pred))

        lr = LinearRegression()
        lr.fit(X_train_s, y_train)
        baseline_lr_scores.append(metrics(y_test.to_numpy(), lr.predict(X_test_s)))

        # DRO approximation: robust radius -> stronger Ridge regularization
        alpha = max(1e-6, robust_radius)
        dro = Ridge(alpha=alpha)
        dro.fit(X_train_s, y_train)
        dro_scores.append(metrics(y_test.to_numpy(), dro.predict(X_test_s)))

    def avg(scores: List[Dict[str, float]]) -> Dict[str, float]:
        return {k: float(np.mean([s[k] for s in scores])) for k in ["mae", "rmse", "mape"]}

    return {
        "baseline_history_only": avg(baseline_hist_scores),
        "baseline_linear_regression": avg(baseline_lr_scores),
        "dro_ridge": avg(dro_scores),
    }


def fit_models_and_forecast(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    selected_features: List[str],
) -> pd.DataFrame:
    X, y = make_supervised(df, cfg.target_col, selected_features)
    X_s, _ = scale_with_train(X, X)

    lr = LinearRegression().fit(X_s, y)
    dro = Ridge(alpha=max(1e-6, cfg.robust_radius)).fit(X_s, y)

    residual = y.to_numpy() - dro.predict(X_s)
    residual_centered = residual - residual.mean()
    safe_alpha = float(min(max(cfg.interval_alpha, 1e-6), 0.99))
    q_low, q_high = np.quantile(
        residual_centered,
        [safe_alpha / 2.0, 1.0 - safe_alpha / 2.0],
    )
    interval_coverage = int(round((1.0 - safe_alpha) * 100))

    last_x = X.iloc[[-1]].copy()
    perturbation_ratio = float(abs(cfg.scenario_delta))
    scenario_multipliers = {"low": 1.0 - perturbation_ratio, "base": 1.00, "high": 1.0 + perturbation_ratio}
    rows = []
    for step in range(1, cfg.horizon + 1):
        base_input = last_x.copy()
        base_input_s, _ = scale_with_train(X, base_input)
        base_pred = float(dro.predict(base_input_s)[0])

        scenario_forecasts = {}
        for name, mul in scenario_multipliers.items():
            inp = base_input * mul
            inp_s, _ = scale_with_train(X, inp)
            scenario_forecasts[name] = float(dro.predict(inp_s)[0])

        rows.append(
            {
                "horizon_step": step,
                "point_forecast": base_pred,
                f"interval_lower_{interval_coverage}": base_pred + float(q_low),
                f"interval_upper_{interval_coverage}": base_pred + float(q_high),
                "scenario_low": scenario_forecasts["low"],
                "scenario_base": scenario_forecasts["base"],
                "scenario_high": scenario_forecasts["high"],
                "baseline_linear_reference": float(lr.predict(base_input_s)[0]),
            }
        )
    return pd.DataFrame(rows)


def choose_robust_radius(X: pd.DataFrame, y: pd.Series, candidates: List[float], splits: int = 4) -> float:
    tscv = TimeSeriesSplit(n_splits=splits)
    best_r, best_score = candidates[0], float("inf")
    for r in candidates:
        fold_rmse = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            X_train_s, X_test_s = scale_with_train(X_train, X_test)
            model = Ridge(alpha=max(1e-6, r)).fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
            fold_rmse.append(np.sqrt(mean_squared_error(y_test, pred)))
        rmse = float(np.mean(fold_rmse))
        if rmse < best_score:
            best_score, best_r = rmse, r
    return best_r


def run_pipeline(data_path: str, output_dir: str, cfg: PipelineConfig) -> None:
    df = load_data(data_path, cfg.date_col)
    df = preprocess(df, cfg.date_col, cfg.target_col, iqr_multiplier=cfg.iqr_multiplier)

    feature_cols = [c for c in df.columns if c not in [cfg.date_col, cfg.target_col]]
    if not feature_cols:
        raise ValueError("At least one feature column is required.")

    gra_scores = grey_relation_scores(df, cfg.target_col, feature_cols, rho=cfg.gra_rho)
    selected = gra_scores.head(min(cfg.top_k, len(gra_scores))).index.tolist()

    X, y = make_supervised(df, cfg.target_col, selected)
    cfg.robust_radius = choose_robust_radius(
        X,
        y,
        candidates=list(cfg.robust_radius_candidates),
        splits=max(2, cfg.test_splits - 1),
    )
    eval_result = rolling_backtest(X, y, robust_radius=cfg.robust_radius, splits=cfg.test_splits)
    future = fit_models_and_forecast(df, cfg, selected)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gra_scores.rename("grey_relation_score").to_csv(out_dir / "grey_relation_scores.csv", header=True)
    future.to_csv(out_dir / "future_forecast.csv", index=False)

    payload = {
        "config": {
            "date_col": cfg.date_col,
            "target_col": cfg.target_col,
            "horizon": cfg.horizon,
            "top_k": cfg.top_k,
            "robust_radius_selected": cfg.robust_radius,
            "gra_rho": cfg.gra_rho,
            "iqr_multiplier": cfg.iqr_multiplier,
            "interval_alpha": cfg.interval_alpha,
            "metrics": ["mae", "rmse", "mape"],
        },
        "selected_features": selected,
        "backtest_metrics": eval_result,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Province electricity forecasting with GRA + DRO pipeline")
    parser.add_argument("--data", required=True, help="Absolute path to input CSV")
    parser.add_argument("--output", required=True, help="Absolute path to output folder")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--target-col", default="target")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--robust-radius", type=float, default=1.0)
    parser.add_argument("--test-splits", type=int, default=5)
    parser.add_argument("--scenario-delta", type=float, default=0.02, help="Scenario perturbation ratio, e.g. 0.02 for ±2%")
    parser.add_argument("--gra-rho", type=float, default=0.5, help="Grey relation distinguishing coefficient")
    parser.add_argument("--iqr-multiplier", type=float, default=1.5, help="IQR multiplier for outlier clipping")
    parser.add_argument("--interval-alpha", type=float, default=0.2, help="Prediction interval tail mass, 0.2 => 80% interval")
    parser.add_argument(
        "--robust-candidates",
        default="0.01,0.1,1.0,3.0,10.0",
        help="Comma-separated robust radius candidates for CV selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        robust_candidates = tuple(float(v.strip()) for v in str(args.robust_candidates).split(",") if v.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid --robust-candidates value: '{args.robust_candidates}'. "
            "Use comma-separated floats, e.g. 0.01,0.1,1,3,10"
        ) from exc
    if not robust_candidates:
        raise ValueError("At least one robust radius candidate is required.")
    cfg = PipelineConfig(
        date_col=args.date_col,
        target_col=args.target_col,
        horizon=args.horizon,
        top_k=args.top_k,
        robust_radius=args.robust_radius,
        test_splits=args.test_splits,
        robust_radius_candidates=robust_candidates,
        scenario_delta=args.scenario_delta,
        gra_rho=args.gra_rho,
        iqr_multiplier=args.iqr_multiplier,
        interval_alpha=args.interval_alpha,
    )
    run_pipeline(args.data, args.output, cfg)


if __name__ == "__main__":
    main()
