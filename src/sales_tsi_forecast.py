from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

EPSILON = 1e-12
DEFAULT_MIN_POSITIVE = 1.0
DEFAULT_STEP_DAYS = 30
CATEGORY_COLUMNS = ["大工业", "居民生活", "农业生产", "工商业", "趸售及其他"]


def parse_mixed_date(value: object) -> pd.Timestamp:
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d")
    m = re.fullmatch(r"(\d{4})/(\d{1,2})月", text)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    return pd.to_datetime(text)


def _clean_target_series(series: pd.Series, col_name: str) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if out.isna().any():
        warnings.warn(
            f"Missing values detected in target series '{col_name}'; applying forward/backward fill.",
            RuntimeWarning,
        )
    out = out.ffill().bfill()
    if (out <= 0).any():
        positive_values = out[out > 0]
        min_positive = float(positive_values.min()) if not positive_values.empty else DEFAULT_MIN_POSITIVE
        clipping_floor = max(min_positive, 1e-6)
        out = out.clip(lower=clipping_floor)
    return out


def load_excel(path: str, sheet: Optional[str], date_col: Optional[str], target_col: Optional[str]) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    if sheet is None:
        monthly_candidates = [s for s in xls.sheet_names if "月度" in s]
        sheet = monthly_candidates[0] if monthly_candidates else xls.sheet_names[0]
    df = pd.read_excel(path, sheet_name=sheet)
    df = df.dropna(how="all").reset_index(drop=True)

    if date_col is None:
        date_col = df.columns[0]
    if date_col not in df.columns:
        raise ValueError(f"Date column not found: {date_col}")

    df["date"] = df[date_col].map(parse_mixed_date)
    df = df.sort_values("date").reset_index(drop=True)

    if target_col and target_col in df.columns:
        df["target"] = _clean_target_series(df[target_col], target_col)
        return df[["date", "target"]]

    category_cols_present = [c for c in CATEGORY_COLUMNS if c in df.columns]
    if len(category_cols_present) == len(CATEGORY_COLUMNS):
        out_cols = ["date"]
        for c in CATEGORY_COLUMNS:
            df[c] = _clean_target_series(df[c], c)
            out_cols.append(c)
        return df[out_cols]

    total_like = [c for c in df.columns if "售电量" in str(c)]
    if total_like:
        chosen = total_like[0]
        df["target"] = _clean_target_series(df[chosen], chosen)
        return df[["date", "target"]]

    numeric_cols = [c for c in df.columns if c not in {date_col, "date"}]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["target"] = _clean_target_series(df[numeric_cols].sum(axis=1), "target_sum")
    return df[["date", "target"]]


def infer_period(dates: pd.Series, explicit_period: Optional[int]) -> int:
    if explicit_period is not None and explicit_period > 1:
        return explicit_period
    if len(dates) < 3:
        return 12
    day_delta = dates.diff().dt.days.dropna()
    med = float(day_delta.median()) if not day_delta.empty else 30.0
    return 7 if med <= 2 else 12


def tsi_decompose_and_forecast(df: pd.DataFrame, period: int, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    y = df["target"].to_numpy(dtype=float)
    n = len(y)

    trend = (
        pd.Series(y)
        .rolling(window=period, center=True, min_periods=max(2, period // 2))
        .mean()
        .ffill()
        .bfill()
        .to_numpy()
    )
    detrended = y / np.clip(trend, EPSILON, None)
    positions = np.arange(n) % period

    seasonal_index = np.ones(period, dtype=float)
    for p in range(period):
        vals = detrended[positions == p]
        seasonal_index[p] = float(np.nanmean(vals)) if len(vals) else 1.0
    seasonal_mean = float(np.nanmean(seasonal_index))
    if abs(seasonal_mean) < EPSILON:
        seasonal_index = np.ones(period, dtype=float)
    else:
        seasonal_index = seasonal_index / seasonal_mean
    seasonal = seasonal_index[positions]

    irregular = y / np.clip(trend * seasonal, EPSILON, None)
    irregular = pd.Series(irregular).replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy()
    trailing_window_size = min(period, len(irregular))
    irregular_coef = float(pd.Series(irregular).tail(trailing_window_size).mean())

    fitted = trend * seasonal * irregular

    t = np.arange(n).reshape(-1, 1)
    trend_model = LinearRegression().fit(t, trend)
    trend_intercept_fitted = float(trend_model.intercept_)
    trend_slope_fitted = float(trend_model.coef_[0])
    t_future = np.arange(n, n + horizon).reshape(-1, 1)
    trend_future = trend_model.predict(t_future)
    seasonal_future = seasonal_index[(np.arange(n, n + horizon) % period)]
    irregular_future = np.repeat(irregular_coef, horizon)
    forecast = trend_future * seasonal_future * irregular_future

    freq = pd.infer_freq(df["date"])
    if freq:
        offset = pd.tseries.frequencies.to_offset(freq)
    else:
        day_delta = df["date"].diff().dt.days.dropna()
        step_days = int(round(float(day_delta.median()))) if not day_delta.empty else DEFAULT_STEP_DAYS
        step_days = max(step_days, 1)
        offset = pd.offsets.Day(step_days)
    future_dates = [df["date"].iloc[-1] + (i + 1) * offset for i in range(horizon)]

    compare = pd.DataFrame(
        {
            "date": df["date"],
            "historical_sales": y,
            "trend_component": trend,
            "seasonal_component": seasonal,
            "irregular_component": irregular,
            "fitted_sales": fitted,
            "error": y - fitted,
        }
    )
    future = pd.DataFrame(
        {
            "date": future_dates,
            "forecast_sales": forecast,
            "trend_component": trend_future,
            "seasonal_component": seasonal_future,
            "irregular_component": irregular_future,
        }
    )
    metric_payload = {
        "mae": float(mean_absolute_error(compare["historical_sales"], compare["fitted_sales"])),
        "rmse": float(np.sqrt(mean_squared_error(compare["historical_sales"], compare["fitted_sales"]))),
        "mape": float(mean_absolute_percentage_error(compare["historical_sales"], compare["fitted_sales"])),
        "period": int(period),
        "irregular_coef_for_future": irregular_coef,
        "model_equations": {
            "decomposition": "Y_t = T_t × S_t × I_t",
            "trend": f"T_t = {trend_intercept_fitted:.6f} + {trend_slope_fitted:.6f} * t",
            "future_forecast": "Ŷ_(t+h) = T_(t+h) × S_((t+h) mod period) × Ī",
        },
    }
    return compare, future, metric_payload


def build_category_result_table(compare: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    hist = compare.copy()
    hist["phase"] = "historical"
    hist["forecast_sales"] = np.nan
    hist = hist[
        [
            "date",
            "phase",
            "historical_sales",
            "fitted_sales",
            "forecast_sales",
            "trend_component",
            "seasonal_component",
            "irregular_component",
            "error",
        ]
    ]

    fut = future.copy()
    fut["phase"] = "forecast"
    fut["historical_sales"] = np.nan
    fut["fitted_sales"] = np.nan
    fut["error"] = np.nan
    fut = fut[
        [
            "date",
            "phase",
            "historical_sales",
            "fitted_sales",
            "forecast_sales",
            "trend_component",
            "seasonal_component",
            "irregular_component",
            "error",
        ]
    ]
    return pd.concat([hist, fut], ignore_index=True)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", str(name)).strip("_") or "result"


def build_summary_table(
    category_compares: dict[str, pd.DataFrame], category_futures: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    hist_concat = pd.concat(
        [
            cdf.assign(category=cat)[["date", "category", "historical_sales", "fitted_sales"]]
            for cat, cdf in category_compares.items()
        ],
        ignore_index=True,
    )
    future_concat = pd.concat(
        [fdf.assign(category=cat)[["date", "category", "forecast_sales"]] for cat, fdf in category_futures.items()],
        ignore_index=True,
    )

    hist_total = hist_concat.groupby("date", as_index=False)[["historical_sales", "fitted_sales"]].sum()
    hist_total["phase"] = "historical"
    hist_total["forecast_sales"] = np.nan
    hist_total["error"] = hist_total["historical_sales"] - hist_total["fitted_sales"]

    future_total = future_concat.groupby("date", as_index=False)[["forecast_sales"]].sum()
    future_total["phase"] = "forecast"
    future_total["historical_sales"] = np.nan
    future_total["fitted_sales"] = np.nan
    future_total["error"] = np.nan

    summary = pd.concat([hist_total, future_total], ignore_index=True)[
        ["date", "phase", "historical_sales", "fitted_sales", "forecast_sales", "error"]
    ]
    return summary.sort_values(["date", "phase"]).reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSI decomposition forecast for electricity sales from Excel")
    parser.add_argument("--data", required=True, help="Absolute path of Excel file")
    parser.add_argument("--output", required=True, help="Absolute path of output directory")
    parser.add_argument("--sheet", default=None, help="Sheet name; default uses monthly sheet when available")
    parser.add_argument("--date-col", default=None, help="Date column name; default uses first column")
    parser.add_argument("--target-col", default=None, help="Target sales column; default auto-detect or sum")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon")
    parser.add_argument("--period", type=int, default=None, help="Seasonal period, e.g. 12 (month), 7 (day)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_excel(args.data, args.sheet, args.date_col, args.target_col)
    period = infer_period(df["date"], args.period)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    category_cols = [c for c in CATEGORY_COLUMNS if c in df.columns]
    if category_cols:
        category_metrics = {}
        category_compares: dict[str, pd.DataFrame] = {}
        category_futures: dict[str, pd.DataFrame] = {}
        for cat in category_cols:
            cat_df = df[["date", cat]].rename(columns={cat: "target"})
            compare, future, metric_payload = tsi_decompose_and_forecast(cat_df, period=period, horizon=args.horizon)
            category_compares[cat] = compare
            category_futures[cat] = future
            category_metrics[cat] = metric_payload

            cat_result = build_category_result_table(compare, future)
            cat_file = sanitize_filename(cat)
            cat_result.to_csv(output / f"{cat_file}_result.csv", index=False)

        summary = build_summary_table(category_compares, category_futures)
        summary.to_csv(output / "summary_result.csv", index=False)
        payload = {
            "period": int(period),
            "categories": category_cols,
            "category_metrics": category_metrics,
        }
        (output / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        compare, future, metric_payload = tsi_decompose_and_forecast(df, period=period, horizon=args.horizon)
        compare.to_csv(output / "historical_vs_fitted.csv", index=False)
        future.to_csv(output / "future_sales_forecast.csv", index=False)
        (output / "metrics.json").write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
