from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


def parse_mixed_date(value: object) -> pd.Timestamp:
    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return pd.to_datetime(text, format="%Y%m%d")
    m = re.fullmatch(r"(\d{4})/(\d{1,2})月", text)
    if m:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    return pd.to_datetime(text)


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
        df["target"] = pd.to_numeric(df[target_col], errors="coerce")
    else:
        total_like = [c for c in df.columns if "售电量" in str(c)]
        if total_like:
            df["target"] = pd.to_numeric(df[total_like[0]], errors="coerce")
        else:
            numeric_cols = [c for c in df.columns if c not in {date_col, "date"}]
            for c in numeric_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df["target"] = df[numeric_cols].sum(axis=1)

    df["target"] = df["target"].ffill().bfill()
    if (df["target"] <= 0).any():
        positive_values = df["target"][df["target"] > 0]
        min_positive = float(positive_values.min()) if not positive_values.empty else 1.0
        floor = max(min_positive, 1e-6)
        df["target"] = df["target"].clip(lower=floor)
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
    detrended = y / np.clip(trend, 1e-12, None)
    positions = np.arange(n) % period

    seasonal_index = np.ones(period, dtype=float)
    for p in range(period):
        vals = detrended[positions == p]
        seasonal_index[p] = float(np.nanmean(vals)) if len(vals) else 1.0
    seasonal_index = seasonal_index / np.mean(seasonal_index)
    seasonal = seasonal_index[positions]

    irregular = y / np.clip(trend * seasonal, 1e-12, None)
    irregular = pd.Series(irregular).replace([np.inf, -np.inf], np.nan).ffill().bfill().to_numpy()
    irregular_coef = float(pd.Series(irregular).tail(min(period, len(irregular))).mean())

    fitted = trend * seasonal * irregular

    t = np.arange(n).reshape(-1, 1)
    trend_model = LinearRegression().fit(t, trend)
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
        step_days = int(round(float(day_delta.median()))) if not day_delta.empty else 30
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
    }
    return compare, future, metric_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSI decomposition forecast for electricity sales from Excel")
    parser.add_argument("--data", required=True, help="Absolute path of Excel file")
    parser.add_argument("--output", required=True, help="Absolute path of output directory")
    parser.add_argument("--sheet", default=None, help="Sheet name; default uses monthly sheet when available")
    parser.add_argument("--date-col", default=None, help="Date column name; default uses first column")
    parser.add_argument("--target-col", default=None, help="Target sales column; default auto-detect or sum")
    parser.add_argument("--horizon", type=int, default=12, help="Forecast horizon")
    parser.add_argument("--period", type=int, default=None, help="Seasonal period, e.g. 12(month), 7(day)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_excel(args.data, args.sheet, args.date_col, args.target_col)
    period = infer_period(df["date"], args.period)
    compare, future, metric_payload = tsi_decompose_and_forecast(df, period=period, horizon=args.horizon)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    compare.to_csv(output / "historical_vs_fitted.csv", index=False)
    future.to_csv(output / "future_sales_forecast.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
