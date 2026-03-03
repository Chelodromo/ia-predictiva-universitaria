"""Train baseline forecasting models and log results to MLflow.

This script reads the latest CSV snapshots from MinIO/S3, trains simple
forecasting baselines, and logs metrics/artifacts in MLflow.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import tempfile

import boto3
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

SNAPSHOT_REGEX = re.compile(r"snapshot_(\d{8}T\d{6})\.csv$")

DEFAULT_BUCKET = os.getenv("DATA_REPO_BUCKET_NAME", "data")
DEFAULT_PREFIX = "mysql_exports/unsta"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def _list_snapshot_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        for obj in response.get("Contents", []):
            key = obj["Key"]
            if SNAPSHOT_REGEX.search(key):
                keys.append(key)
        if not response.get("IsTruncated"):
            break
        continuation_token = response.get("NextContinuationToken")
    return keys


def _latest_snapshot_key(keys: list[str]) -> str:
    if not keys:
        raise ValueError("No snapshot files found for requested prefix.")
    return sorted(keys, key=lambda k: SNAPSHOT_REGEX.search(k).group(1))[-1]


def _read_csv_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read()
    return pd.read_csv(io.BytesIO(content))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def _first_existing(options: list[str], columns: list[str]) -> str | None:
    for col in options:
        if col in columns:
            return col
    return None


def _infer_target_column(df: pd.DataFrame, excluded: set[str]) -> str:
    preferred = [
        "matriculados",
        "total_matriculados",
        "cantidad_matriculados",
        "n_matriculados",
        "cantidad",
        "total",
    ]
    for col in preferred:
        if col in df.columns and col not in excluded:
            return col

    numeric_candidates = [
        c
        for c in df.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_candidates:
        raise ValueError("Could not infer target column for forecasting.")
    return numeric_candidates[0]


def _build_series(
    df: pd.DataFrame,
    target_col: str,
    career_col: str | None = None,
) -> dict[str, pd.Series]:
    year_col = _first_existing(["periodo", "anio", "year"], list(df.columns))
    term_col = _first_existing(
        ["id_cuatrimestre", "cuatrimestre", "term", "semester"],
        list(df.columns),
    )
    if year_col is None:
        raise ValueError("Missing year column (expected one of: periodo, anio, year).")

    work = df.copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work = work.dropna(subset=[year_col])
    work[year_col] = work[year_col].astype(int)

    if term_col is not None:
        work[term_col] = pd.to_numeric(work[term_col], errors="coerce").fillna(0).astype(int)
        work["time_order"] = work[year_col] * 10 + work[term_col]
    else:
        work["time_order"] = work[year_col]

    if career_col:
        grouped = (
            work.groupby([career_col, "time_order"], as_index=False)[target_col]
            .sum()
            .sort_values([career_col, "time_order"])
        )
        out: dict[str, pd.Series] = {}
        for career_id, chunk in grouped.groupby(career_col):
            series = pd.Series(
                chunk[target_col].values.astype(float),
                index=chunk["time_order"].values,
                name=str(career_id),
            )
            out[str(career_id)] = series
        return out

    grouped = work.groupby("time_order", as_index=False)[target_col].sum().sort_values("time_order")
    series = pd.Series(
        grouped[target_col].values.astype(float),
        index=grouped["time_order"].values,
        name="total",
    )
    return {"total": series}


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom < 1e-9, np.nan, denom)
    return float(np.nanmean(np.abs(y_true - y_pred) / denom) * 100.0)


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": _mape(y_true, y_pred),
        "smape": _smape(y_true, y_pred),
    }


def _split_index(series: pd.Series, train_fraction: float = 0.8, min_train_size: int = 8) -> int:
    n = len(series)
    split = max(min_train_size, int(np.floor(n * train_fraction)))
    split = min(split, n - 1)
    if split <= 0 or split >= n:
        raise ValueError("Series too short for train/test split.")
    return split


def _predict_naive(series: pd.Series, split_idx: int) -> tuple[np.ndarray, np.ndarray]:
    shifted = series.shift(1)
    y_true = series.iloc[split_idx:]
    y_pred = shifted.iloc[split_idx:]
    valid = ~y_pred.isna()
    return y_true[valid].values, y_pred[valid].values


def _predict_seasonal_naive(series: pd.Series, split_idx: int, season_lag: int) -> tuple[np.ndarray, np.ndarray]:
    shifted = series.shift(season_lag)
    y_true = series.iloc[split_idx:]
    y_pred = shifted.iloc[split_idx:]
    valid = ~y_pred.isna()
    return y_true[valid].values, y_pred[valid].values


def _predict_moving_average(series: pd.Series, split_idx: int, window: int) -> tuple[np.ndarray, np.ndarray]:
    moving = series.rolling(window=window).mean().shift(1)
    y_true = series.iloc[split_idx:]
    y_pred = moving.iloc[split_idx:]
    valid = ~y_pred.isna()
    return y_true[valid].values, y_pred[valid].values


def _supervised_lag_frame(series: pd.Series, lags: int) -> pd.DataFrame:
    data = {"y": series.values}
    for lag in range(1, lags + 1):
        data[f"lag_{lag}"] = series.shift(lag).values
    frame = pd.DataFrame(data, index=series.index).dropna()
    return frame


def _predict_linear_lags(series: pd.Series, split_idx: int, lags: int) -> tuple[np.ndarray, np.ndarray]:
    frame = _supervised_lag_frame(series, lags=lags)
    split_time = series.index[split_idx]
    train = frame.loc[frame.index < split_time]
    test = frame.loc[frame.index >= split_time]
    if train.empty or test.empty:
        raise ValueError("Not enough samples for linear-lag model.")
    x_train = train[[f"lag_{i}" for i in range(1, lags + 1)]]
    y_train = train["y"]
    x_test = test[[f"lag_{i}" for i in range(1, lags + 1)]]
    y_test = test["y"]
    model = LinearRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return y_test.values, preds


def _evaluate_one_series(
    series: pd.Series,
    model_name: str,
    season_lag: int,
    ma_window: int,
    lag_features: int,
) -> dict[str, float]:
    split_idx = _split_index(series)

    if model_name == "naive_last":
        y_true, y_pred = _predict_naive(series, split_idx)
    elif model_name == "seasonal_naive":
        y_true, y_pred = _predict_seasonal_naive(series, split_idx, season_lag=season_lag)
    elif model_name == "moving_average":
        y_true, y_pred = _predict_moving_average(series, split_idx, window=ma_window)
    elif model_name == "linear_lags":
        y_true, y_pred = _predict_linear_lags(series, split_idx, lags=lag_features)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    if len(y_true) == 0:
        raise ValueError("No evaluation samples after feature/lag filtering.")

    metrics = _evaluate_predictions(y_true, y_pred)
    metrics["n_test"] = int(len(y_true))
    return metrics


def _aggregate_metrics(metrics_rows: list[dict[str, float]]) -> dict[str, float]:
    frame = pd.DataFrame(metrics_rows)
    weights = frame["n_test"].astype(float).values
    out = {}
    for col in ["mae", "rmse", "mape", "smape"]:
        values = frame[col].astype(float).values
        out[col] = float(np.average(values, weights=weights))
    out["series_evaluated"] = int(len(frame))
    out["n_test_total"] = int(frame["n_test"].sum())
    return out


def _run_level(
    level: str,
    series_map: dict[str, pd.Series],
    snapshot_key: str,
    season_lag: int,
    ma_window: int,
    lag_features: int,
) -> None:
    models: list[str] = ["naive_last", "seasonal_naive", "moving_average", "linear_lags"]

    for model_name in models:
        metrics_rows: list[dict[str, float]] = []
        errors: list[dict[str, str]] = []

        with mlflow.start_run(run_name=f"level={level}__model={model_name}", nested=True):
            mlflow.log_params(
                {
                    "level": level,
                    "model_name": model_name,
                    "season_lag": season_lag,
                    "moving_average_window": ma_window,
                    "lag_features": lag_features,
                    "snapshot_key": snapshot_key,
                }
            )

            for series_id, series in series_map.items():
                if len(series) < 10:
                    errors.append({"series_id": series_id, "error": "too_short"})
                    continue
                try:
                    metric_row = _evaluate_one_series(
                        series=series,
                        model_name=model_name,
                        season_lag=season_lag,
                        ma_window=ma_window,
                        lag_features=lag_features,
                    )
                    metric_row["series_id"] = series_id
                    metrics_rows.append(metric_row)
                except Exception as exc:  # noqa: BLE001
                    errors.append({"series_id": series_id, "error": str(exc)})

            if not metrics_rows:
                mlflow.set_tag("status", "no_valid_series")
                if errors:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
                        pd.DataFrame(errors).to_csv(tmp.name, index=False)
                        mlflow.log_artifact(tmp.name, artifact_path="errors")
                continue

            aggregated = _aggregate_metrics(metrics_rows)
            mlflow.log_metrics(aggregated)

            metrics_df = pd.DataFrame(metrics_rows).sort_values("series_id")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_metrics:
                metrics_df.to_csv(tmp_metrics.name, index=False)
                mlflow.log_artifact(tmp_metrics.name, artifact_path="per_series_metrics")

            if errors:
                errors_df = pd.DataFrame(errors).sort_values("series_id")
                with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp_errors:
                    errors_df.to_csv(tmp_errors.name, index=False)
                    mlflow.log_artifact(tmp_errors.name, artifact_path="errors")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train baseline forecasting models and log in MLflow.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3/MinIO bucket with snapshots.")
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help="Base prefix where exported MySQL views are stored.",
    )
    parser.add_argument(
        "--level",
        choices=["total", "carrera", "both"],
        default="both",
        help="Which forecasting level to train.",
    )
    parser.add_argument(
        "--experiment-name",
        default="enrollment_forecasting_baselines",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--season-lag",
        type=int,
        default=2,
        help="Lag for seasonal naive baseline.",
    )
    parser.add_argument(
        "--moving-average-window",
        type=int,
        default=3,
        help="Window size for moving average baseline.",
    )
    parser.add_argument(
        "--lag-features",
        type=int,
        default=4,
        help="Number of lag features for linear model.",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    s3 = _get_s3_client()

    total_key = None
    carrera_key = None

    if args.level in {"total", "both"}:
        total_prefix = f"{args.prefix}/vw_dm_matricula_total/"
        total_key = _latest_snapshot_key(_list_snapshot_keys(s3, args.bucket, total_prefix))
        total_df = _normalize_columns(_read_csv_from_s3(s3, args.bucket, total_key))
        total_target = _infer_target_column(total_df, excluded={"periodo", "anio", "year", "id_cuatrimestre", "cuatrimestre", "term", "semester"})
        total_series_map = _build_series(total_df, target_col=total_target)

    if args.level in {"carrera", "both"}:
        carrera_prefix = f"{args.prefix}/vw_dm_matricula_carrera/"
        carrera_key = _latest_snapshot_key(_list_snapshot_keys(s3, args.bucket, carrera_prefix))
        carrera_df = _normalize_columns(_read_csv_from_s3(s3, args.bucket, carrera_key))
        if "id_carrera" in carrera_df.columns:
            career_col = "id_carrera"
        elif "carrera_id" in carrera_df.columns:
            career_col = "carrera_id"
        else:
            raise ValueError("Missing career identifier column (expected id_carrera or carrera_id).")
        carrera_target = _infer_target_column(
            carrera_df,
            excluded={
                "periodo",
                "anio",
                "year",
                "id_cuatrimestre",
                "cuatrimestre",
                "term",
                "semester",
                career_col,
            },
        )
        carrera_series_map = _build_series(carrera_df, target_col=carrera_target, career_col=career_col)

    with mlflow.start_run(run_name="baseline_training_parent"):
        mlflow.log_params(
            {
                "bucket": args.bucket,
                "prefix": args.prefix,
                "level": args.level,
                "tracking_uri": args.tracking_uri,
            }
        )
        if total_key:
            mlflow.log_param("total_snapshot_key", total_key)
        if carrera_key:
            mlflow.log_param("carrera_snapshot_key", carrera_key)

        if args.level in {"total", "both"}:
            _run_level(
                level="total",
                series_map=total_series_map,
                snapshot_key=total_key,
                season_lag=args.season_lag,
                ma_window=args.moving_average_window,
                lag_features=args.lag_features,
            )
        if args.level in {"carrera", "both"}:
            _run_level(
                level="carrera",
                series_map=carrera_series_map,
                snapshot_key=carrera_key,
                season_lag=args.season_lag,
                ma_window=args.moving_average_window,
                lag_features=args.lag_features,
            )

    print("Baseline training finished and logged to MLflow.")


if __name__ == "__main__":
    main()
