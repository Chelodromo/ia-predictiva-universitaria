"""Extract enrollment data views from a remote MySQL and upload CSV snapshots to MinIO."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

import boto3
import pymysql
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.hooks.base import BaseHook
from airflow.models import Variable
from airflow.operators.python import PythonOperator

DAG_ID = "extract_mysql_views_to_minio"
MYSQL_CONN_ID = "mysql_unsta"
DEFAULT_SCHEMA = "eUNSTAv3"
VIEW_NAMES = [
    "vw_dm_matricula_total",
    "vw_dm_matricula_carrera",
]


def _build_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def _get_mysql_connection(conn_id: str, schema: str):
    conn = BaseHook.get_connection(conn_id)
    extra = conn.extra_dejson or {}

    return pymysql.connect(
        host=conn.host,
        port=conn.port or 3306,
        user=conn.login,
        password=conn.password,
        database=schema or conn.schema,
        charset=extra.get("charset", "utf8mb4"),
        connect_timeout=30,
    )


def _extract_and_upload_views(**context):
    run_dt = context.get("data_interval_end") or context["logical_date"]
    run_ts = run_dt.strftime("%Y%m%dT%H%M%S")

    schema = Variable.get("mysql_source_schema", default_var=DEFAULT_SCHEMA)
    conn_id = Variable.get("mysql_source_conn_id", default_var=MYSQL_CONN_ID)
    bucket = Variable.get("data_repo_bucket", default_var=os.getenv("DATA_REPO_BUCKET_NAME", "data"))
    prefix = Variable.get("mysql_export_prefix", default_var="mysql_exports/unsta")

    mysql_conn = _get_mysql_connection(conn_id=conn_id, schema=schema)
    s3 = _build_s3_client()

    exported_files = []

    try:
        for view_name in VIEW_NAMES:
            query = f"SELECT * FROM {schema}.{view_name}"
            with mysql_conn.cursor() as cursor:
                cursor.execute(query)
                headers = [col[0] for col in cursor.description]
                rows = cursor.fetchall()

            local_path = Path(f"/tmp/{view_name}_{run_ts}.csv")
            with local_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(headers)
                writer.writerows(rows)

            s3_key = f"{prefix}/{view_name}/snapshot_{run_ts}.csv"
            s3.upload_file(str(local_path), bucket, s3_key)
            exported_files.append((view_name, len(rows), s3_key))

    except pymysql.MySQLError as exc:
        raise AirflowFailException(f"MySQL extraction failed: {exc}") from exc
    finally:
        mysql_conn.close()

    if not exported_files:
        raise AirflowFailException("No files were exported.")

    for view_name, row_count, s3_key in exported_files:
        print(f"Exported {view_name}: {row_count} rows -> s3://{bucket}/{s3_key}")


with DAG(
    dag_id=DAG_ID,
    start_date=datetime(2026, 1, 1),
    schedule="0 3 * * *",
    catchup=False,
    tags=["etl", "mysql", "minio", "enrollment"],
) as dag:
    export_mysql_views = PythonOperator(
        task_id="export_mysql_views_to_minio",
        python_callable=_extract_and_upload_views,
    )
