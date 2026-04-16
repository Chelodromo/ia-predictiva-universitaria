from __future__ import annotations

import argparse
import math
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


class TableParser(HTMLParser):
    """Minimal parser for the SIS export saved as .xls but containing HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[dict[str, str]]] = []
        self._current_row: list[dict[str, str]] | None = None
        self._current_cell: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag == "td":
            attr_dict = {k: v or "" for k, v in attrs}
            self._current_cell = {"text": "", "style": attr_dict.get("style", "")}

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell["text"] += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._current_cell is not None:
            if self._current_row is not None:
                self._current_row.append(self._current_cell)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None:
            self.rows.append(self._current_row)
            self._current_row = None


def parse_money(value: Any) -> float:
    text = str(value).strip()
    if not text or text == "-":
        return math.nan

    text = re.sub(r"[^\d,.-]", "", text)
    if "," in text:
        # Argentine format: 250.990,00 -> 250990.00
        text = text.replace(".", "").replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return math.nan


def is_red_style(style: str) -> bool:
    return "color" in style.lower() and "red" in style.lower()


def parse_quota_cell(cell: dict[str, str]) -> tuple[float, float]:
    text = cell["text"].strip()
    red = is_red_style(cell["style"])

    if not text:
        return math.nan, math.nan

    if red:
        return 0.0, parse_money(text)

    if text == "-":
        return math.nan, math.nan

    return 1.0, parse_money(text)


def parse_deudores_file(input_path: Path, anio_reporte: int) -> pd.DataFrame:
    parser = TableParser()
    parser.feed(input_path.read_text(encoding="utf-8"))

    if not parser.rows:
        raise ValueError(f"No se encontraron filas en {input_path}")

    headers = [cell["text"].strip() for cell in parser.rows[0]]
    records: list[dict[str, Any]] = []

    for row in parser.rows[1:]:
        if not any(cell["text"].strip() for cell in row):
            continue

        raw = {
            header: row[i]["text"].strip() if i < len(row) else ""
            for i, header in enumerate(headers)
        }
        if not raw.get("Legajo") or not raw.get("Carrera"):
            continue

        record: dict[str, Any] = {
            "Periodo Ingreso": pd.to_numeric(raw.get("Periodo Ingreso"), errors="coerce"),
            "Legajo": raw.get("Legajo", ""),
            "Alumno": raw.get("Alumno", ""),
            "Documento": pd.to_numeric(raw.get("Documento"), errors="coerce"),
            "Carrera": raw.get("Carrera", ""),
        }

        pagos: list[float] = []
        montos: list[float] = []
        for cuota in range(1, 13):
            idx = headers.index(f"Cuota_{cuota}")
            cell = row[idx] if idx < len(row) else {"text": "", "style": ""}
            pago, monto = parse_quota_cell(cell)
            record[f"pago_cuota_{cuota}"] = pago
            record[f"monto_cuota_{cuota}"] = monto
            pagos.append(pago)
            montos.append(monto)

        for col in ["Categoria", "1° Ingreso EaD", "Ult. Ingreso EaD"]:
            record[col] = raw.get(col, "")

        record["Monto Deuda"] = parse_money(raw.get("Monto Deuda", ""))
        record["n_generadas"] = sum(not math.isnan(pago) for pago in pagos)
        record["n_pagadas"] = sum(pago == 1 for pago in pagos)
        record["n_impagas"] = sum(pago == 0 for pago in pagos)
        record["ratio_pago"] = (
            record["n_pagadas"] / record["n_generadas"]
            if record["n_generadas"]
            else math.nan
        )
        record["deuda_calc"] = sum(monto for monto in montos if not math.isnan(monto))
        record["deuda_sistema"] = record["Monto Deuda"]
        record["anio_reporte"] = anio_reporte

        records.append(record)

    return pd.DataFrame(records)


def generate(input_path: Path, output_path: Path, anio_reporte: int) -> None:
    df = parse_deudores_file(input_path, anio_reporte)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"{output_path}: {len(df):,} filas")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera los CSV deudores_YYYY_parsed.csv desde los .xls HTML."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Carpeta donde estan los .xls y donde se escriben los .csv.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    jobs = [
        (
            data_dir / "2024_sede_0_unidad_0_oferta_0_deuda_.xls",
            data_dir / "deudores_2024_parsed.csv",
            2024,
        ),
        (
            data_dir / "2025_sede_0_unidad_0_oferta_0_deuda_.xls",
            data_dir / "deudores_2025_parsed.csv",
            2025,
        ),
        (
            data_dir / "2026_sede_0_unidad_0_oferta_0_deuda_.xls",
            data_dir / "deudores_2026_parsed.csv",
            2026,
        ),
    ]

    for input_path, output_path, anio in jobs:
        if not input_path.is_file():
            raise FileNotFoundError(f"No existe {input_path}")
        generate(input_path, output_path, anio)


if __name__ == "__main__":
    main()
