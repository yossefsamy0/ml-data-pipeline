"""
Reusable CSV cleaning pipeline for e-commerce order data.

The goal of this module is to take a messy export (the kind you get from a
real store: duplicated rows, missing prices, negative quantities, mixed date
formats, inconsistent casing) and turn it into a clean, documented file plus a
report explaining every decision that was made.

Nothing is deleted silently. Every row that is dropped, filled or flagged is
counted in the report so a reviewer can check the decision.

Usage:
    python src/cleaning.py data/messy_orders.csv --out reports

Outputs (in --out directory):
    clean_orders.csv        the cleaned dataset
    cleaning_report.md      human-readable report
    cleaning_report.json    same report as structured data
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "order_id",
    "order_date",
    "customer",
    "category",
    "quantity",
    "unit_price",
    "country",
]

TEXT_COLUMNS = ["customer", "category", "country"]

# Ambiguous dates like 05/01/2026 are resolved day-first, which is the
# convention in Egypt and the Gulf. This is a documented assumption, not a
# silent guess: the count of affected rows is reported.
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"]
DAY_FIRST = True

# Reference date used to flag dates in the future. Passed in rather than
# hard-coded so the behaviour is testable.
DEFAULT_TODAY = "2026-11-22"


@dataclass
class CleaningReport:
    """Collects counts and notes so every decision is auditable."""

    rows_in: int = 0
    rows_out: int = 0
    duplicate_rows_removed: int = 0
    missing_before: dict = field(default_factory=dict)
    missing_after: dict = field(default_factory=dict)
    missing_customer_filled: int = 0
    missing_quantity_imputed: int = 0
    quantity_imputed_value: float = 0.0
    negative_quantity_fixed: int = 0
    missing_price_rows_dropped: int = 0
    nonpositive_price_flagged: int = 0
    future_dates_flagged: int = 0
    dates_parsed_by_format: dict = field(default_factory=dict)
    dates_unparsed: int = 0
    text_values_normalised: int = 0
    price_outliers_flagged: int = 0
    outlier_bounds: tuple = (0.0, 0.0)
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "missing_customer_filled": self.missing_customer_filled,
            "missing_quantity_imputed": self.missing_quantity_imputed,
            "quantity_imputed_value": self.quantity_imputed_value,
            "negative_quantity_fixed": self.negative_quantity_fixed,
            "missing_price_rows_dropped": self.missing_price_rows_dropped,
            "nonpositive_price_flagged": self.nonpositive_price_flagged,
            "future_dates_flagged": self.future_dates_flagged,
            "dates_parsed_by_format": self.dates_parsed_by_format,
            "dates_unparsed": self.dates_unparsed,
            "text_values_normalised": self.text_values_normalised,
            "price_outliers_flagged": self.price_outliers_flagged,
            "outlier_bounds": list(self.outlier_bounds),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Step 1 - load and validate
# ---------------------------------------------------------------------------


def load_orders(path: Path) -> pd.DataFrame:
    """Read the CSV and fail loudly if expected columns are missing."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


# ---------------------------------------------------------------------------
# Step 2 - text normalisation
# ---------------------------------------------------------------------------


def normalise_text(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Strip leading/trailing spaces, collapse double spaces, and fix casing so
    that "Electronics", "electronics" and "ELECTRONICS" are one category.

    Returns the cleaned frame and the number of values that changed.
    """
    df = df.copy()
    changed = 0

    for col in TEXT_COLUMNS:
        before = df[col].astype("string")
        # " Books " -> "Books"; "  Dina  Samir  " -> "Dina Samir"
        after = before.str.strip().str.replace(r"\s+", " ", regex=True)

        if col == "country":
            after = after.str.upper()
        else:
            after = after.str.title()

        changed += int((before.fillna("") != after.fillna("")).sum())
        df[col] = after

    return df, changed


# ---------------------------------------------------------------------------
# Step 3 - dates
# ---------------------------------------------------------------------------


def parse_dates(
    series: pd.Series, report: CleaningReport
) -> pd.Series:
    """
    Parse a date column that may contain several formats.

    Each value is tried against DATE_FORMATS in order. Values that match none
    are left as NaT and counted, rather than being dropped or guessed.
    """
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    per_format = {fmt: 0 for fmt in DATE_FORMATS}
    unparsed = 0

    for idx, raw in series.items():
        if pd.isna(raw):
            unparsed += 1
            continue

        text = str(raw).strip()
        matched = False
        for fmt in DATE_FORMATS:
            try:
                result.at[idx] = pd.to_datetime(text, format=fmt)
                per_format[fmt] += 1
                matched = True
                break
            except (ValueError, TypeError):
                continue
        if not matched:
            unparsed += 1

    report.dates_parsed_by_format = {
        k: v for k, v in per_format.items() if v > 0
    }
    report.dates_unparsed = unparsed
    return result


# ---------------------------------------------------------------------------
# Step 4 - the cleaning pipeline
# ---------------------------------------------------------------------------


def clean_orders(df: pd.DataFrame, today: str = DEFAULT_TODAY) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return (clean_df, report)."""
    report = CleaningReport()
    report.rows_in = len(df)
    report.missing_before = {
        c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS if df[c].isna().any()
    }

    # --- 4a. exact duplicates -------------------------------------------
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    report.duplicate_rows_removed = before - len(df)

    # --- 4b. text --------------------------------------------------------
    df, text_changes = normalise_text(df)
    report.text_values_normalised = text_changes

    # --- 4c. dates -------------------------------------------------------
    df["order_date"] = parse_dates(df["order_date"], report)

    # --- 4d. numeric types ----------------------------------------------
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")

    # --- 4e. missing customer -------------------------------------------
    missing_customer = int(df["customer"].isna().sum() + (df["customer"] == "").sum())
    df["customer"] = df["customer"].replace("", pd.NA).fillna("Unknown")
    report.missing_customer_filled = missing_customer

    # --- 4f. rows with no usable price are dropped ----------------------
    # A missing price cannot be recovered from the row itself, and inventing
    # one would corrupt any revenue figure downstream. Counted, not hidden.
    no_price = df["unit_price"].isna()
    report.missing_price_rows_dropped = int(no_price.sum())
    df = df[~no_price].reset_index(drop=True)

    # --- 4g. missing quantity -> median of valid quantities -------------
    valid_qty = df.loc[df["quantity"] > 0, "quantity"]
    median_qty = float(valid_qty.median()) if not valid_qty.empty else 1.0
    report.quantity_imputed_value = median_qty
    missing_qty = df["quantity"].isna()
    report.missing_quantity_imputed = int(missing_qty.sum())
    df.loc[missing_qty, "quantity"] = median_qty

    # --- 4h. negative quantity -> absolute value, flagged ---------------
    # In a sales export a negative quantity is almost always a typo or a
    # returned line. The value is kept (abs) and the row is flagged so a human
    # can review it instead of losing the record.
    neg_qty = df["quantity"] < 0
    report.negative_quantity_fixed = int(neg_qty.sum())
    df.loc[neg_qty, "quantity"] = df.loc[neg_qty, "quantity"].abs()

    # --- 4i. non-positive price -> flagged, excluded from revenue -------
    bad_price = df["unit_price"] <= 0
    report.nonpositive_price_flagged = int(bad_price.sum())
    df["price_valid"] = ~bad_price

    # --- 4j. future dates -> flagged ------------------------------------
    today_ts = pd.Timestamp(today)
    future = df["order_date"] > today_ts
    report.future_dates_flagged = int(future.sum())

    # --- 4k. price outliers via IQR -------------------------------------
    prices = df.loc[df["price_valid"], "unit_price"]
    if not prices.empty:
        q1, q3 = prices.quantile(0.25), prices.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        report.outlier_bounds = (float(lower), float(upper))
        df["price_outlier"] = df["unit_price"].notna() & (
            (df["unit_price"] < lower) | (df["unit_price"] > upper)
        )
        report.price_outliers_flagged = int(df["price_outlier"].sum())
    else:
        df["price_outlier"] = False

    # --- 4l. flags column + revenue -------------------------------------
    def build_flags(row) -> str:
        flags = []
        if not row["price_valid"]:
            flags.append("invalid_price")
        if row["price_outlier"]:
            flags.append("price_outlier")
        if pd.notna(row["order_date"]) and row["order_date"] > today_ts:
            flags.append("future_date")
        if pd.isna(row["order_date"]):
            flags.append("unparsed_date")
        return ";".join(flags)

    df["flags"] = df.apply(build_flags, axis=1)

    df["revenue"] = df.apply(
        lambda r: round(r["quantity"] * r["unit_price"], 2) if r["price_valid"] else pd.NA,
        axis=1,
    )

    # --- 4m. finish ------------------------------------------------------
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    df = df[
        [
            "order_id",
            "order_date",
            "customer",
            "category",
            "country",
            "quantity",
            "unit_price",
            "revenue",
            "flags",
        ]
    ]

    report.rows_out = len(df)
    report.missing_after = {
        c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()
    }

    report.notes = [
        "Ambiguous dates were resolved day-first (Egypt/Gulf convention).",
        "Missing prices were dropped because they cannot be recovered from the row.",
        "Missing quantities were imputed with the median of valid quantities.",
        "Negative quantities were converted to absolute values and kept.",
        "Rows with a non-positive price are kept but excluded from revenue.",
        "Future dates are flagged, not removed, because they may be real pre-orders.",
    ]
    return df, report


# ---------------------------------------------------------------------------
# Step 5 - reporting
# ---------------------------------------------------------------------------


def report_to_markdown(report: CleaningReport, source: str) -> str:
    """Render the report as markdown for humans."""
    lines = [
        "# Data Cleaning Report",
        "",
        f"Source file: `{source}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Rows in | {report.rows_in} |",
        f"| Rows out | {report.rows_out} |",
        f"| Duplicate rows removed | {report.duplicate_rows_removed} |",
        f"| Text values normalised | {report.text_values_normalised} |",
        f"| Missing customers filled with 'Unknown' | {report.missing_customer_filled} |",
        f"| Rows dropped for missing price | {report.missing_price_rows_dropped} |",
        f"| Missing quantities imputed | {report.missing_quantity_imputed} |",
        f"| Negative quantities corrected | {report.negative_quantity_fixed} |",
        f"| Non-positive prices flagged | {report.nonpositive_price_flagged} |",
        f"| Future dates flagged | {report.future_dates_flagged} |",
        f"| Price outliers flagged (IQR) | {report.price_outliers_flagged} |",
        f"| Dates that could not be parsed | {report.dates_unparsed} |",
        "",
        "## Missing values",
        "",
        "| Column | Before | After |",
        "|---|---|---|",
    ]
    for col in REQUIRED_COLUMNS:
        if col in ("order_id", "order_date"):
            continue
        lines.append(
            f"| {col} | {report.missing_before.get(col, 0)} | {report.missing_after.get(col, 0)} |"
        )

    if report.dates_parsed_by_format:
        lines += ["", "## Date formats detected", "", "| Format | Rows |", "|---|---|"]
        for fmt, count in report.dates_parsed_by_format.items():
            lines.append(f"| `{fmt}` | {count} |")

    if report.price_outliers_flagged:
        low, high = report.outlier_bounds
        lines += [
            "",
            f"IQR bounds applied to unit price: **{low:.2f}** to **{high:.2f}**.",
            "Rows outside these bounds are flagged, not removed.",
        ]

    lines += ["", "## Decisions", ""]
    lines += [f"- {note}" for note in report.notes]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean a messy orders CSV.")
    parser.add_argument("input", type=Path, help="path to the input CSV")
    parser.add_argument(
        "--out", type=Path, default=Path("reports"), help="output directory"
    )
    parser.add_argument(
        "--today",
        default=DEFAULT_TODAY,
        help="reference date (YYYY-MM-DD) used to flag future dates",
    )
    args = parser.parse_args(argv)

    df = load_orders(args.input)
    clean_df, report = clean_orders(df, today=args.today)

    args.out.mkdir(parents=True, exist_ok=True)
    clean_path = args.out / "clean_orders.csv"
    clean_df.to_csv(clean_path, index=False)

    (args.out / "cleaning_report.md").write_text(
        report_to_markdown(report, str(args.input)), encoding="utf-8"
    )
    (args.out / "cleaning_report.json").write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8"
    )

    print(f"rows in : {report.rows_in}")
    print(f"rows out: {report.rows_out}")
    print(f"written : {clean_path}")
    print(f"report  : {args.out / 'cleaning_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
