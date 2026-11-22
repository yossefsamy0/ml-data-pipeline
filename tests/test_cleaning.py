"""
Tests for the cleaning pipeline.

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cleaning import clean_orders, normalise_text, parse_dates, CleaningReport  # noqa: E402


def make_frame(**overrides) -> pd.DataFrame:
    """One valid row, with any column overridable per test."""
    row = {
        "order_id": 1,
        "order_date": "2026-01-05",
        "customer": "Ahmed Ali",
        "category": "Electronics",
        "quantity": 2,
        "unit_price": 100.0,
        "country": "EG",
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestDuplicates(unittest.TestCase):
    def test_exact_duplicates_are_removed(self):
        df = pd.concat([make_frame(), make_frame()], ignore_index=True)
        clean, report = clean_orders(df)
        self.assertEqual(report.rows_in, 2)
        self.assertEqual(report.rows_out, 1)
        self.assertEqual(report.duplicate_rows_removed, 1)


class TestTextNormalisation(unittest.TestCase):
    def test_whitespace_and_case_are_normalised(self):
        df = make_frame(category="  eLeCtRoNiCs  ", customer="  Dina   Samir ")
        clean, _ = normalise_text(df)
        self.assertEqual(clean.loc[0, "category"], "Electronics")
        self.assertEqual(clean.loc[0, "customer"], "Dina Samir")

    def test_country_is_uppercased(self):
        df = make_frame(country="eg")
        clean, _ = normalise_text(df)
        self.assertEqual(clean.loc[0, "country"], "EG")


class TestMissingValues(unittest.TestCase):
    def test_missing_customer_becomes_unknown(self):
        df = make_frame(customer=None)
        clean, report = clean_orders(df)
        self.assertEqual(clean.loc[0, "customer"], "Unknown")
        self.assertEqual(report.missing_customer_filled, 1)

    def test_row_with_missing_price_is_dropped_and_counted(self):
        df = make_frame(unit_price=None)
        clean, report = clean_orders(df)
        self.assertEqual(len(clean), 0)
        self.assertEqual(report.missing_price_rows_dropped, 1)

    def test_missing_quantity_uses_median_of_valid_rows(self):
        df = pd.concat(
            [
                make_frame(order_id=1, quantity=2),
                make_frame(order_id=2, quantity=4),
                make_frame(order_id=3, quantity=None),
            ],
            ignore_index=True,
        )
        clean, report = clean_orders(df)
        self.assertEqual(report.missing_quantity_imputed, 1)
        self.assertEqual(clean.loc[clean["order_id"] == 3, "quantity"].iloc[0], 3.0)


class TestInvalidValues(unittest.TestCase):
    def test_negative_quantity_is_converted_to_absolute(self):
        df = make_frame(quantity=-3)
        clean, report = clean_orders(df)
        self.assertEqual(clean.loc[0, "quantity"], 3.0)
        self.assertEqual(report.negative_quantity_fixed, 1)

    def test_zero_price_is_flagged_and_excluded_from_revenue(self):
        df = make_frame(unit_price=0.0)
        clean, report = clean_orders(df)
        self.assertEqual(report.nonpositive_price_flagged, 1)
        self.assertIn("invalid_price", clean.loc[0, "flags"])
        self.assertTrue(pd.isna(clean.loc[0, "revenue"]))

    def test_valid_row_gets_revenue(self):
        clean, _ = clean_orders(make_frame(quantity=3, unit_price=10.0))
        self.assertEqual(clean.loc[0, "revenue"], 30.0)


class TestDates(unittest.TestCase):
    def test_iso_date_is_parsed(self):
        report = CleaningReport()
        parsed = parse_dates(pd.Series(["2026-01-05"]), report)
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2026-01-05"))

    def test_ambiguous_date_is_day_first(self):
        report = CleaningReport()
        parsed = parse_dates(pd.Series(["05/01/2026"]), report)
        self.assertEqual(parsed.iloc[0], pd.Timestamp("2026-01-05"))

    def test_unparseable_date_is_counted_not_dropped(self):
        df = make_frame(order_date="not a date")
        clean, report = clean_orders(df)
        self.assertEqual(report.dates_unparsed, 1)
        self.assertEqual(len(clean), 1)
        self.assertIn("unparsed_date", clean.loc[0, "flags"])

    def test_future_date_is_flagged_not_removed(self):
        df = make_frame(order_date="2030-01-01")
        clean, report = clean_orders(df, today="2026-11-22")
        self.assertEqual(report.future_dates_flagged, 1)
        self.assertEqual(len(clean), 1)
        self.assertIn("future_date", clean.loc[0, "flags"])


class TestEndToEnd(unittest.TestCase):
    def test_sample_file_runs(self):
        path = Path(__file__).resolve().parent.parent / "data" / "messy_orders.csv"
        df = pd.read_csv(path)
        clean, report = clean_orders(df)
        self.assertEqual(report.rows_in, 17)
        self.assertEqual(report.rows_out, 14)
        self.assertFalse(clean["customer"].isna().any())
        self.assertEqual(clean["category"].nunique(), 3)


if __name__ == "__main__":
    unittest.main()
