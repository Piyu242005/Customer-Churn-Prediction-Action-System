"""
Data Drift Monitoring using Evidently AI
Compares inference-time feature distributions against training-time distributions.

This module is designed to be lightweight and optional so it does not
interfere with the existing training/evaluation pipeline.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


class DataDriftMonitor:
    """
    Utility class to run data drift checks between training and inference data.

    Typical usage:
        monitor = DataDriftMonitor.from_artifacts(
            artifacts_dir="outputs/run_20260315_153552/models"
        )
        report, summary = monitor.run(
            current_data=current_df,
            save_html_path="outputs/drift/drift_report.html"
        )
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        self.reference_data = reference_data.copy()
        self.feature_names = feature_names or list(reference_data.columns)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: str,
        reference_data: Optional[pd.DataFrame] = None,
    ) -> "DataDriftMonitor":
        """
        Build a monitor from saved preprocessing artifacts.

        Args:
            artifacts_dir: Directory containing `scaler.pkl` and `feature_names.pkl`.
            reference_data: Optional pre-loaded reference feature matrix
                            as a pandas DataFrame. If not provided, this
                            method will attempt to load a saved NumPy
                            reference matrix `reference_X.npy` from
                            the same directory.
        """
        scaler_path = os.path.join(artifacts_dir, "scaler.pkl")
        feature_names_path = os.path.join(artifacts_dir, "feature_names.pkl")

        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(
                f"feature_names.pkl not found in {artifacts_dir}. "
                "Drift monitoring expects saved feature names."
            )

        feature_names = joblib.load(feature_names_path)

        if reference_data is None:
            # Optional, only if user decides to persist training matrix.
            ref_np_path = os.path.join(artifacts_dir, "reference_X.npy")
            if not os.path.exists(ref_np_path):
                raise FileNotFoundError(
                    "No reference_data provided and "
                    f"`{ref_np_path}` not found. "
                    "Pass a reference DataFrame explicitly when creating the monitor."
                )
            X_ref = np.load(ref_np_path)
            reference_data = pd.DataFrame(X_ref, columns=feature_names)

        return cls(reference_data=reference_data, feature_names=feature_names)

    # ------------------------------------------------------------------
    # Drift computation
    # ------------------------------------------------------------------
    def run(
        self,
        current_data: pd.DataFrame,
        save_html_path: Optional[str] = None,
    ) -> Tuple[Report, dict]:
        """
        Run data drift report between reference and current data.

        Args:
            current_data: Inference-time features as a pandas DataFrame.
                          Column order/names should match training features.
            save_html_path: Optional path to save an HTML report.

        Returns:
            (evidently Report instance, summary dict with high-level drift info)
        """
        # Align columns and ensure same ordering
        current_aligned = current_data[self.feature_names].copy()

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self.reference_data, current_data=current_aligned)

        result = report.as_dict()
        # Extract a compact summary for logging/alerting
        data_drift = result.get("metrics", [])[0].get("result", {}).get(
            "dataset_drift", False
        )
        drift_share = result.get("metrics", [])[0].get("result", {}).get(
            "share_drifted_columns", 0.0
        )

        summary = {
            "dataset_drift": bool(data_drift),
            "share_drifted_columns": float(drift_share),
        }

        if save_html_path:
            os.makedirs(os.path.dirname(save_html_path), exist_ok=True)
            report.save_html(save_html_path)

        return report, summary


__all__ = ["DataDriftMonitor"]


