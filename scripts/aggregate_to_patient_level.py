#!/usr/bin/env python3
"""
Aggregate visit-level patient data to patient-level profiles.

This script takes the visit-level data (patients_processed.csv) and aggregates
it to create unique patient-level profiles suitable for PSN construction.

Input:  data/processed/patients_processed.csv (135,096 visits)
Output: data/processed/patients_patient_level.csv (84,665 patients)

Aggregation Strategy (First Encounter):
- Encounter-specific variables (AGE, LENSTAYD, ZIPCODE, PAYER): first encounter
  → Ensures AGE and LENSTAYD are from the same encounter (no temporal mixing)
  → Represents baseline health status at initial presentation
- Demographics (SEX, Race): mode (should be constant across visits)
- Risk factors: max (ever diagnosed = 1 across any visit)
- Readmission: READMIT_COUNT = sum(REVISIT_30),
               READMIT_RATE = READMIT_COUNT / max(NUM_VISITS - 1, 1)

Usage:
    python scripts/aggregate_to_patient_level.py
    python scripts/aggregate_to_patient_level.py --input path/to/visits.csv --output path/to/patients.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def safe_mode(series):
    """Return mode of series, or first value if mode is empty."""
    mode_vals = series.mode()
    if len(mode_vals) > 0:
        return mode_vals.iloc[0]
    return series.iloc[0] if len(series) > 0 else None


def aggregate_to_patient_level(visits_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Aggregate visit-level data to patient-level profiles using first encounter.

    Encounter-specific variables (AGE, LENSTAYD, ZIPCODE, PAYER) are taken from
    the first encounter (lowest SEQ_NO) so that AGE and LENSTAYD are always from
    the same visit. Risk factor conditions are aggregated as 'ever diagnosed' (max).

    Args:
        visits_df: DataFrame with visit-level data (one row per hospital visit)
        verbose: Print progress messages

    Returns:
        DataFrame with patient-level data (one row per unique patient)
    """
    if verbose:
        print("=" * 80)
        print("AGGREGATING VISIT-LEVEL DATA TO PATIENT-LEVEL (First Encounter)")
        print("=" * 80)
        print(f"\nInput: {len(visits_df):,} visits from {visits_df['PATIENTID'].nunique():,} patients")

    # Sort so that 'first' always picks the earliest encounter
    visits_df = visits_df.sort_values(["PATIENTID", "SEQ_NO"]).reset_index(drop=True)

    # Define aggregation rules
    agg_rules = {
        # Encounter-specific: first encounter (AGE + LENSTAYD from same visit)
        "AGE":      "first",   # Age at first hospitalisation
        "LENSTAYD": "first",   # Length of stay at first hospitalisation
        "ZIPCODE":  "first",   # Residence at first hospitalisation
        "PAYER":    "first",   # Insurance at first hospitalisation

        # Demographics: should be constant; mode handles any inconsistencies
        "SEX":  safe_mode,
        "Race": safe_mode,

        # Risk factors: ever diagnosed across any visit
        "Hearingloss": "max",
        "BrainInjury": "max",
        "Hypertension": "max",
        "Alcohol":      "max",
        "Obesity":      "max",
        "Diabetes":     "max",

        # Readmission: total count of 30-day readmissions
        "REVISIT_30": "sum",
    }

    if verbose:
        print("\nApplying aggregation rules...")
        print("-" * 80)
        for col, rule in agg_rules.items():
            rule_str = rule if isinstance(rule, str) else rule.__name__
            print(f"  {col:20} → {rule_str}")

    # Group by patient and aggregate
    patient_df = visits_df.groupby("PATIENTID", as_index=False).agg(agg_rules)

    if verbose:
        print(f"\nAggregated to {len(patient_df):,} patient profiles")

    # ------------------------------------------------------------------
    # Add derived columns
    # ------------------------------------------------------------------
    if verbose:
        print("\nAdding derived columns...")
        print("-" * 80)

    # NUM_VISITS: total hospitalizations per patient
    num_visits = visits_df.groupby("PATIENTID").size().reset_index(name="NUM_VISITS")
    patient_df = patient_df.merge(num_visits, on="PATIENTID", how="left")
    if verbose:
        print(
            f"  NUM_VISITS:    Added "
            f"(mean={patient_df['NUM_VISITS'].mean():.2f}, "
            f"max={patient_df['NUM_VISITS'].max()})"
        )

    # READMIT_COUNT: total 30-day readmissions (already summed in REVISIT_30)
    patient_df = patient_df.rename(columns={"REVISIT_30": "READMIT_COUNT"})
    if verbose:
        print(
            f"  READMIT_COUNT: Added "
            f"(mean={patient_df['READMIT_COUNT'].mean():.2f}, "
            f"max={patient_df['READMIT_COUNT'].max()})"
        )

    # READMIT_RATE: readmissions as proportion of eligible encounters
    # (divide by NUM_VISITS - 1 because last visit cannot trigger a readmission;
    #  clip to 1 to avoid division by zero for single-visit patients)
    patient_df["READMIT_RATE"] = (
        patient_df["READMIT_COUNT"]
        / np.maximum(patient_df["NUM_VISITS"] - 1, 1)
    ).round(4)
    if verbose:
        print(f"  READMIT_RATE:  Added (mean={patient_df['READMIT_RATE'].mean():.3f})")

    # EVER_READMITTED: binary flag
    patient_df["EVER_READMITTED"] = (patient_df["READMIT_COUNT"] > 0).astype(int)
    if verbose:
        n_ever = patient_df["EVER_READMITTED"].sum()
        print(
            f"  EVER_READMITTED: Added "
            f"({n_ever:,} patients = {n_ever / len(patient_df):.1%})"
        )

    # ------------------------------------------------------------------
    # Derived continuous / categorical features
    # ------------------------------------------------------------------
    if verbose:
        print("\nRecalculating derived columns...")
        print("-" * 80)

    # LENSTAYD_LOG: log-transform of first-encounter LOS
    patient_df["LENSTAYD_LOG"] = np.log1p(patient_df["LENSTAYD"])
    if verbose:
        print("  LENSTAYD_LOG: Calculated from first-encounter LENSTAYD")

    # AGE_BIN: bins from first-encounter AGE
    age_bins   = [0, 65, 70, 75, 80, 85, 90, np.inf]
    age_labels = ["<65", "65-69", "70-74", "75-79", "80-84", "85-89", ">=90"]
    patient_df["AGE_BIN"] = pd.cut(
        patient_df["AGE"], bins=age_bins, labels=age_labels, right=True
    )
    if verbose:
        print("  AGE_BIN: Created from first-encounter AGE")

    # LENSTAYD_BIN: bins from first-encounter LOS
    los_bins   = [0, 3, 5, 10, 20, np.inf]
    los_labels = ["Short Stay", "Medium Stay", "Long Stay", "Extended Stay", "Very Long Stay"]
    patient_df["LENSTAYD_BIN"] = pd.cut(
        patient_df["LENSTAYD"], bins=los_bins, labels=los_labels, right=False
    )
    if verbose:
        print("  LENSTAYD_BIN: Created from first-encounter LENSTAYD")

    # ------------------------------------------------------------------
    # Reorder columns
    # ------------------------------------------------------------------
    column_order = [
        # Identifiers
        "PATIENTID",
        # Demographics
        "AGE", "SEX", "Race", "AGE_BIN",
        # Location
        "ZIPCODE",
        # Utilization (first encounter + overall visit count)
        "LENSTAYD", "LENSTAYD_LOG", "LENSTAYD_BIN", "PAYER", "NUM_VISITS",
        # Risk factors
        "Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes",
        # Outcomes
        "READMIT_COUNT", "READMIT_RATE", "EVER_READMITTED",
    ]

    patient_df = patient_df[column_order]

    if verbose:
        print(
            f"\nFinal patient-level data: "
            f"{len(patient_df):,} rows × {len(patient_df.columns)} columns"
        )

    return patient_df


def validate_patient_data(df: pd.DataFrame, verbose: bool = True) -> bool:
    """Validate patient-level data quality."""
    if verbose:
        print("\n" + "=" * 80)
        print("VALIDATION CHECKS")
        print("=" * 80)

    issues = []

    required_cols = [
        "PATIENTID", "AGE", "SEX", "Race", "AGE_BIN", "ZIPCODE",
        "LENSTAYD", "LENSTAYD_LOG", "LENSTAYD_BIN", "PAYER", "NUM_VISITS",
        "Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes",
        "READMIT_COUNT", "READMIT_RATE", "EVER_READMITTED",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    elif verbose:
        print("✓ All required columns present")

    n_duplicates = df["PATIENTID"].duplicated().sum()
    if n_duplicates > 0:
        issues.append(f"Found {n_duplicates} duplicate PATIENTIDs")
    elif verbose:
        print("✓ No duplicate PATIENTIDs")

    if "SEX" in df.columns:
        invalid_sex = set(df["SEX"].unique()) - {"F", "M"}
        if invalid_sex:
            issues.append(f"Invalid SEX values: {invalid_sex}")
        elif verbose:
            print("✓ SEX values valid (F/M only)")

    if "AGE_BIN" in df.columns:
        n_missing = df["AGE_BIN"].isna().sum()
        if n_missing > 0:
            issues.append(f"AGE_BIN has {n_missing} missing values")
        elif verbose:
            print("✓ AGE_BIN created for all rows")

    if "LENSTAYD_BIN" in df.columns:
        n_missing = df["LENSTAYD_BIN"].isna().sum()
        if n_missing > 0:
            issues.append(f"LENSTAYD_BIN has {n_missing} missing values")
        elif verbose:
            print("✓ LENSTAYD_BIN created for all rows")

    if "NUM_VISITS" in df.columns:
        if (df["NUM_VISITS"] < 1).any():
            issues.append("NUM_VISITS has values < 1")
        elif verbose:
            print(f"✓ NUM_VISITS valid (min=1, max={df['NUM_VISITS'].max()})")

    if "READMIT_COUNT" in df.columns:
        if (df["READMIT_COUNT"] < 0).any():
            issues.append("READMIT_COUNT has negative values")
        elif verbose:
            print("✓ READMIT_COUNT non-negative")

    if "READMIT_RATE" in df.columns:
        if (df["READMIT_RATE"] < 0).any():
            issues.append("READMIT_RATE has negative values")
        elif verbose:
            print("✓ READMIT_RATE non-negative")

    if "EVER_READMITTED" in df.columns:
        if not set(df["EVER_READMITTED"].unique()).issubset({0, 1}):
            issues.append("EVER_READMITTED has non-binary values")
        elif verbose:
            print("✓ EVER_READMITTED is binary (0/1)")

    if issues:
        if verbose:
            print("\n❌ VALIDATION FAILED:")
            for issue in issues:
                print(f"  • {issue}")
        return False

    if verbose:
        print("\n✅ All validation checks passed!")
    return True


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for patient-level data."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nTotal patients: {len(df):,}")

    print("\nVisit Distribution:")
    print(f"  Mean visits/patient:   {df['NUM_VISITS'].mean():.2f}")
    print(f"  Median visits/patient: {df['NUM_VISITS'].median():.0f}")
    print(f"  Max visits/patient:    {df['NUM_VISITS'].max()}")
    single = (df["NUM_VISITS"] == 1).sum()
    print(f"  Single-visit patients: {single:,} ({single / len(df):.1%})")
    multi  = (df["NUM_VISITS"] >= 2).sum()
    print(f"  Multi-visit patients:  {multi:,} ({multi / len(df):.1%})")

    print("\nDemographics (first encounter):")
    print(f"  Mean age at first encounter: {df['AGE'].mean():.1f} years")
    print(f"  Sex distribution: {df['SEX'].value_counts().to_dict()}")

    print("\nUtilization (first encounter):")
    print(f"  Mean LOS:   {df['LENSTAYD'].mean():.1f} days")
    print(f"  Median LOS: {df['LENSTAYD'].median():.1f} days")

    print("\nRisk Factor Prevalence (ever diagnosed):")
    for risk_factor in ["Hearingloss", "BrainInjury", "Hypertension", "Alcohol", "Obesity", "Diabetes"]:
        if risk_factor in df.columns:
            print(f"  {risk_factor:15}: {df[risk_factor].mean():6.1%}")

    print("\nReadmission:")
    print(
        f"  Ever readmitted:      "
        f"{df['EVER_READMITTED'].sum():,} patients ({df['EVER_READMITTED'].mean():.1%})"
    )
    print(f"  Mean READMIT_COUNT:   {df['READMIT_COUNT'].mean():.2f}")
    print(f"  Mean READMIT_RATE:    {df['READMIT_RATE'].mean():.1%}")

    print("\nAge Distribution (first encounter):")
    for age_bin, count in df["AGE_BIN"].value_counts().sort_index().items():
        print(f"  {age_bin:10}: {count:6,} ({count / len(df):5.1%})")

    print("\n" + "=" * 80)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Aggregate visit-level patient data to patient-level profiles (first encounter)"
    )
    parser.add_argument(
        "--input",
        default="data/processed/patients_processed.csv",
        help="Path to input visit-level data (default: data/processed/patients_processed.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/patients_patient_level.csv",
        help="Path to output patient-level data (default: data/processed/patients_patient_level.csv)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    if verbose:
        print(f"Input file:  {args.input}")
        print(f"Output file: {args.output}")
        print()

    visits_df = pd.read_csv(args.input)

    patient_df = aggregate_to_patient_level(visits_df, verbose=verbose)

    valid = validate_patient_data(patient_df, verbose=verbose)
    if not valid:
        print("\n❌ Validation failed! Please review the issues above.")
        return 1

    if verbose:
        print_summary_statistics(patient_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSaving patient-level data to: {args.output}")
    patient_df.to_csv(args.output, index=False)

    if verbose:
        file_size = output_path.stat().st_size / 1024 / 1024
        print(f"✅ Saved successfully! ({file_size:.1f} MB)")
        print(f"\nOutput: {len(patient_df):,} patients × {len(patient_df.columns)} columns")

    return 0


if __name__ == "__main__":
    exit(main())
