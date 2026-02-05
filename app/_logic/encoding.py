# app/_logic/encoding.py
"""
Shared encoding utilities for patient similarity network (PSN) feature engineering.

This module provides functions for compact encoding of categorical and binary features,
enabling efficient similarity computation and reduced dimensionality.
"""

import pandas as pd
import numpy as np


def integer_encode_categoricals(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, dict]:
    """
    Integer-encode categorical columns (one column per categorical variable).

    Each unique category is mapped to a sequential integer (0, 1, 2, ..., N-1).
    This encoding is useful for exact-match similarity metrics and reduces dimensionality
    compared to one-hot encoding.

    Args:
        df: Input DataFrame
        cols: List of categorical column names to encode

    Returns:
        Tuple of (encoded_df, mappings) where:
        - encoded_df: DataFrame with integer-encoded columns (col -> col_encoded)
        - mappings: Dict mapping {col: {category: int, ...}}

    Example:
        >>> df = pd.DataFrame({"Sex": ["M", "F", "M"], "Race": ["White", "Black", "White"]})
        >>> encoded, mappings = integer_encode_categoricals(df, ["Sex", "Race"])
        >>> print(encoded)
           Sex_encoded  Race_encoded
        0            1             1
        1            0             0
        2            1             1
        >>> print(mappings)
        {'Sex': {'F': 0, 'M': 1}, 'Race': {'Black': 0, 'White': 1}}
    """
    if not cols:
        return pd.DataFrame(index=df.index), {}

    encoded = pd.DataFrame(index=df.index)
    mappings = {}

    for colname in cols:
        if colname not in df.columns:
            continue

        # Get unique categories
        categories = df[colname].astype("string").fillna("Unknown").unique()
        # Map each category to an integer
        cat_to_int = {cat: idx for idx, cat in enumerate(sorted(categories))}

        # Encode
        encoded[f"{colname}_encoded"] = (
            df[colname].astype("string").fillna("Unknown").map(cat_to_int).astype(int)
        )
        mappings[colname] = cat_to_int

    return encoded, mappings


def bitflag_encode_multibinary(
    df: pd.DataFrame, cols: list[str], column_name: str = "comorbidities_encoded"
) -> tuple[pd.DataFrame, dict]:
    """
    Combine multiple binary columns into a single bitflag integer column.
    Each bit represents one binary feature.

    This encoding is useful for efficient storage and Jaccard/Hamming similarity computation.
    Instead of N binary columns, creates a single integer column where bit positions
    correspond to each binary feature.

    Args:
        df: Input DataFrame
        cols: List of binary column names to encode
        column_name: Name for the output encoded column (default: "comorbidities_encoded")

    Returns:
        Tuple of (encoded_df, bit_mapping) where:
        - encoded_df: DataFrame with single encoded column
        - bit_mapping: Dict mapping {bit_position: column_name}

    Example:
        >>> df = pd.DataFrame({
        ...     "Diabetes": [1, 0, 1],
        ...     "Hypertension": [1, 1, 0],
        ...     "Obesity": [0, 1, 1]
        ... })
        >>> encoded, mapping = bitflag_encode_multibinary(df, ["Diabetes", "Hypertension", "Obesity"])
        >>> print(encoded)
           comorbidities_encoded
        0                      3  # Binary: 011 (Diabetes=1, Hypertension=1, Obesity=0)
        1                      6  # Binary: 110 (Diabetes=0, Hypertension=1, Obesity=1)
        2                      5  # Binary: 101 (Diabetes=1, Hypertension=0, Obesity=1)
        >>> print(mapping)
        {0: 'Diabetes', 1: 'Hypertension', 2: 'Obesity'}
    """
    if not cols:
        return pd.DataFrame(index=df.index), {}

    # Filter to columns that exist
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return pd.DataFrame(index=df.index), {}

    # Create bit mapping
    bit_mapping = dict(enumerate(valid_cols))

    # Compute bitflag
    bitflags = pd.Series(0, index=df.index, dtype=int)
    for bit_pos, colname in bit_mapping.items():
        # Convert to binary (0 or 1)
        binary_val = pd.to_numeric(
            df[colname], errors="coerce"
        ).fillna(0).clip(0, 1).astype(int)
        # Set the bit if 1
        bitflags += binary_val * (2 ** bit_pos)

    encoded = pd.DataFrame({column_name: bitflags})
    return encoded, bit_mapping


def format_encoded_display(
    encoded_df: pd.DataFrame,
    categorical_mappings: dict,
    bitflag_mapping: dict,
    bitflag_column: str = "comorbidities_encoded"
) -> pd.DataFrame:
    """
    Format encoded DataFrame for display with leading zeros.

    Categorical columns are formatted as decimal integers with leading zeros.
    Bitflag column is formatted as binary string with leading zeros.

    Args:
        encoded_df: DataFrame with encoded columns
        categorical_mappings: Dict of {col: {category: int}} from integer_encode_categoricals
        bitflag_mapping: Dict of {bit_position: col} from bitflag_encode_multibinary
        bitflag_column: Name of the bitflag column (default: "comorbidities_encoded")

    Returns:
        DataFrame with formatted string representations for display
    """
    display_df = encoded_df.copy()

    # Format categorical integer-encoded columns with leading zeros
    for col_name in display_df.columns:
        if col_name.endswith("_encoded") and col_name != bitflag_column:
            # Get the original column name (remove _encoded suffix)
            orig_name = col_name.replace("_encoded", "")
            if orig_name in categorical_mappings:
                # Determine width based on max category value
                max_val = max(categorical_mappings[orig_name].values())
                num_digits = len(str(max_val))
                # Format with leading zeros
                fmt_str = f"{{:0{num_digits}d}}"
                display_df[col_name] = (
                    display_df[col_name]
                    .astype(int)
                    .apply(fmt_str.format)
                )

    # Format bitflag column as binary string
    if bitflag_column in display_df.columns and bitflag_mapping:
        # Determine bit width from number of binary features
        num_bits = len(bitflag_mapping)
        # Convert to binary string with leading zeros
        fmt_str = f"{{:0{num_bits}b}}"
        display_df[bitflag_column] = (
            display_df[bitflag_column]
            .astype(int)
            .apply(fmt_str.format)
        )

    return display_df


def decode_bitflag(bitflag_value: int, bit_mapping: dict) -> list[str]:
    """
    Decode a bitflag integer back to list of active feature names.

    Args:
        bitflag_value: Integer bitflag value
        bit_mapping: Dict mapping {bit_position: column_name}

    Returns:
        List of feature names where bit is set to 1

    Example:
        >>> mapping = {0: 'Diabetes', 1: 'Hypertension', 2: 'Obesity'}
        >>> decode_bitflag(5, mapping)  # Binary: 101
        ['Diabetes', 'Obesity']
    """
    active_features = []
    for bit_pos, feature_name in bit_mapping.items():
        if bitflag_value & (2 ** bit_pos):
            active_features.append(feature_name)
    return active_features


def decode_categorical(encoded_value: int, category_mapping: dict) -> str:
    """
    Decode an integer-encoded categorical value back to original category.

    Args:
        encoded_value: Integer encoded value
        category_mapping: Dict mapping {category: int}

    Returns:
        Original category string, or "Unknown" if not found

    Example:
        >>> mapping = {'F': 0, 'M': 1}
        >>> decode_categorical(1, mapping)
        'M'
    """
    # Reverse the mapping
    reverse_map = {v: k for k, v in category_mapping.items()}
    return reverse_map.get(encoded_value, "Unknown")
