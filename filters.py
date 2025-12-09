"""
Filter logic for ETH Stakers Portfolio Dashboard.
Provides filter application and set-based intersection analysis.
"""

from dataclasses import dataclass
from typing import Optional, List, Any, Set, Tuple

import pandas as pd


@dataclass
class FilterConfig:
    """Configuration for a single filter."""
    name: str
    column: str
    filter_type: str  # "range", "min", "max", "contains", "equals"
    value: Any
    label: str  # Human-readable label for display


def apply_range_filter(
    df: pd.DataFrame,
    column: str,
    min_val: Optional[float],
    max_val: Optional[float]
) -> pd.DataFrame:
    """Apply min/max range filter to a column."""
    result = df.copy()
    if min_val is not None:
        result = result[result[column] >= min_val]
    if max_val is not None:
        result = result[result[column] <= max_val]
    return result


def apply_min_filter(
    df: pd.DataFrame,
    column: str,
    min_val: float
) -> pd.DataFrame:
    """Filter where column >= min_val."""
    return df[df[column] >= min_val]


def apply_max_filter(
    df: pd.DataFrame,
    column: str,
    max_val: float
) -> pd.DataFrame:
    """Filter where column <= max_val."""
    return df[df[column] <= max_val]


def apply_protocol_filter(
    df: pd.DataFrame,
    protocols: List[str],
    require_all: bool = False
) -> pd.DataFrame:
    """
    Filter wallets by protocol usage.

    Args:
        df: DataFrame with 'protocols_used' column (list of strings)
        protocols: List of protocol names to filter by
        require_all: If True, wallet must use ALL protocols; if False, ANY
    """
    if not protocols:
        return df

    def check_protocols(wallet_protocols):
        if not isinstance(wallet_protocols, list):
            return False
        wallet_set = set(p.upper() for p in wallet_protocols)
        filter_set = set(p.upper() for p in protocols)

        if require_all:
            return filter_set.issubset(wallet_set)
        else:
            return bool(wallet_set & filter_set)

    return df[df["protocols_used"].apply(check_protocols)]


def apply_token_filter(
    df: pd.DataFrame,
    tokens: List[str],
    require_all: bool = True
) -> pd.DataFrame:
    """
    Filter wallets by token holdings.

    Args:
        df: DataFrame with 'tokens_held' column (list of strings)
        tokens: List of token symbols to filter by
        require_all: If True, wallet must hold ALL tokens; if False, ANY
    """
    if not tokens:
        return df

    if "tokens_held" not in df.columns:
        return df

    def check_tokens(wallet_tokens):
        if not isinstance(wallet_tokens, list):
            return False
        wallet_set = set(t.upper() for t in wallet_tokens)
        filter_set = set(t.upper() for t in tokens)

        if require_all:
            return filter_set.issubset(wallet_set)
        else:
            return bool(wallet_set & filter_set)

    return df[df["tokens_held"].apply(check_tokens)]


def apply_vault_filter(
    df: pd.DataFrame,
    vaults: List[str]
) -> pd.DataFrame:
    """
    Filter wallets by vault membership (OR logic).

    Args:
        df: DataFrame with 'vault_name' column
        vaults: List of vault names to filter by (OR condition)
    """
    if not vaults:
        return df

    if "vault_name" not in df.columns:
        return df

    # OR logic: wallet must be in ANY of the selected vaults
    return df[df["vault_name"].isin(vaults)]


def apply_filters(
    df: pd.DataFrame,
    wallet_days_range: Optional[Tuple[int, int]] = None,
    total_usd_range: Optional[Tuple[float, float]] = None,
    eth_amount_range: Optional[Tuple[float, float]] = None,
    eth_ratio_range: Optional[Tuple[float, float]] = None,
    usd_ratio_range: Optional[Tuple[float, float]] = None,
    btc_ratio_range: Optional[Tuple[float, float]] = None,
    eth_defi_ratio_range: Optional[Tuple[float, float]] = None,
    usd_defi_ratio_range: Optional[Tuple[float, float]] = None,
    btc_defi_ratio_range: Optional[Tuple[float, float]] = None,
    protocols: Optional[List[str]] = None,
    protocol_require_all: bool = False,
    tokens: Optional[List[str]] = None,
    token_require_all: bool = True,
    vaults: Optional[List[str]] = None,
    vault_share_range: Optional[Tuple[float, float]] = None
) -> pd.DataFrame:
    """
    Apply all filters to the DataFrame.

    Args:
        df: Input DataFrame
        wallet_days_range: (min_days, max_days) filter
        total_usd_range: (min_usd, max_usd) filter
        eth_amount_range: (min_eth, max_eth) filter for ETH quantity
        eth_ratio_range: (min_ratio, max_ratio) for ETH (0-1)
        usd_ratio_range: (min_ratio, max_ratio) for USD (0-1)
        btc_ratio_range: (min_ratio, max_ratio) for BTC (0-1)
        eth_defi_ratio_range: (min, max) for ETH DeFi ratio (0-1)
        usd_defi_ratio_range: (min, max) for USD DeFi ratio (0-1)
        btc_defi_ratio_range: (min, max) for BTC DeFi ratio (0-1)
        protocols: List of protocols to filter by
        protocol_require_all: Require all protocols or any
        tokens: List of tokens to filter by
        token_require_all: Require all tokens (AND) or any (OR)
        vaults: List of vault names to filter by (OR logic)
        vault_share_range: (min_share, max_share) filter for vault share %

    Returns:
        Filtered DataFrame
    """
    result = df.copy()

    # Wallet age filter
    if wallet_days_range:
        result = apply_range_filter(
            result, "wallet_days",
            wallet_days_range[0], wallet_days_range[1]
        )

    # Total USD filter
    if total_usd_range:
        result = apply_range_filter(
            result, "total_usd",
            total_usd_range[0], total_usd_range[1]
        )

    # ETH amount filter
    if eth_amount_range:
        result = apply_range_filter(
            result, "eth_amount",
            eth_amount_range[0], eth_amount_range[1]
        )

    # Asset ratio filters (range-based)
    if eth_ratio_range and (eth_ratio_range[0] > 0 or eth_ratio_range[1] < 1.0):
        result = apply_range_filter(
            result, "eth_ratio",
            eth_ratio_range[0], eth_ratio_range[1]
        )

    if usd_ratio_range and (usd_ratio_range[0] > 0 or usd_ratio_range[1] < 1.0):
        result = apply_range_filter(
            result, "usd_ratio",
            usd_ratio_range[0], usd_ratio_range[1]
        )

    if btc_ratio_range and (btc_ratio_range[0] > 0 or btc_ratio_range[1] < 1.0):
        result = apply_range_filter(
            result, "btc_ratio",
            btc_ratio_range[0], btc_ratio_range[1]
        )

    # DeFi ratio filters (range-based)
    if eth_defi_ratio_range and (eth_defi_ratio_range[0] > 0 or eth_defi_ratio_range[1] < 1.0):
        if "eth_defi_ratio" in result.columns:
            result = apply_range_filter(
                result, "eth_defi_ratio",
                eth_defi_ratio_range[0], eth_defi_ratio_range[1]
            )

    if usd_defi_ratio_range and (usd_defi_ratio_range[0] > 0 or usd_defi_ratio_range[1] < 1.0):
        if "usd_defi_ratio" in result.columns:
            result = apply_range_filter(
                result, "usd_defi_ratio",
                usd_defi_ratio_range[0], usd_defi_ratio_range[1]
            )

    if btc_defi_ratio_range and (btc_defi_ratio_range[0] > 0 or btc_defi_ratio_range[1] < 1.0):
        if "btc_defi_ratio" in result.columns:
            result = apply_range_filter(
                result, "btc_defi_ratio",
                btc_defi_ratio_range[0], btc_defi_ratio_range[1]
            )

    # Protocol filter
    if protocols:
        result = apply_protocol_filter(result, protocols, protocol_require_all)

    # Token filter
    if tokens:
        result = apply_token_filter(result, tokens, token_require_all)

    # Vault filter (OR logic)
    if vaults:
        result = apply_vault_filter(result, vaults)

    # Vault share filter
    if vault_share_range and "vault_share" in result.columns:
        result = apply_range_filter(
            result, "vault_share",
            vault_share_range[0], vault_share_range[1]
        )

    return result


def get_filter_set(
    df: pd.DataFrame,
    filter_config: FilterConfig
) -> Set[str]:
    """
    Get set of addresses matching a single filter.
    Used for Venn diagram intersection analysis.
    """
    if filter_config.filter_type == "range":
        min_val, max_val = filter_config.value
        filtered = apply_range_filter(df, filter_config.column, min_val, max_val)

    elif filter_config.filter_type == "min":
        filtered = apply_min_filter(df, filter_config.column, filter_config.value)

    elif filter_config.filter_type == "max":
        filtered = apply_max_filter(df, filter_config.column, filter_config.value)

    elif filter_config.filter_type == "contains":
        # For protocol list containment
        filtered = apply_protocol_filter(df, filter_config.value, require_all=False)

    else:
        filtered = df

    return set(filtered["address"].tolist())


def calculate_intersections(
    sets: List[Set[str]],
    labels: List[str]
) -> dict:
    """
    Calculate all intersection combinations for 2-3 sets.

    Returns dict with keys like:
        "A": count of A only
        "B": count of B only
        "A∩B": count of intersection
        etc.
    """
    n = len(sets)
    if n < 2 or n > 3:
        raise ValueError("Venn diagram requires 2 or 3 sets")

    result = {}

    if n == 2:
        a, b = sets
        la, lb = labels

        # Individual sets (exclusive)
        result[f"{la} only"] = len(a - b)
        result[f"{lb} only"] = len(b - a)

        # Intersection
        result[f"{la} ∩ {lb}"] = len(a & b)

        # Totals
        result[f"{la} total"] = len(a)
        result[f"{lb} total"] = len(b)
        result["Union"] = len(a | b)

    elif n == 3:
        a, b, c = sets
        la, lb, lc = labels

        # Exclusive regions (only in one set)
        result[f"{la} only"] = len(a - b - c)
        result[f"{lb} only"] = len(b - a - c)
        result[f"{lc} only"] = len(c - a - b)

        # Pairwise intersections (exclusive of third)
        result[f"{la} ∩ {lb} only"] = len((a & b) - c)
        result[f"{la} ∩ {lc} only"] = len((a & c) - b)
        result[f"{lb} ∩ {lc} only"] = len((b & c) - a)

        # Triple intersection
        result[f"{la} ∩ {lb} ∩ {lc}"] = len(a & b & c)

        # Totals
        result[f"{la} total"] = len(a)
        result[f"{lb} total"] = len(b)
        result[f"{lc} total"] = len(c)
        result["Union"] = len(a | b | c)

    return result


def get_intersection_addresses(
    sets: List[Set[str]],
    include_indices: List[int],
    exclude_indices: List[int]
) -> Set[str]:
    """
    Get addresses that are in specified sets and not in others.

    Args:
        sets: List of address sets
        include_indices: Indices of sets to intersect
        exclude_indices: Indices of sets to exclude

    Returns:
        Set of addresses matching criteria
    """
    if not include_indices:
        return set()

    result = sets[include_indices[0]].copy()
    for idx in include_indices[1:]:
        result &= sets[idx]

    for idx in exclude_indices:
        result -= sets[idx]

    return result


# Predefined filter templates for quick access
FILTER_TEMPLATES = {
    "new_wallets": FilterConfig(
        name="new_wallets",
        column="wallet_days",
        filter_type="max",
        value=365,
        label="Wallet Age < 1 Year"
    ),
    "old_wallets": FilterConfig(
        name="old_wallets",
        column="wallet_days",
        filter_type="min",
        value=730,
        label="Wallet Age > 2 Years"
    ),
    "eth_heavy": FilterConfig(
        name="eth_heavy",
        column="eth_ratio",
        filter_type="min",
        value=0.5,
        label="ETH 그룹 비율 > 50%"
    ),
    "whale": FilterConfig(
        name="whale",
        column="total_usd",
        filter_type="min",
        value=1_000_000,
        label="Total Value > $1M"
    ),
    "lido_user": FilterConfig(
        name="lido_user",
        column="protocols_used",
        filter_type="contains",
        value=["LIDO"],
        label="Uses LIDO"
    ),
    "rocket_pool_user": FilterConfig(
        name="rocket_pool_user",
        column="protocols_used",
        filter_type="contains",
        value=["Rocket Pool"],
        label="Uses Rocket Pool"
    ),
    "multi_protocol": FilterConfig(
        name="multi_protocol",
        column="protocol_count",
        filter_type="min",
        value=5,
        label="Uses 5+ Protocols"
    )
}
