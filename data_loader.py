"""
Data loading and preprocessing for ETH Stakers Portfolio Dashboard.
Loads JSON files from data/results/ and converts to pandas DataFrame.
"""

import json
import re
from pathlib import Path
from typing import Optional, Union, Dict, List

import pandas as pd

from config import DEFAULT_ASSET_GROUPS


def parse_usd_value(value_str: str) -> float:
    """Parse USD value string like '$812,321' or '$1.2M' to float."""
    if not value_str or value_str == "-":
        return 0.0

    # Remove $ and commas
    cleaned = value_str.replace("$", "").replace(",", "").strip()

    # Handle K/M/B suffixes
    multipliers = {"K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if cleaned.endswith(suffix):
            return float(cleaned[:-1]) * mult

    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_days(lifetime_str: str) -> int:
    """Parse wallet lifetime string like '678 days' to int."""
    if not lifetime_str:
        return 0

    match = re.search(r"(\d+)\s*days?", lifetime_str, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0


def parse_amount(amount_str: str) -> float:
    """Parse token amount string like '137,873.6181' to float."""
    if not amount_str:
        return 0.0

    cleaned = str(amount_str).replace(",", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def calculate_asset_group_values(
    tokens: List[dict],
    asset_groups: dict
) -> Dict[str, float]:
    """Calculate USD value for each asset group from token list."""
    group_values = {group: 0.0 for group in asset_groups}
    group_values["OTHER"] = 0.0

    # Build token -> group mapping
    token_to_group = {}
    for group_name, group_data in asset_groups.items():
        for token in group_data["tokens"]:
            token_to_group[token.upper()] = group_name

    for token in tokens:
        symbol = token.get("symbol", "").upper()
        usd_value = parse_usd_value(token.get("usdValue", "0"))

        group = token_to_group.get(symbol, "OTHER")
        group_values[group] += usd_value

    return group_values


def calculate_eth_amount(tokens: List[dict], asset_groups: dict) -> float:
    """Calculate total ETH amount (including derivatives)."""
    eth_tokens = set(t.upper() for t in asset_groups.get("ETH", {}).get("tokens", []))
    total_eth = 0.0

    for token in tokens:
        symbol = token.get("symbol", "").upper()
        if symbol in eth_tokens:
            amount = parse_amount(token.get("amount", "0"))
            total_eth += amount

    return total_eth


def calculate_btc_amount(tokens: List[dict], asset_groups: dict) -> float:
    """Calculate total BTC amount (including wrapped variants)."""
    btc_tokens = set(t.upper() for t in asset_groups.get("BTC", {}).get("tokens", []))
    total_btc = 0.0

    for token in tokens:
        symbol = token.get("symbol", "").upper()
        if symbol in btc_tokens:
            amount = parse_amount(token.get("amount", "0"))
            total_btc += amount

    return total_btc


def calculate_defi_ratio_by_group(
    positions: List[dict],
    group_tokens: List[str]
) -> float:
    """
    Calculate ratio of DeFi positions for a specific asset group.

    Uses TOKEN-BASED logic: if the position's token is in the group,
    it counts as that group's DeFi regardless of category.

    Args:
        positions: List of DeFi position dicts
        group_tokens: List of token symbols for this asset group
    """
    if not positions or not group_tokens:
        return 0.0

    # Normalize tokens to uppercase for matching
    group_tokens_upper = {t.upper() for t in group_tokens}

    total_defi_value = 0.0
    group_defi_value = 0.0

    for pos in positions:
        usd_value = parse_usd_value(pos.get("usdValue", "0"))
        total_defi_value += usd_value

        # Check if token is in this asset group
        token = pos.get("token", "").upper()
        if token in group_tokens_upper:
            group_defi_value += usd_value

    if total_defi_value == 0:
        return 0.0

    return group_defi_value / total_defi_value


def calculate_all_defi_ratios(
    positions: List[dict],
    asset_groups: dict
) -> Dict[str, float]:
    """
    Calculate DeFi ratios for all asset groups.

    Returns dict like: {"eth_defi_ratio": 0.5, "usd_defi_ratio": 0.3, ...}
    """
    ratios = {}
    for group_name, group_config in asset_groups.items():
        tokens = group_config.get("tokens", [])
        ratio = calculate_defi_ratio_by_group(positions, tokens)
        ratios[f"{group_name.lower()}_defi_ratio"] = ratio
    return ratios


def calculate_eth_defi_ratio(
    positions: List[dict],
    eth_tokens: Optional[List[str]] = None
) -> float:
    """
    Calculate ratio of ETH-related DeFi positions to total DeFi.
    Wrapper for backward compatibility.
    """
    if eth_tokens is None:
        eth_tokens = DEFAULT_ASSET_GROUPS.get("ETH", {}).get("tokens", ["ETH"])
    return calculate_defi_ratio_by_group(positions, eth_tokens)


def extract_protocols_used(positions: List[dict]) -> List[str]:
    """Extract list of protocols used from positions."""
    protocols = set()
    for pos in positions:
        protocol = pos.get("protocol", "").strip()
        if protocol and not protocol.startswith("0x"):
            protocols.add(protocol)
    return list(protocols)


def parse_wallet_json(json_path: Path, asset_groups: dict) -> Optional[dict]:
    """Parse a single wallet JSON file into a flat dictionary."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    portfolio = data.get("portfolio", {})
    tokens = portfolio.get("tokens", [])
    positions = portfolio.get("positions", [])

    # Basic info
    address = portfolio.get("address", json_path.stem)
    total_usd = parse_usd_value(portfolio.get("totalValue", "0"))
    wallet_days = parse_days(portfolio.get("walletLifetime", "0 days"))

    # Asset group values
    group_values = calculate_asset_group_values(tokens, asset_groups)

    # Calculate ratios
    total_group_value = sum(group_values.values())
    group_ratios = {}
    for group, value in group_values.items():
        ratio = value / total_group_value if total_group_value > 0 else 0.0
        group_ratios[f"{group.lower()}_ratio"] = ratio
        group_ratios[f"{group.lower()}_value"] = value

    # ETH and BTC specific calculations
    eth_amount = calculate_eth_amount(tokens, asset_groups)
    btc_amount = calculate_btc_amount(tokens, asset_groups)

    # Calculate DeFi ratios for ALL asset groups
    defi_ratios = calculate_all_defi_ratios(positions, asset_groups)

    # Protocol info
    protocols_used = extract_protocols_used(positions)
    protocol_count = len(protocols_used)

    # Token info - list of token symbols held
    tokens_held = [t.get("symbol", "").upper() for t in tokens if t.get("symbol", "").strip()]
    tokens_held = list(set(t for t in tokens_held if t and not t.startswith("0X")))

    # Total DeFi value
    total_defi_value = sum(parse_usd_value(p.get("usdValue", "0")) for p in positions)

    # Vault info (will be populated when merged with query data)
    # Expected fields: vault_name (str), vault_share (float 0-100)
    vault_name = data.get("vault_name", None)
    vault_share = data.get("vault_share", None)

    return {
        "address": address,
        "total_usd": total_usd,
        "wallet_days": wallet_days,
        "eth_amount": eth_amount,
        "btc_amount": btc_amount,
        "protocol_count": protocol_count,
        "protocols_used": protocols_used,
        "tokens_held": tokens_held,
        "total_defi_value": total_defi_value,
        "vault_name": vault_name,
        "vault_share": vault_share,
        **group_ratios,
        **defi_ratios
    }


def load_vault_mapping(data_dir: Union[str, Path]) -> Dict[str, dict]:
    """
    Load vault source and percentage mapping from all_addresses.json.

    Returns:
        Dict mapping address (lowercase) to {'vault_name': str, 'vault_share': float}
    """
    data_path = Path(data_dir)

    # Try to find all_addresses.json in parent directory
    possible_paths = [
        data_path.parent / "all_addresses.json",
        data_path.parent.parent / "data" / "all_addresses.json",
    ]

    for addresses_file in possible_paths:
        if addresses_file.exists():
            try:
                with open(addresses_file, "r", encoding="utf-8") as f:
                    addresses_data = json.load(f)

                mapping = {}
                for item in addresses_data:
                    addr = item.get("address", "").lower()
                    if addr:
                        mapping[addr] = {
                            "vault_name": item.get("source", "unknown"),
                            "vault_share": item.get("vaultPct", 0) * 100  # Convert to percentage
                        }
                return mapping
            except (json.JSONDecodeError, IOError):
                continue

    return {}


def load_all_wallets(
    data_dir: Union[str, Path],
    asset_groups: Optional[dict] = None
) -> pd.DataFrame:
    """
    Load all wallet JSON files from directory into a DataFrame.

    Args:
        data_dir: Path to directory containing JSON files
        asset_groups: Custom asset group configuration (uses default if None)

    Returns:
        DataFrame with one row per wallet
    """
    if asset_groups is None:
        asset_groups = DEFAULT_ASSET_GROUPS

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    json_files = list(data_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found in: {data_dir}")

    # Load vault mapping
    vault_mapping = load_vault_mapping(data_path)

    records = []
    for json_file in json_files:
        record = parse_wallet_json(json_file, asset_groups)
        if record:
            # Add vault info from mapping
            addr = record["address"].lower()
            if addr in vault_mapping:
                record["vault_name"] = vault_mapping[addr]["vault_name"]
                record["vault_share"] = vault_mapping[addr]["vault_share"]
            records.append(record)

    df = pd.DataFrame(records)

    # Ensure ratio columns exist even if no data
    for group in list(asset_groups.keys()) + ["OTHER"]:
        ratio_col = f"{group.lower()}_ratio"
        value_col = f"{group.lower()}_value"
        defi_ratio_col = f"{group.lower()}_defi_ratio"
        if ratio_col not in df.columns:
            df[ratio_col] = 0.0
        if value_col not in df.columns:
            df[value_col] = 0.0
        if defi_ratio_col not in df.columns:
            df[defi_ratio_col] = 0.0

    return df


def get_all_protocols(df: pd.DataFrame) -> List[str]:
    """Extract all unique protocols from the DataFrame."""
    all_protocols = set()
    for protocols in df["protocols_used"]:
        if isinstance(protocols, list):
            all_protocols.update(protocols)
    return sorted(all_protocols)


def get_all_tokens_from_data(data_dir: Union[str, Path]) -> List[str]:
    """Extract all unique token symbols from the data directory."""
    data_path = Path(data_dir)
    tokens = set()

    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for token in data.get("portfolio", {}).get("tokens", []):
                symbol = token.get("symbol", "").strip()
                if symbol:
                    tokens.add(symbol)
        except (json.JSONDecodeError, IOError):
            continue

    return sorted(tokens)


def get_all_defi_categories_from_data(data_dir: Union[str, Path]) -> List[str]:
    """Extract all unique DeFi categories from the data directory."""
    data_path = Path(data_dir)
    categories = set()

    for json_file in data_path.glob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for pos in data.get("portfolio", {}).get("positions", []):
                category = pos.get("category", "").strip().lower()
                if category:
                    categories.add(category)
        except (json.JSONDecodeError, IOError):
            continue

    return sorted(categories)


def get_data_summary(df: pd.DataFrame) -> dict:
    """Get summary statistics for the loaded data."""
    return {
        "total_wallets": len(df),
        "total_value_usd": df["total_usd"].sum(),
        "avg_wallet_value": df["total_usd"].mean(),
        "median_wallet_value": df["total_usd"].median(),
        "avg_eth_amount": df["eth_amount"].mean(),
        "avg_eth_ratio": df["eth_ratio"].mean() if "eth_ratio" in df.columns else 0,
        "avg_wallet_age_days": df["wallet_days"].mean(),
        "total_protocols": len(get_all_protocols(df))
    }
