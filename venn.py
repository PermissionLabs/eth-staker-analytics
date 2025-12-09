"""
Venn diagram logic for filter intersection analysis.
"""

from typing import Optional, List, Set, Tuple
import io

import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import pandas as pd

from filters import FilterConfig, get_filter_set, calculate_intersections


def create_venn_diagram(
    sets: List[Set[str]],
    labels: List[str],
    title: str = "Filter Intersection Analysis"
) -> plt.Figure:
    """
    Create a Venn diagram for 2 or 3 sets.

    Args:
        sets: List of 2 or 3 sets of addresses
        labels: Labels for each set
        title: Chart title

    Returns:
        matplotlib Figure
    """
    n = len(sets)
    if n < 2 or n > 3:
        raise ValueError("Venn diagram requires 2 or 3 sets")

    # Color palette
    colors = ["#627EEA", "#F7931A", "#26A17B"]

    fig, ax = plt.subplots(figsize=(10, 8))

    if n == 2:
        venn = venn2(
            subsets=sets,
            set_labels=labels,
            ax=ax,
            set_colors=colors[:2],
            alpha=0.7
        )
    else:
        venn = venn3(
            subsets=sets,
            set_labels=labels,
            ax=ax,
            set_colors=colors[:3],
            alpha=0.7
        )

    # Style the labels
    for text in venn.set_labels:
        if text:
            text.set_fontsize(12)
            text.set_fontweight("bold")

    for text in venn.subset_labels:
        if text:
            text.set_fontsize(11)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


def create_intersection_table(
    sets: List[Set[str]],
    labels: List[str]
) -> pd.DataFrame:
    """
    Create a summary table of set intersections.

    Returns DataFrame with intersection statistics.
    """
    stats = calculate_intersections(sets, labels)

    # Convert to DataFrame for nice display
    rows = []
    for key, value in stats.items():
        rows.append({"Region": key, "Count": value})

    df = pd.DataFrame(rows)
    return df


def get_venn_sets_from_filters(
    df: pd.DataFrame,
    filter_configs: List[FilterConfig]
) -> Tuple[List[Set[str]], List[str]]:
    """
    Convert filter configurations to address sets for Venn diagram.

    Args:
        df: Source DataFrame
        filter_configs: List of FilterConfig objects

    Returns:
        Tuple of (list of sets, list of labels)
    """
    sets = []
    labels = []

    for config in filter_configs:
        address_set = get_filter_set(df, config)
        sets.append(address_set)
        labels.append(config.label)

    return sets, labels


def fig_to_image(fig: plt.Figure) -> bytes:
    """Convert matplotlib figure to PNG bytes for Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


# Predefined filter combinations for quick analysis
VENN_PRESETS = {
    "staking_behavior": {
        "name": "Staking Behavior",
        "description": "Compare LIDO vs Rocket Pool users",
        "filters": [
            FilterConfig(
                name="lido",
                column="protocols_used",
                filter_type="contains",
                value=["LIDO"],
                label="LIDO"
            ),
            FilterConfig(
                name="rocket_pool",
                column="protocols_used",
                filter_type="contains",
                value=["Rocket Pool"],
                label="Rocket Pool"
            )
        ]
    },
    "wallet_profile": {
        "name": "Wallet Profile",
        "description": "New wallets vs ETH-heavy vs whales",
        "filters": [
            FilterConfig(
                name="new",
                column="wallet_days",
                filter_type="max",
                value=365,
                label="< 1 Year"
            ),
            FilterConfig(
                name="eth_heavy",
                column="eth_ratio",
                filter_type="min",
                value=0.5,
                label="ETH 그룹 > 50%"
            ),
            FilterConfig(
                name="whale",
                column="total_usd",
                filter_type="min",
                value=1_000_000,
                label="> $1M"
            )
        ]
    },
    "eth_exposure": {
        "name": "ETH 그룹 노출도",
        "description": "ETH 그룹 비율 vs DeFi 활동 vs 멀티 프로토콜",
        "filters": [
            FilterConfig(
                name="eth_heavy",
                column="eth_ratio",
                filter_type="min",
                value=0.5,
                label="ETH 그룹 > 50%"
            ),
            FilterConfig(
                name="high_defi",
                column="eth_defi_ratio",
                filter_type="min",
                value=0.3,
                label="ETH 그룹 DeFi > 30%"
            ),
            FilterConfig(
                name="multi_protocol",
                column="protocol_count",
                filter_type="min",
                value=5,
                label="5+ Protocols"
            )
        ]
    }
}
