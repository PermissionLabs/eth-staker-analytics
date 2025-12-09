"""
Plotly chart generation for ETH Stakers Portfolio Dashboard.
"""

from typing import Optional, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import CHART_COLORS, DEFAULT_ASSET_GROUPS


def create_metric_cards(df: pd.DataFrame) -> dict:
    """
    Calculate metrics for summary cards.

    Returns dict with:
        - wallet_count: Number of wallets
        - avg_eth_amount: Average ETH group amount held
        - median_eth_amount: Median ETH group amount held
        - avg_eth_ratio: Average ETH group portfolio ratio
        - avg_protocol_count: Average protocols used
        - total_value: Total USD value
    """
    # Calculate median for ETH holders only (non-zero)
    eth_holders = df[df["eth_amount"] > 0] if len(df) > 0 else df
    median_eth_holders = eth_holders["eth_amount"].median() if len(eth_holders) > 0 else 0

    return {
        "wallet_count": len(df),
        "avg_eth_amount": df["eth_amount"].mean() if len(df) > 0 else 0,
        "median_eth_amount": df["eth_amount"].median() if len(df) > 0 else 0,
        "median_eth_holders": median_eth_holders,
        "eth_holder_count": len(eth_holders),
        "avg_eth_ratio": df["eth_ratio"].mean() * 100 if len(df) > 0 else 0,
        "avg_protocol_count": df["protocol_count"].mean() if len(df) > 0 else 0,
        "total_value": df["total_usd"].sum() if len(df) > 0 else 0,
        "avg_wallet_age": df["wallet_days"].mean() if len(df) > 0 else 0,
        "avg_eth_defi_ratio": df["eth_defi_ratio"].mean() * 100 if len(df) > 0 else 0
    }


def create_asset_pie_chart(
    df: pd.DataFrame,
    asset_groups: Optional[dict] = None
) -> go.Figure:
    """Create pie chart showing asset group distribution."""
    if asset_groups is None:
        asset_groups = DEFAULT_ASSET_GROUPS

    # Calculate total value per group
    group_totals = {}
    colors = {}

    for group_name, group_config in asset_groups.items():
        col_name = f"{group_name.lower()}_value"
        if col_name in df.columns:
            group_totals[group_name] = df[col_name].sum()
            colors[group_name] = group_config.get("color", "#888888")

    # Add OTHER
    if "other_value" in df.columns:
        group_totals["OTHER"] = df["other_value"].sum()
        colors["OTHER"] = "#888888"

    # Filter out zero values
    group_totals = {k: v for k, v in group_totals.items() if v > 0}

    if not group_totals:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    labels = list(group_totals.keys())
    values = list(group_totals.values())
    color_list = [colors.get(l, "#888888") for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker_colors=color_list,
        textinfo="label+percent",
        textposition="outside",
        hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>"
    )])

    fig.update_layout(
        title="Asset Group Distribution",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=50, b=50, l=20, r=20),
        height=400
    )

    return fig


def create_distribution_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    x_label: str,
    nbins: int = 30,
    log_x: bool = False
) -> go.Figure:
    """Create histogram for distribution analysis."""
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title,
        labels={column: x_label},
        color_discrete_sequence=[CHART_COLORS[0]]
    )

    if log_x:
        fig.update_xaxes(type="log")

    # Add clear hover template
    fig.update_traces(
        hovertemplate=f"<b>{x_label}</b>: %{{x:,.0f}}<br><b>지갑 수</b>: %{{y}}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title="지갑 수",
        bargap=0.1,
        height=350,
        margin=dict(t=50, b=50, l=50, r=20)
    )

    return fig


def create_eth_ratio_histogram(df: pd.DataFrame) -> go.Figure:
    """Create histogram for ETH group portfolio ratio distribution."""
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    # Convert to percentage
    df_plot = df.copy()
    df_plot["eth_ratio_pct"] = df_plot["eth_ratio"] * 100

    fig = px.histogram(
        df_plot,
        x="eth_ratio_pct",
        nbins=20,
        title="ETH 그룹 자산 비율 분포",
        labels={"eth_ratio_pct": "ETH 그룹 비율 (%)"},
        color_discrete_sequence=[CHART_COLORS[0]]
    )

    fig.update_traces(
        hovertemplate="<b>ETH 그룹 비율</b>: %{x:.0f}%<br><b>지갑 수</b>: %{y}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="ETH 그룹 비율 (%)",
        yaxis_title="지갑 수",
        xaxis=dict(range=[0, 100], ticksuffix="%"),
        bargap=0.1,
        height=350
    )

    return fig


def create_wallet_age_histogram(df: pd.DataFrame) -> go.Figure:
    """Create histogram for wallet age distribution."""
    return create_distribution_histogram(
        df,
        column="wallet_days",
        title="지갑 나이 분포",
        x_label="지갑 나이 (일)",
        nbins=30
    )


def create_total_value_histogram(df: pd.DataFrame) -> go.Figure:
    """Create histogram for total USD value distribution (log scale)."""
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    # Filter out zero/negative values for log scale
    df_plot = df[df["total_usd"] > 0].copy()

    if len(df_plot) == 0:
        return go.Figure().add_annotation(text="No wallets with value > 0", showarrow=False)

    # Use log-transformed values for proper histogram binning
    import numpy as np
    df_plot["log_usd"] = np.log10(df_plot["total_usd"])

    fig = px.histogram(
        df_plot,
        x="log_usd",
        nbins=30,
        title="총 자산 분포",
        color_discrete_sequence=[CHART_COLORS[0]]
    )

    # Dynamic tick values with intermediate steps (1, 2, 5 pattern)
    min_log = df_plot["log_usd"].min()
    max_log = df_plot["log_usd"].max()

    def format_usd_tick(value):
        """Format USD value for tick label."""
        if value >= 1e9:
            return f"${value/1e9:.0f}B"
        elif value >= 1e6:
            return f"${value/1e6:.0f}M"
        elif value >= 1e3:
            return f"${value/1e3:.0f}K"
        else:
            return f"${value:.0f}"

    def format_usd_hover(value):
        """Format USD value for hover tooltip."""
        if value >= 1e9:
            return f"${value/1e9:,.2f}B"
        elif value >= 1e6:
            return f"${value/1e6:,.2f}M"
        elif value >= 1e3:
            return f"${value/1e3:,.1f}K"
        else:
            return f"${value:,.0f}"

    # Generate ticks at 1, 2, 5 intervals for each decade
    tickvals = []
    ticktext = []
    base = int(np.floor(min_log))
    max_base = int(np.ceil(max_log))

    for exp in range(base, max_base + 1):
        for mult in [1, 2, 5]:
            log_val = exp + np.log10(mult)
            if min_log - 0.1 <= log_val <= max_log + 0.1:
                tickvals.append(log_val)
                ticktext.append(format_usd_tick(10 ** log_val))

    # Custom hover with actual USD value
    fig.update_traces(
        hovertemplate="<b>총 자산</b>: ~%{customdata}<br><b>지갑 수</b>: %{y}<extra></extra>",
        customdata=[format_usd_hover(10 ** v) for v in df_plot["log_usd"]]
    )

    fig.update_layout(
        xaxis_title="총 자산 (USD, 로그 스케일)",
        yaxis_title="지갑 수",
        xaxis=dict(tickvals=tickvals, ticktext=ticktext),
        bargap=0.1,
        height=350
    )

    return fig


def create_protocol_bar_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Create horizontal bar chart of most used protocols."""
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    # Count protocol usage
    protocol_counts = {}
    for protocols in df["protocols_used"]:
        if isinstance(protocols, list):
            for protocol in protocols:
                if protocol and not protocol.startswith("0x"):
                    protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1

    if not protocol_counts:
        return go.Figure().add_annotation(text="No protocol data", showarrow=False)

    # Sort and take top N
    sorted_protocols = sorted(protocol_counts.items(), key=lambda x: x[1], reverse=True)
    top_protocols = sorted_protocols[:top_n]

    names = [p[0] for p in top_protocols]
    counts = [p[1] for p in top_protocols]

    fig = go.Figure(go.Bar(
        x=counts,
        y=names,
        orientation="h",
        marker_color=CHART_COLORS[0],
        hovertemplate="<b>%{y}</b><br>%{x} wallets<extra></extra>"
    ))

    fig.update_layout(
        title=f"Top {top_n} Protocols by Usage",
        xaxis_title="Number of Wallets",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 25),
        margin=dict(l=150)
    )

    return fig


def create_protocol_count_histogram(df: pd.DataFrame) -> go.Figure:
    """Create histogram for number of protocols used per wallet."""
    return create_distribution_histogram(
        df,
        column="protocol_count",
        title="프로토콜 수 분포",
        x_label="프로토콜 수",
        nbins=20
    )


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    log_x: bool = False,
    log_y: bool = False,
    color_col: Optional[str] = None
) -> go.Figure:
    """Create scatter plot for correlation analysis."""
    if len(df) == 0:
        return go.Figure().add_annotation(text="No data", showarrow=False)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=title,
        labels={x_col: x_label, y_col: y_label},
        color_continuous_scale="Viridis" if color_col else None,
        opacity=0.6
    )

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")

    fig.update_layout(
        height=400,
        margin=dict(t=50, b=50, l=50, r=20)
    )

    return fig


def create_comparison_bar_chart(
    metrics_all: dict,
    metrics_filtered: dict,
    metric_keys: List[str],
    labels: List[str]
) -> go.Figure:
    """Create grouped bar chart comparing all wallets vs filtered."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="All Wallets",
        x=labels,
        y=[metrics_all.get(k, 0) for k in metric_keys],
        marker_color=CHART_COLORS[0],
        opacity=0.7
    ))

    fig.add_trace(go.Bar(
        name="Filtered",
        x=labels,
        y=[metrics_filtered.get(k, 0) for k in metric_keys],
        marker_color=CHART_COLORS[2]
    ))

    fig.update_layout(
        title="Filtered vs All Wallets",
        barmode="group",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def format_usd(value: float) -> str:
    """Format USD value with appropriate suffix."""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.0f}"


def format_number(value: float, decimals: int = 1) -> str:
    """Format number with appropriate suffix."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"
