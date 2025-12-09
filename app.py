"""
ETH Stakers Portfolio Dashboard - Main Streamlit Application.

Run with: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import streamlit as st
import pandas as pd
from pathlib import Path

from config import DEFAULT_ASSET_GROUPS, KNOWN_PROTOCOLS, UI_CONFIG
from data_loader import (
    load_all_wallets, get_all_protocols, get_data_summary,
    get_all_tokens_from_data
)
from filters import (
    apply_filters, FilterConfig, get_filter_set,
    calculate_intersections, FILTER_TEMPLATES
)
from charts import (
    create_metric_cards, create_asset_pie_chart,
    create_eth_ratio_histogram, create_wallet_age_histogram,
    create_total_value_histogram, create_protocol_bar_chart,
    create_protocol_count_histogram, format_usd, format_number
)
from venn import (
    create_venn_diagram, create_intersection_table,
    get_venn_sets_from_filters, fig_to_image, VENN_PRESETS
)

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"]
)


def find_data_directory():
    """Find the data directory."""
    app_dir = Path(__file__).parent.resolve()
    possible_paths = [
        app_dir.parent / "data" / "results",
        app_dir.parent / "data" / "backup_20251208_142052" / "results",
        app_dir / "data" / "results",
        Path.cwd().parent / "data" / "results",
        Path.cwd().parent / "data" / "backup_20251208_142052" / "results",
    ]
    for data_dir in possible_paths:
        if data_dir.exists() and list(data_dir.glob("*.json")):
            return data_dir
    raise FileNotFoundError(f"Data directory not found. Tried: {[str(p) for p in possible_paths]}")


def load_wallet_details(address: str) -> dict:
    """Load full wallet details from JSON file."""
    import json
    data_dir = find_data_directory()
    json_path = data_dir / f"{address.lower()}.json"

    if not json_path.exists():
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def load_data(eth_defi_tokens: tuple = None):
    """Load wallet data with optional ETH DeFi tokens config."""
    data_dir = find_data_directory()
    df = _load_data_cached(data_dir)

    # Recalculate eth_defi_ratio if custom tokens differ from default
    default_tokens = set(DEFAULT_ASSET_GROUPS.get("ETH", {}).get("tokens", []))
    if eth_defi_tokens and set(eth_defi_tokens) != default_tokens:
        df = recalculate_eth_defi_ratio(df, list(eth_defi_tokens), data_dir)

    return df


@st.cache_data(ttl=3600)
def _load_data_cached(data_dir):
    """Internal cached data loader."""
    return load_all_wallets(data_dir)


def recalculate_eth_defi_ratio(df: pd.DataFrame, eth_tokens: list, data_dir) -> pd.DataFrame:
    """Recalculate eth_defi_ratio based on custom token list."""
    import json
    from data_loader import parse_usd_value

    eth_tokens_upper = {t.upper() for t in eth_tokens}

    new_ratios = []
    for address in df["address"]:
        json_path = data_dir / f"{address.lower()}.json"
        ratio = 0.0

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                positions = data.get("portfolio", {}).get("positions", [])

                total_defi = 0.0
                eth_defi = 0.0
                for pos in positions:
                    usd_value = parse_usd_value(pos.get("usdValue", "0"))
                    total_defi += usd_value
                    if pos.get("token", "").upper() in eth_tokens_upper:
                        eth_defi += usd_value

                if total_defi > 0:
                    ratio = eth_defi / total_defi
            except:
                pass

        new_ratios.append(ratio)

    df = df.copy()
    df["eth_defi_ratio"] = new_ratios
    return df


@st.cache_data(ttl=3600)
def load_all_tokens():
    """Load and cache all unique tokens from data."""
    data_dir = find_data_directory()
    return get_all_tokens_from_data(data_dir)


def init_session_state():
    """Initialize session state variables."""
    if "asset_groups" not in st.session_state:
        st.session_state.asset_groups = DEFAULT_ASSET_GROUPS.copy()

    if "custom_filters" not in st.session_state:
        st.session_state.custom_filters = []

    # DeFi tokens for each asset group (defaults to asset group tokens)
    for group in ["ETH", "USD", "BTC", "EUR"]:
        key = f"{group.lower()}_defi_tokens"
        if key not in st.session_state:
            st.session_state[key] = DEFAULT_ASSET_GROUPS.get(group, {}).get("tokens", []).copy()


def render_sidebar(df: pd.DataFrame, all_protocols: list):
    """Render sidebar with filters and settings."""
    # Filtered wallet count placeholder (updated after filtering in main)
    filter_count_placeholder = st.sidebar.empty()

    # ETH Price reference display
    st.sidebar.markdown("### 💰 기준 가격")
    st.sidebar.info("📅 데이터 수집 시점: 2025-12\n💵 ETH 기준가: **$3,150**")

    st.sidebar.divider()
    st.sidebar.header("🔧 필터 설정")

    # Wallet age filter
    st.sidebar.subheader("지갑 나이 (일)")
    max_days = int(df["wallet_days"].max()) if len(df) > 0 else 1000
    wallet_days = st.sidebar.slider(
        "Days",
        min_value=0,
        max_value=max_days,
        value=(0, max_days),
        key="wallet_days_filter",
        label_visibility="collapsed"
    )

    # Total USD filter with log scale
    st.sidebar.subheader("총 자산 ($)")
    max_usd = float(df["total_usd"].max()) if len(df) > 0 else 10_000_000

    # Generate log-scale options (1-2-5 pattern)
    usd_options = [0]
    for exp in range(0, 10):  # $1 to $1B
        for mult in [1, 2, 5]:
            val = mult * (10 ** exp)
            if val <= max_usd * 1.1:
                usd_options.append(val)
    usd_options = sorted(set(usd_options))
    if max_usd not in usd_options:
        usd_options.append(max_usd)
    usd_options = sorted(usd_options)

    def format_usd_option(val):
        if val == 0:
            return "$0"
        elif val >= 1e9:
            return f"${val/1e9:.0f}B"
        elif val >= 1e6:
            return f"${val/1e6:.0f}M"
        elif val >= 1e3:
            return f"${val/1e3:.0f}K"
        else:
            return f"${val:.0f}"

    total_usd = st.sidebar.select_slider(
        "USD Range",
        options=usd_options,
        value=(0, usd_options[-1]),
        format_func=format_usd_option,
        key="total_usd_filter",
        label_visibility="collapsed"
    )

    # ETH amount filter
    st.sidebar.subheader("ETH 수량")
    max_eth = float(df["eth_amount"].max()) if len(df) > 0 else 1000

    # Generate ETH options (0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, ...)
    eth_options = [0]
    for exp in range(-1, 5):  # 0.1 to 10000
        for mult in [1, 2, 5]:
            val = mult * (10 ** exp)
            if val <= max_eth * 1.1:
                eth_options.append(val)
    eth_options = sorted(set(eth_options))
    if max_eth not in eth_options:
        eth_options.append(max_eth)
    eth_options = sorted(eth_options)

    def format_eth_option(val):
        if val == 0:
            return "0"
        elif val >= 1000:
            return f"{val/1000:.0f}K"
        elif val >= 1:
            return f"{val:.0f}"
        else:
            return f"{val:.1f}"

    eth_amount_range = st.sidebar.select_slider(
        "ETH Amount",
        options=eth_options,
        value=(0, eth_options[-1]),
        format_func=format_eth_option,
        key="eth_amount_filter",
        label_visibility="collapsed"
    )

    # ETH ratio filter (range)
    st.sidebar.subheader("ETH 비율 (%)")
    eth_ratio_range = st.sidebar.slider(
        "ETH Ratio Range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        format="%d%%",
        key="eth_ratio_filter",
        label_visibility="collapsed"
    )

    # USD ratio filter (range)
    st.sidebar.subheader("USD 비율 (%)")
    usd_ratio_range = st.sidebar.slider(
        "USD Ratio Range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        format="%d%%",
        key="usd_ratio_filter",
        label_visibility="collapsed"
    )

    # BTC ratio filter (range)
    st.sidebar.subheader("BTC 비율 (%)")
    btc_ratio_range = st.sidebar.slider(
        "BTC Ratio Range",
        min_value=0,
        max_value=100,
        value=(0, 100),
        format="%d%%",
        key="btc_ratio_filter",
        label_visibility="collapsed"
    )

    # DeFi Ratio Filters Section
    st.sidebar.divider()
    st.sidebar.subheader("🏦 DeFi 자산 비율 필터")
    st.sidebar.caption("DeFi 포지션 중 해당 자산 그룹의 비율")

    # ETH DeFi ratio filter
    st.sidebar.markdown("**ETH DeFi 비율 (%)**")
    eth_defi_ratio_range = st.sidebar.slider(
        "ETH DeFi Ratio",
        min_value=0,
        max_value=100,
        value=(0, 100),
        format="%d%%",
        key="eth_defi_ratio_filter",
        label_visibility="collapsed"
    )

    # USD DeFi ratio filter
    st.sidebar.markdown("**USD DeFi 비율 (%)**")
    usd_defi_ratio_range = st.sidebar.slider(
        "USD DeFi Ratio",
        min_value=0,
        max_value=100,
        value=(0, 100),
        format="%d%%",
        key="usd_defi_ratio_filter",
        label_visibility="collapsed"
    )

    # BTC DeFi ratio filter
    st.sidebar.markdown("**BTC DeFi 비율 (%)**")
    btc_defi_ratio_range = st.sidebar.slider(
        "BTC DeFi Ratio",
        min_value=0,
        max_value=100,
        value=(0, 100),
        format="%d%%",
        key="btc_defi_ratio_filter",
        label_visibility="collapsed"
    )

    st.sidebar.divider()

    # Protocol filter
    st.sidebar.subheader("프로토콜 선택")
    # Get protocols that exist in current data
    available_protocols = [p for p in all_protocols if p in KNOWN_PROTOCOLS or len(all_protocols) <= 30]
    if len(available_protocols) > 30:
        available_protocols = KNOWN_PROTOCOLS

    selected_protocols = st.sidebar.multiselect(
        "Protocols",
        options=available_protocols,
        default=[],
        key="protocol_filter",
        label_visibility="collapsed"
    )

    # Token filter
    st.sidebar.subheader("토큰 보유 필터")
    st.sidebar.caption("선택된 모든 토큰을 보유한 지갑 (AND 조건)")

    # Get unique tokens from DataFrame
    all_token_options = set()
    if "tokens_held" in df.columns:
        for tokens_list in df["tokens_held"]:
            if isinstance(tokens_list, list):
                all_token_options.update(tokens_list)
    all_token_options = sorted(all_token_options)

    selected_tokens = st.sidebar.multiselect(
        "Tokens",
        options=all_token_options,
        default=[],
        key="token_filter",
        label_visibility="collapsed",
        placeholder="토큰 검색..."
    )

    # Vault filter (OR logic)
    st.sidebar.divider()
    st.sidebar.subheader("🏛️ 볼트 필터")
    st.sidebar.caption("선택된 볼트 중 하나라도 해당하는 지갑 (OR 조건)")

    # Get unique vaults from DataFrame
    all_vault_options = []
    if "vault_name" in df.columns:
        all_vault_options = sorted(df["vault_name"].dropna().unique().tolist())

    selected_vaults = st.sidebar.multiselect(
        "Vaults",
        options=all_vault_options,
        default=[],
        key="vault_filter",
        label_visibility="collapsed",
        placeholder="볼트 검색..."
    )

    # Vault share filter
    if "vault_share" in df.columns and len(df) > 0:
        st.sidebar.subheader("볼트 지분율 (%)")
        max_share = float(df["vault_share"].max()) if df["vault_share"].notna().any() else 100
        vault_share_range = st.sidebar.slider(
            "Vault Share",
            min_value=0.0,
            max_value=min(100.0, max_share * 1.1),
            value=(0.0, min(100.0, max_share * 1.1)),
            format="%.2f%%",
            key="vault_share_filter",
            label_visibility="collapsed"
        )
    else:
        vault_share_range = None

    # Asset group editor
    st.sidebar.divider()

    # Load all tokens for the dropdown
    try:
        all_tokens = load_all_tokens()
    except Exception:
        all_tokens = []

    with st.sidebar.expander("📦 자산 그룹 편집"):
        render_asset_group_editor(all_tokens)

    with st.sidebar.expander("🏦 DeFi 토큰 설정"):
        render_defi_token_editor(all_tokens)

    # Cache clear button
    st.sidebar.divider()
    if st.sidebar.button("🔄 데이터 새로고침", help="캐시를 클리어하고 데이터를 다시 로드합니다"):
        st.cache_data.clear()
        st.rerun()

    return {
        "wallet_days_range": wallet_days,
        "total_usd_range": total_usd,
        "eth_amount_range": eth_amount_range,
        "eth_ratio_range": (eth_ratio_range[0] / 100.0, eth_ratio_range[1] / 100.0),
        "usd_ratio_range": (usd_ratio_range[0] / 100.0, usd_ratio_range[1] / 100.0),
        "btc_ratio_range": (btc_ratio_range[0] / 100.0, btc_ratio_range[1] / 100.0),
        "eth_defi_ratio_range": (eth_defi_ratio_range[0] / 100.0, eth_defi_ratio_range[1] / 100.0),
        "usd_defi_ratio_range": (usd_defi_ratio_range[0] / 100.0, usd_defi_ratio_range[1] / 100.0),
        "btc_defi_ratio_range": (btc_defi_ratio_range[0] / 100.0, btc_defi_ratio_range[1] / 100.0),
        "protocols": selected_protocols,
        "tokens": selected_tokens,
        "vaults": selected_vaults,
        "vault_share_range": vault_share_range,
        "_filter_count_placeholder": filter_count_placeholder
    }


def render_asset_group_editor(all_tokens: list):
    """Render asset group editor in sidebar with searchable multiselect."""
    groups = st.session_state.asset_groups

    st.caption("💡 드롭다운에서 토큰 검색 가능")

    for group_name, group_data in groups.items():
        st.write(f"**{group_name}**")

        # Current tokens in this group
        current_tokens = group_data.get("tokens", [])

        # Use multiselect with search
        new_tokens = st.multiselect(
            f"{group_name} 토큰 선택",
            options=all_tokens,
            default=[t for t in current_tokens if t in all_tokens],
            key=f"tokens_{group_name}",
            label_visibility="collapsed",
            placeholder="토큰 검색..."
        )

        # Update on change
        if set(new_tokens) != set(current_tokens):
            st.session_state.asset_groups[group_name]["tokens"] = new_tokens

    # Add new group
    st.divider()
    new_group_name = st.text_input("새 그룹 이름", key="new_group_name")

    if new_group_name:
        new_group_tokens = st.multiselect(
            "토큰 선택",
            options=all_tokens,
            default=[],
            key="new_group_tokens",
            placeholder="토큰 검색..."
        )

        if st.button("그룹 추가") and new_group_tokens:
            st.session_state.asset_groups[new_group_name.upper()] = {
                "tokens": new_group_tokens,
                "color": "#888888",
                "description": f"Custom group: {new_group_name}"
            }
            st.rerun()


def render_defi_token_editor(all_tokens: list):
    """Render DeFi token editor for all asset groups in sidebar."""
    st.caption("💡 각 자산 그룹의 DeFi Ratio 계산에 포함될 토큰")

    defi_groups = [
        ("ETH", "🔷", "#627EEA"),
        ("USD", "💵", "#26A17B"),
        ("BTC", "🟠", "#F7931A"),
        ("EUR", "💶", "#003399"),
    ]

    for group_name, icon, color in defi_groups:
        key = f"{group_name.lower()}_defi_tokens"
        current_tokens = st.session_state.get(key, [])

        st.markdown(f"**{icon} {group_name} DeFi 토큰**")
        new_tokens = st.multiselect(
            f"{group_name} DeFi 토큰",
            options=all_tokens,
            default=[t for t in current_tokens if t in all_tokens],
            key=f"{key}_select",
            label_visibility="collapsed",
            placeholder="토큰 검색..."
        )

        # Update on change
        if set(new_tokens) != set(current_tokens):
            st.session_state[key] = new_tokens
            st.rerun()

    st.divider()

    # Sync all button
    if st.button("🔄 모든 그룹 동기화", help="자산 그룹의 토큰 목록으로 DeFi 토큰 동기화"):
        for group_name, _, _ in defi_groups:
            key = f"{group_name.lower()}_defi_tokens"
            group_tokens = st.session_state.asset_groups.get(group_name, {}).get("tokens", [])
            st.session_state[key] = group_tokens.copy()
        st.rerun()


def render_overview_tab(df: pd.DataFrame, df_filtered: pd.DataFrame, eth_price: float = 3150.0):
    """Render the overview tab with summary metrics and pie chart."""
    # Metrics row
    metrics = create_metric_cards(df_filtered)
    metrics_all = create_metric_cards(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "필터된 지갑 수",
            f"{metrics['wallet_count']:,}",
            delta=f"{metrics['wallet_count'] - metrics_all['wallet_count']:,}" if metrics['wallet_count'] != metrics_all['wallet_count'] else None
        )

    with col2:
        avg_eth = metrics['avg_eth_amount']
        avg_eth_usd = avg_eth * eth_price
        st.metric(
            "평균 ETH 수량",
            f"{avg_eth:.2f} (${avg_eth_usd:,.0f})",
            delta=f"{metrics['avg_eth_amount'] - metrics_all['avg_eth_amount']:.2f}" if abs(metrics['avg_eth_amount'] - metrics_all['avg_eth_amount']) > 0.01 else None
        )

    with col3:
        # Show median for ETH holders only (more meaningful)
        median_eth = metrics['median_eth_holders']
        median_eth_usd = median_eth * eth_price
        eth_holder_count = metrics['eth_holder_count']
        # Format based on magnitude
        if median_eth >= 1:
            eth_fmt = f"{median_eth:.2f}"
        elif median_eth >= 0.01:
            eth_fmt = f"{median_eth:.3f}"
        else:
            eth_fmt = f"{median_eth:.4f}"
        st.metric(
            f"중앙값 ETH (보유자 {eth_holder_count}명)",
            f"{eth_fmt} (${median_eth_usd:,.0f})",
            delta=f"{metrics['median_eth_holders'] - metrics_all['median_eth_holders']:.4f}" if abs(metrics['median_eth_holders'] - metrics_all['median_eth_holders']) > 0.0001 else None
        )

    with col4:
        st.metric(
            "평균 ETH 비율",
            f"{metrics['avg_eth_ratio']:.1f}%",
            delta=f"{metrics['avg_eth_ratio'] - metrics_all['avg_eth_ratio']:.1f}%" if abs(metrics['avg_eth_ratio'] - metrics_all['avg_eth_ratio']) > 0.1 else None
        )

    # Second metrics row
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("총 자산 가치", format_usd(metrics["total_value"]))

    with col6:
        st.metric("평균 지갑 나이", f"{metrics['avg_wallet_age']:.0f}일")

    with col7:
        st.metric("평균 프로토콜 수", f"{metrics['avg_protocol_count']:.1f}")

    with col8:
        st.metric("평균 ETH DeFi 비율", f"{metrics['avg_eth_defi_ratio']:.1f}%")

    st.divider()

    # Pie chart
    col_pie, col_proto = st.columns([1, 1])

    with col_pie:
        pie_fig = create_asset_pie_chart(df_filtered, st.session_state.asset_groups)
        st.plotly_chart(pie_fig, use_container_width=True, key="overview_pie")

    with col_proto:
        proto_fig = create_protocol_bar_chart(df_filtered, top_n=10)
        st.plotly_chart(proto_fig, use_container_width=True, key="overview_proto")


def render_distribution_tab(df_filtered: pd.DataFrame):
    """Render the distribution analysis tab."""
    col1, col2 = st.columns(2)

    with col1:
        eth_ratio_fig = create_eth_ratio_histogram(df_filtered)
        st.plotly_chart(eth_ratio_fig, use_container_width=True, key="dist_eth_ratio")

    with col2:
        wallet_age_fig = create_wallet_age_histogram(df_filtered)
        st.plotly_chart(wallet_age_fig, use_container_width=True, key="dist_wallet_age")

    col3, col4 = st.columns(2)

    with col3:
        value_fig = create_total_value_histogram(df_filtered)
        st.plotly_chart(value_fig, use_container_width=True, key="dist_value")

    with col4:
        proto_count_fig = create_protocol_count_histogram(df_filtered)
        st.plotly_chart(proto_count_fig, use_container_width=True, key="dist_proto_count")


def render_intersection_tab(df: pd.DataFrame):
    """Render the Venn diagram intersection analysis tab with fully customizable filters."""
    st.subheader("필터 교차 분석")
    st.caption("2~3개의 커스텀 필터를 정의하여 지갑 그룹 간 교차점을 분석합니다")

    # Get available columns and protocols for filter options
    numeric_columns = {
        "wallet_days": "지갑 나이 (일)",
        "total_usd": "총 자산 ($)",
        "eth_ratio": "ETH 비율",
        "usd_ratio": "USD 비율",
        "btc_ratio": "BTC 비율",
        "eth_amount": "ETH 수량",
        "btc_amount": "BTC 수량",
        "protocol_count": "프로토콜 수",
        "eth_defi_ratio": "ETH DeFi 비율",
        "usd_defi_ratio": "USD DeFi 비율",
        "btc_defi_ratio": "BTC DeFi 비율"
    }
    # Filter to only columns that exist
    numeric_columns = {k: v for k, v in numeric_columns.items() if k in df.columns}

    all_protocols = get_all_protocols(df)

    filter_configs = []

    st.markdown("### 🔧 필터 정의 (2~3개)")

    for i in range(3):
        with st.expander(f"필터 {i+1}", expanded=(i < 2)):
            col1, col2 = st.columns([1, 2])

            with col1:
                filter_type = st.selectbox(
                    "필터 유형",
                    options=["(사용 안함)", "숫자 범위", "프로토콜 사용"],
                    key=f"venn_type_{i}"
                )

            if filter_type == "숫자 범위":
                with col2:
                    column = st.selectbox(
                        "컬럼 선택",
                        options=list(numeric_columns.keys()),
                        format_func=lambda x: numeric_columns.get(x, x),
                        key=f"venn_col_{i}"
                    )

                # Get column stats for slider range
                col_min = float(df[column].min()) if len(df) > 0 else 0
                col_max = float(df[column].max()) if len(df) > 0 else 100

                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    range_type = st.selectbox(
                        "조건",
                        options=["최소값 이상", "최대값 이하", "범위"],
                        key=f"venn_range_type_{i}"
                    )

                # Adjust input based on column type (ratio vs absolute)
                is_ratio = "ratio" in column
                if is_ratio:
                    col_min, col_max = 0.0, 1.0
                    step = 0.05
                    format_str = "%.0f%%"
                    display_mult = 100
                else:
                    step = (col_max - col_min) / 100 if col_max > col_min else 1
                    format_str = None
                    display_mult = 1

                with col2:
                    if range_type == "최소값 이상":
                        if is_ratio:
                            val = st.slider(f"최소 (%)", 0, 100, 50, key=f"venn_min_{i}")
                            threshold = val / 100.0
                        else:
                            threshold = st.number_input("최소값", value=col_min, key=f"venn_min_{i}")
                        filter_configs.append(FilterConfig(
                            name=f"filter_{i}",
                            column=column,
                            filter_type="min",
                            value=threshold,
                            label=f"{numeric_columns[column]} ≥ {val if is_ratio else threshold:.0f}{'%' if is_ratio else ''}"
                        ))
                    elif range_type == "최대값 이하":
                        if is_ratio:
                            val = st.slider(f"최대 (%)", 0, 100, 50, key=f"venn_max_{i}")
                            threshold = val / 100.0
                        else:
                            threshold = st.number_input("최대값", value=col_max, key=f"venn_max_{i}")
                        filter_configs.append(FilterConfig(
                            name=f"filter_{i}",
                            column=column,
                            filter_type="max",
                            value=threshold,
                            label=f"{numeric_columns[column]} ≤ {val if is_ratio else threshold:.0f}{'%' if is_ratio else ''}"
                        ))
                    else:  # 범위
                        if is_ratio:
                            val_range = st.slider(f"범위 (%)", 0, 100, (25, 75), key=f"venn_range_{i}")
                            range_vals = (val_range[0] / 100.0, val_range[1] / 100.0)
                        else:
                            c1, c2 = st.columns(2)
                            with c1:
                                min_v = st.number_input("최소", value=col_min, key=f"venn_rmin_{i}")
                            with c2:
                                max_v = st.number_input("최대", value=col_max, key=f"venn_rmax_{i}")
                            range_vals = (min_v, max_v)
                            val_range = range_vals
                        filter_configs.append(FilterConfig(
                            name=f"filter_{i}",
                            column=column,
                            filter_type="range",
                            value=range_vals,
                            label=f"{numeric_columns[column]} {val_range[0]}-{val_range[1]}{'%' if is_ratio else ''}"
                        ))

            elif filter_type == "프로토콜 사용":
                with col2:
                    selected_protocols = st.multiselect(
                        "프로토콜 선택",
                        options=all_protocols,
                        key=f"venn_proto_{i}",
                        placeholder="프로토콜 검색..."
                    )

                if selected_protocols:
                    col1, col2 = st.columns(2)
                    with col1:
                        match_mode = st.radio(
                            "매칭 방식",
                            options=["ANY (하나라도)", "ALL (모두)"],
                            key=f"venn_proto_mode_{i}",
                            horizontal=True
                        )

                    label = f"{'&'.join(selected_protocols[:2])}{'...' if len(selected_protocols) > 2 else ''}"
                    filter_configs.append(FilterConfig(
                        name=f"filter_{i}",
                        column="protocols_used",
                        filter_type="contains",
                        value=selected_protocols,
                        label=label
                    ))

    st.divider()

    # Generate Venn diagram if we have 2+ filters
    if len(filter_configs) >= 2:
        filter_configs = filter_configs[:3]  # Max 3

        sets, labels = get_venn_sets_from_filters(df, filter_configs)

        col_venn, col_table = st.columns([2, 1])

        with col_venn:
            fig = create_venn_diagram(sets, labels)
            st.image(fig_to_image(fig))

        with col_table:
            st.write("**교집합 통계**")
            intersection_df = create_intersection_table(sets, labels)
            st.dataframe(intersection_df, width="stretch", hide_index=True)

            # Show percentage
            total_wallets = len(df)
            st.caption(f"전체 지갑 수: {total_wallets:,}")

    else:
        st.info("👆 위에서 최소 2개의 필터를 정의하세요.")


def render_wallet_details(address: str):
    """Render detailed wallet information."""
    from data_loader import parse_usd_value
    from collections import Counter

    details = load_wallet_details(address)

    if not details:
        st.warning(f"상세 정보를 찾을 수 없습니다: {address}")
        return

    portfolio = details.get("portfolio", {})
    tokens = portfolio.get("tokens", [])
    positions = portfolio.get("positions", [])

    # Header with DeBank link
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📋 {address[:10]}...{address[-8:]}")
    with col2:
        st.link_button("🔗 DeBank에서 보기", f"https://debank.com/profile/{address}")

    # Summary metrics - Row 1
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총 자산", portfolio.get("totalValue", "$0"))
    with col2:
        st.metric("지갑 나이", portfolio.get("walletLifetime", "N/A"))
    with col3:
        st.metric("토큰 수", len(tokens))
    with col4:
        st.metric("포지션 수", len(positions))

    # Calculate asset ratios for this wallet
    total_value = parse_usd_value(portfolio.get("totalValue", "0"))
    eth_tokens = {"ETH", "WETH", "STETH", "WSTETH", "RETH", "CBETH", "METH", "EETH"}
    btc_tokens = {"BTC", "WBTC", "TBTC", "CBBTC"}
    usd_tokens = {"USDT", "USDC", "DAI", "FRAX", "TUSD", "BUSD", "LUSD"}

    eth_value = sum(parse_usd_value(t.get("usdValue", "0")) for t in tokens if t.get("symbol", "").upper() in eth_tokens)
    btc_value = sum(parse_usd_value(t.get("usdValue", "0")) for t in tokens if t.get("symbol", "").upper() in btc_tokens)
    usd_value = sum(parse_usd_value(t.get("usdValue", "0")) for t in tokens if t.get("symbol", "").upper() in usd_tokens)

    # Summary metrics - Row 2 (Asset Ratios)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        eth_ratio = (eth_value / total_value * 100) if total_value > 0 else 0
        st.metric("ETH 비율", f"{eth_ratio:.1f}%")
    with col2:
        btc_ratio = (btc_value / total_value * 100) if total_value > 0 else 0
        st.metric("BTC 비율", f"{btc_ratio:.1f}%")
    with col3:
        usd_ratio = (usd_value / total_value * 100) if total_value > 0 else 0
        st.metric("USD 비율", f"{usd_ratio:.1f}%")
    with col4:
        other_ratio = 100 - eth_ratio - btc_ratio - usd_ratio
        st.metric("기타 비율", f"{other_ratio:.1f}%")

    # Top tokens and protocols summary
    col1, col2 = st.columns(2)

    with col1:
        # Top 5 tokens by value
        if tokens:
            token_vals = [(t.get("symbol", "?"), parse_usd_value(t.get("usdValue", "0"))) for t in tokens]
            token_vals.sort(key=lambda x: x[1], reverse=True)
            top_5_tokens = token_vals[:5]
            st.markdown("**🪙 Top 5 토큰 볼륨:**")
            for symbol, val in top_5_tokens:
                st.write(f"  • {symbol}: ${val:,.0f}")

    with col2:
        # Top protocols by value
        if positions:
            protocol_vals = Counter()
            for pos in positions:
                protocol = pos.get("protocol", "Unknown")
                val = parse_usd_value(pos.get("usdValue", "0"))
                protocol_vals[protocol] += val
            top_5_protocols = protocol_vals.most_common(5)
            st.markdown("**🏦 Top 5 프로토콜 볼륨:**")
            for protocol, val in top_5_protocols:
                st.write(f"  • {protocol}: ${val:,.0f}")

    st.divider()

    # Tokens table
    if tokens:
        st.markdown("#### 💰 토큰 보유 상세")
        st.caption("usdValue = 토큰 볼륨")
        tokens_df = pd.DataFrame(tokens)
        if not tokens_df.empty:
            display_cols = [c for c in ["symbol", "amount", "usdValue", "chain"] if c in tokens_df.columns]
            st.dataframe(tokens_df[display_cols], width="stretch", hide_index=True)

    # DeFi positions table
    if positions:
        st.markdown("#### 🏦 DeFi 포지션 상세")
        st.caption("usdValue = 프로토콜 볼륨")
        positions_df = pd.DataFrame(positions)
        if not positions_df.empty:
            display_cols = [c for c in ["protocol", "category", "token", "usdValue", "chain"] if c in positions_df.columns]
            st.dataframe(positions_df[display_cols], width="stretch", hide_index=True)


def render_bias_analysis_tab(df_filtered: pd.DataFrame, data_dir):
    """Render the bias analysis tab with aggregated insights."""
    import json
    import numpy as np
    from collections import Counter
    from data_loader import parse_usd_value
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("🎯 편향 분석")
    st.caption("필터된 지갑들의 공통 특성 및 편향성 분석")

    if len(df_filtered) == 0:
        st.warning("분석할 지갑이 없습니다.")
        return

    # Collect aggregated data from all filtered wallets
    all_tokens = Counter()  # token symbol -> count of wallets holding it
    all_protocols = Counter()  # protocol -> count of wallets using it
    chain_usage = Counter()  # chain -> count
    category_usage = Counter()  # defi category -> count
    token_values = Counter()  # token symbol -> total USD value
    protocol_values = Counter()  # protocol -> total USD value

    with st.spinner("지갑 데이터 분석 중..."):
        for address in df_filtered["address"]:
            json_path = data_dir / f"{address.lower()}.json"
            if not json_path.exists():
                continue
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                portfolio = data.get("portfolio", {})

                # Tokens
                wallet_tokens = set()
                for token in portfolio.get("tokens", []):
                    symbol = token.get("symbol", "").upper()
                    if symbol and not symbol.startswith("0X"):
                        wallet_tokens.add(symbol)
                        usd_val = parse_usd_value(token.get("usdValue", "0"))
                        token_values[symbol] += usd_val
                        chain = token.get("chain", "unknown")
                        chain_usage[chain] += 1

                for t in wallet_tokens:
                    all_tokens[t] += 1

                # Protocols & Categories
                wallet_protocols = set()
                for pos in portfolio.get("positions", []):
                    protocol = pos.get("protocol", "").strip()
                    if protocol and not protocol.startswith("0x"):
                        wallet_protocols.add(protocol)
                        usd_val = parse_usd_value(pos.get("usdValue", "0"))
                        protocol_values[protocol] += usd_val

                    category = pos.get("category", "").strip().lower()
                    if category:
                        category_usage[category] += 1

                    chain = pos.get("chain", "unknown")
                    chain_usage[chain] += 1

                for p in wallet_protocols:
                    all_protocols[p] += 1

            except:
                continue

    # === Row 0: Basic Distribution (moved from 분포 tab) ===
    st.markdown("### 📈 기본 분포")
    col1, col2 = st.columns(2)

    with col1:
        # Wallet age distribution
        wallet_age_fig = create_wallet_age_histogram(df_filtered)
        st.plotly_chart(wallet_age_fig, use_container_width=True, key="bias_wallet_age")

    with col2:
        # Total value distribution
        value_fig = create_total_value_histogram(df_filtered)
        st.plotly_chart(value_fig, use_container_width=True, key="bias_value")

    col3, col4 = st.columns(2)

    with col3:
        # Protocol count distribution
        proto_count_fig = create_protocol_count_histogram(df_filtered)
        st.plotly_chart(proto_count_fig, use_container_width=True, key="bias_proto_count")

    with col4:
        # ETH ratio distribution (from original distribution tab)
        eth_ratio_fig = create_eth_ratio_histogram(df_filtered)
        st.plotly_chart(eth_ratio_fig, use_container_width=True, key="bias_eth_ratio")

    st.divider()

    # === Row 1: ETH & BTC Amount Distribution ===
    st.markdown("### 🪙 수량 분포 (ETH / BTC)")
    col1, col2 = st.columns(2)

    with col1:
        # ETH amount distribution with range tooltip
        if "eth_amount" in df_filtered.columns:
            df_eth = df_filtered[df_filtered["eth_amount"] > 0].copy()
            if len(df_eth) > 0:
                # Create histogram bins manually for range tooltip
                eth_values = df_eth["eth_amount"].values
                counts, bin_edges = np.histogram(np.log10(eth_values), bins=25)

                # Convert back to actual values for display
                bin_starts = 10 ** bin_edges[:-1]
                bin_ends = 10 ** bin_edges[1:]

                # Format bin labels
                def format_eth(v):
                    if v >= 1000:
                        return f"{v/1000:.1f}K"
                    elif v >= 1:
                        return f"{v:.1f}"
                    else:
                        return f"{v:.3f}"

                hover_texts = [f"<b>ETH 수량</b>: {format_eth(s)}~{format_eth(e)}<br><b>지갑 수</b>: {c}"
                              for s, e, c in zip(bin_starts, bin_ends, counts)]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[(np.log10(s) + np.log10(e)) / 2 for s, e in zip(bin_starts, bin_ends)],
                    y=counts,
                    width=[(np.log10(e) - np.log10(s)) * 0.9 for s, e in zip(bin_starts, bin_ends)],
                    marker_color="#627EEA",
                    hovertext=hover_texts,
                    hoverinfo="text"
                ))

                # Custom tick labels
                tick_vals = [np.log10(v) for v in [0.1, 1, 10, 100, 1000, 10000] if 0.1 <= v <= max(eth_values) * 1.1]
                tick_text = [format_eth(10**v) for v in tick_vals]

                fig.update_layout(
                    title="ETH 수량 분포 (로그 스케일)",
                    height=350,
                    xaxis_title="ETH 수량",
                    yaxis_title="지갑 수",
                    xaxis=dict(tickvals=tick_vals, ticktext=tick_text),
                    bargap=0.1
                )
                st.plotly_chart(fig, use_container_width=True, key="bias_eth_amount")
            else:
                st.info("ETH를 보유한 지갑이 없습니다.")

    with col2:
        # BTC amount distribution with range tooltip
        if "btc_amount" in df_filtered.columns:
            df_btc = df_filtered[df_filtered["btc_amount"] > 0].copy()
            if len(df_btc) > 0:
                # Create histogram bins manually for range tooltip
                btc_values = df_btc["btc_amount"].values
                counts, bin_edges = np.histogram(np.log10(btc_values), bins=25)

                # Convert back to actual values for display
                bin_starts = 10 ** bin_edges[:-1]
                bin_ends = 10 ** bin_edges[1:]

                # Format bin labels for BTC
                def format_btc(v):
                    if v >= 1:
                        return f"{v:.2f}"
                    elif v >= 0.01:
                        return f"{v:.3f}"
                    else:
                        return f"{v:.4f}"

                hover_texts = [f"<b>BTC 수량</b>: {format_btc(s)}~{format_btc(e)}<br><b>지갑 수</b>: {c}"
                              for s, e, c in zip(bin_starts, bin_ends, counts)]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[(np.log10(s) + np.log10(e)) / 2 for s, e in zip(bin_starts, bin_ends)],
                    y=counts,
                    width=[(np.log10(e) - np.log10(s)) * 0.9 for s, e in zip(bin_starts, bin_ends)],
                    marker_color="#F7931A",
                    hovertext=hover_texts,
                    hoverinfo="text"
                ))

                # Custom tick labels
                tick_vals = [np.log10(v) for v in [0.001, 0.01, 0.1, 1, 10, 100] if 0.001 <= v <= max(btc_values) * 1.1]
                tick_text = [format_btc(10**v) for v in tick_vals]

                fig.update_layout(
                    title="BTC 수량 분포 (로그 스케일)",
                    height=350,
                    xaxis_title="BTC 수량",
                    yaxis_title="지갑 수",
                    xaxis=dict(tickvals=tick_vals, ticktext=tick_text),
                    bargap=0.1
                )
                st.plotly_chart(fig, use_container_width=True, key="bias_btc_amount")
            else:
                st.info("BTC를 보유한 지갑이 없습니다.")
        else:
            st.info("BTC 수량 데이터가 없습니다. 데이터를 다시 로드해주세요.")

    st.divider()

    # === Row 2: Asset Ratio Distribution ===
    st.markdown("### 📊 자산 비율 분포")
    col1, col2, col3 = st.columns(3)

    with col1:
        # ETH ratio distribution with range tooltip
        if "eth_ratio" in df_filtered.columns:
            ratio_values = (df_filtered["eth_ratio"] * 100).values
            counts, bin_edges = np.histogram(ratio_values, bins=20, range=(0, 100))
            bin_starts = bin_edges[:-1]
            bin_ends = bin_edges[1:]

            hover_texts = [f"<b>ETH 비율</b>: {int(s)}%~{int(e)}%<br><b>지갑 수</b>: {c}"
                          for s, e, c in zip(bin_starts, bin_ends, counts)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[(s + e) / 2 for s, e in zip(bin_starts, bin_ends)],
                y=counts,
                width=4.5,
                marker_color="#627EEA",
                hovertext=hover_texts,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="ETH 비율 분포",
                height=300,
                xaxis=dict(range=[0, 100], ticksuffix="%", title="ETH 비율 (%)"),
                yaxis_title="지갑 수",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True, key="bias_eth_ratio_dist")

    with col2:
        # USD ratio distribution with range tooltip
        if "usd_ratio" in df_filtered.columns:
            ratio_values = (df_filtered["usd_ratio"] * 100).values
            counts, bin_edges = np.histogram(ratio_values, bins=20, range=(0, 100))
            bin_starts = bin_edges[:-1]
            bin_ends = bin_edges[1:]

            hover_texts = [f"<b>USD 비율</b>: {int(s)}%~{int(e)}%<br><b>지갑 수</b>: {c}"
                          for s, e, c in zip(bin_starts, bin_ends, counts)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[(s + e) / 2 for s, e in zip(bin_starts, bin_ends)],
                y=counts,
                width=4.5,
                marker_color="#26A17B",
                hovertext=hover_texts,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="USD 비율 분포",
                height=300,
                xaxis=dict(range=[0, 100], ticksuffix="%", title="USD 비율 (%)"),
                yaxis_title="지갑 수",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True, key="bias_usd_ratio_dist")

    with col3:
        # BTC ratio distribution with range tooltip
        if "btc_ratio" in df_filtered.columns:
            ratio_values = (df_filtered["btc_ratio"] * 100).values
            counts, bin_edges = np.histogram(ratio_values, bins=20, range=(0, 100))
            bin_starts = bin_edges[:-1]
            bin_ends = bin_edges[1:]

            hover_texts = [f"<b>BTC 비율</b>: {int(s)}%~{int(e)}%<br><b>지갑 수</b>: {c}"
                          for s, e, c in zip(bin_starts, bin_ends, counts)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[(s + e) / 2 for s, e in zip(bin_starts, bin_ends)],
                y=counts,
                width=4.5,
                marker_color="#F7931A",
                hovertext=hover_texts,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="BTC 비율 분포",
                height=300,
                xaxis=dict(range=[0, 100], ticksuffix="%", title="BTC 비율 (%)"),
                yaxis_title="지갑 수",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True, key="bias_btc_ratio_dist")

    st.divider()

    # === Row 2.5: DeFi Asset Ratio Distribution ===
    st.markdown("### 🏦 DeFi 자산 비율 분포")
    st.caption("DeFi 포지션 중 해당 자산 그룹의 비율 (토큰 기준)")
    col1, col2, col3 = st.columns(3)

    with col1:
        # ETH DeFi ratio distribution
        if "eth_defi_ratio" in df_filtered.columns:
            ratio_values = (df_filtered["eth_defi_ratio"] * 100).values
            counts, bin_edges = np.histogram(ratio_values, bins=20, range=(0, 100))
            bin_starts = bin_edges[:-1]
            bin_ends = bin_edges[1:]

            hover_texts = [f"<b>ETH DeFi 비율</b>: {int(s)}%~{int(e)}%<br><b>지갑 수</b>: {c}"
                          for s, e, c in zip(bin_starts, bin_ends, counts)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[(s + e) / 2 for s, e in zip(bin_starts, bin_ends)],
                y=counts,
                width=4.5,
                marker_color="#627EEA",
                hovertext=hover_texts,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="ETH DeFi 비율 분포",
                height=300,
                xaxis=dict(range=[0, 100], ticksuffix="%", title="ETH DeFi 비율 (%)"),
                yaxis_title="지갑 수",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True, key="bias_eth_defi_ratio")

    with col2:
        # USD DeFi ratio distribution
        if "usd_defi_ratio" in df_filtered.columns:
            ratio_values = (df_filtered["usd_defi_ratio"] * 100).values
            counts, bin_edges = np.histogram(ratio_values, bins=20, range=(0, 100))
            bin_starts = bin_edges[:-1]
            bin_ends = bin_edges[1:]

            hover_texts = [f"<b>USD DeFi 비율</b>: {int(s)}%~{int(e)}%<br><b>지갑 수</b>: {c}"
                          for s, e, c in zip(bin_starts, bin_ends, counts)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[(s + e) / 2 for s, e in zip(bin_starts, bin_ends)],
                y=counts,
                width=4.5,
                marker_color="#26A17B",
                hovertext=hover_texts,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="USD DeFi 비율 분포",
                height=300,
                xaxis=dict(range=[0, 100], ticksuffix="%", title="USD DeFi 비율 (%)"),
                yaxis_title="지갑 수",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True, key="bias_usd_defi_ratio")

    with col3:
        # BTC DeFi ratio distribution
        if "btc_defi_ratio" in df_filtered.columns:
            ratio_values = (df_filtered["btc_defi_ratio"] * 100).values
            counts, bin_edges = np.histogram(ratio_values, bins=20, range=(0, 100))
            bin_starts = bin_edges[:-1]
            bin_ends = bin_edges[1:]

            hover_texts = [f"<b>BTC DeFi 비율</b>: {int(s)}%~{int(e)}%<br><b>지갑 수</b>: {c}"
                          for s, e, c in zip(bin_starts, bin_ends, counts)]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[(s + e) / 2 for s, e in zip(bin_starts, bin_ends)],
                y=counts,
                width=4.5,
                marker_color="#F7931A",
                hovertext=hover_texts,
                hoverinfo="text"
            ))
            fig.update_layout(
                title="BTC DeFi 비율 분포",
                height=300,
                xaxis=dict(range=[0, 100], ticksuffix="%", title="BTC DeFi 비율 (%)"),
                yaxis_title="지갑 수",
                bargap=0.1
            )
            st.plotly_chart(fig, use_container_width=True, key="bias_btc_defi_ratio")

    st.divider()

    # === Row 3: Top Tokens & Protocols ===
    st.markdown("### 🏆 토큰 보유 현황 & 프로토콜 사용 현황")
    st.caption("💡 토큰 볼륨 = 필터된 지갑들의 해당 토큰 총 보유 금액 합계 | 프로토콜 볼륨 = 필터된 지갑들의 해당 프로토콜 예치 금액 합계")
    col1, col2 = st.columns(2)

    wallet_count = len(df_filtered)

    with col1:
        st.markdown("#### 🪙 토큰 보유 현황")
        top_tokens = all_tokens.most_common(15)
        if top_tokens:
            tokens_df = pd.DataFrame(top_tokens, columns=["토큰", "지갑 수"])
            tokens_df["보유율"] = (tokens_df["지갑 수"] / wallet_count * 100).round(1)
            tokens_df["토큰 볼륨"] = tokens_df["토큰"].apply(lambda t: token_values.get(t, 0))

            # Format volume
            def fmt_usd(v):
                if v >= 1e9: return f"${v/1e9:.1f}B"
                if v >= 1e6: return f"${v/1e6:.1f}M"
                if v >= 1e3: return f"${v/1e3:.0f}K"
                return f"${v:.0f}"

            fig = go.Figure(go.Bar(
                x=tokens_df["보유율"],
                y=tokens_df["토큰"],
                orientation="h",
                text=tokens_df.apply(lambda r: f"{r['보유율']:.1f}% ({r['지갑 수']}개) · 토큰 볼륨 {fmt_usd(r['토큰 볼륨'])}", axis=1),
                textposition="outside",
                marker_color="#627EEA",
                hovertemplate="<b>%{y}</b><br>보유 지갑: %{customdata[0]}개 (%{x:.1f}%)<br>토큰 볼륨: %{customdata[1]}<extra></extra>",
                customdata=tokens_df.apply(lambda r: [r['지갑 수'], fmt_usd(r['토큰 볼륨'])], axis=1).tolist()
            ))
            fig.update_layout(
                title=f"토큰별 보유 현황 (총 {wallet_count}개 지갑)",
                xaxis_title="보유율 (%)",
                yaxis=dict(autorange="reversed"),
                height=450,
                margin=dict(l=100, r=140)
            )
            st.plotly_chart(fig, use_container_width=True, key="bias_token_chart")

    with col2:
        st.markdown("#### 🏦 프로토콜 사용 현황")
        top_protocols = all_protocols.most_common(15)
        if top_protocols:
            protocols_df = pd.DataFrame(top_protocols, columns=["프로토콜", "지갑 수"])
            protocols_df["사용율"] = (protocols_df["지갑 수"] / wallet_count * 100).round(1)
            protocols_df["프로토콜 볼륨"] = protocols_df["프로토콜"].apply(lambda p: protocol_values.get(p, 0))

            def fmt_usd(v):
                if v >= 1e9: return f"${v/1e9:.1f}B"
                if v >= 1e6: return f"${v/1e6:.1f}M"
                if v >= 1e3: return f"${v/1e3:.0f}K"
                return f"${v:.0f}"

            fig = go.Figure(go.Bar(
                x=protocols_df["사용율"],
                y=protocols_df["프로토콜"],
                orientation="h",
                text=protocols_df.apply(lambda r: f"{r['사용율']:.1f}% ({r['지갑 수']}개) · 프로토콜 볼륨 {fmt_usd(r['프로토콜 볼륨'])}", axis=1),
                textposition="outside",
                marker_color="#8B5CF6",
                hovertemplate="<b>%{y}</b><br>사용 지갑: %{customdata[0]}개 (%{x:.1f}%)<br>프로토콜 볼륨: %{customdata[1]}<extra></extra>",
                customdata=protocols_df.apply(lambda r: [r['지갑 수'], fmt_usd(r['프로토콜 볼륨'])], axis=1).tolist()
            ))
            fig.update_layout(
                title=f"프로토콜별 사용 현황 (총 {wallet_count}개 지갑)",
                xaxis_title="사용율 (%)",
                yaxis=dict(autorange="reversed"),
                height=450,
                margin=dict(l=120, r=140)
            )
            st.plotly_chart(fig, use_container_width=True, key="bias_protocol_chart")

    st.divider()

    # === Row 4: Chain & Category Distribution ===
    st.markdown("### 🔗 체인 & DeFi 카테고리 분포")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ⛓️ 체인별 사용 현황")
        if chain_usage:
            top_chains = chain_usage.most_common(10)
            chains_df = pd.DataFrame(top_chains, columns=["체인", "사용 횟수"])

            fig = px.pie(
                chains_df, values="사용 횟수", names="체인",
                title="체인 분포",
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="bias_chain_pie")

    with col2:
        st.markdown("#### 📂 DeFi 카테고리 현황")
        if category_usage:
            top_categories = category_usage.most_common(10)
            cat_df = pd.DataFrame(top_categories, columns=["카테고리", "사용 횟수"])

            fig = px.pie(
                cat_df, values="사용 횟수", names="카테고리",
                title="DeFi 카테고리 분포",
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="bias_category_pie")

    st.divider()

    # === Row 5: Summary Statistics ===
    st.markdown("### 📈 요약 통계")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("고유 토큰 수", f"{len(all_tokens):,}")
    with col2:
        st.metric("고유 프로토콜 수", f"{len(all_protocols):,}")
    with col3:
        st.metric("활성 체인 수", f"{len(chain_usage):,}")
    with col4:
        st.metric("DeFi 카테고리 수", f"{len(category_usage):,}")

    # Top overlapping stats
    if top_tokens:
        most_common_token = top_tokens[0]
        st.info(f"🪙 **가장 많이 보유된 토큰**: {most_common_token[0]} - {most_common_token[1]}개 지갑 ({most_common_token[1]/wallet_count*100:.1f}%)")

    if top_protocols:
        most_common_protocol = top_protocols[0]
        st.info(f"🏦 **가장 많이 사용된 프로토콜**: {most_common_protocol[0]} - {most_common_protocol[1]}개 지갑 ({most_common_protocol[1]/wallet_count*100:.1f}%)")


def render_table_tab(df_filtered: pd.DataFrame):
    """Render the data table tab."""
    st.subheader("지갑 상세 데이터")

    # Column selection
    all_columns = df_filtered.columns.tolist()
    display_columns = ["address", "vault_name", "vault_share", "total_usd", "wallet_days", "eth_amount",
                       "btc_amount", "eth_ratio", "protocol_count",
                       "eth_defi_ratio", "usd_defi_ratio", "btc_defi_ratio"]
    display_columns = [c for c in display_columns if c in all_columns]

    selected_cols = st.multiselect(
        "표시할 컬럼",
        options=all_columns,
        default=display_columns
    )

    if not selected_cols:
        selected_cols = display_columns

    # Sort options
    col_sort, col_order = st.columns([2, 1])
    with col_sort:
        sort_by = st.selectbox("정렬 기준", options=selected_cols)
    with col_order:
        ascending = st.selectbox("순서", options=["내림차순", "오름차순"]) == "오름차순"

    # Display table with selection
    df_display = df_filtered[selected_cols].sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    # Use data_editor for row selection
    event = st.dataframe(
        df_display,
        width="stretch",
        height=400,
        column_config={
            "address": st.column_config.TextColumn("Address", width="medium"),
            "vault_name": st.column_config.TextColumn("볼트", width="medium"),
            "vault_share": st.column_config.NumberColumn("볼트 지분율", format="%.2f%%"),
            "total_usd": st.column_config.NumberColumn("Total USD", format="$%.2f"),
            "eth_ratio": st.column_config.ProgressColumn("ETH Ratio", min_value=0, max_value=1),
            "eth_defi_ratio": st.column_config.ProgressColumn("ETH DeFi Ratio", min_value=0, max_value=1),
            "usd_defi_ratio": st.column_config.ProgressColumn("USD DeFi Ratio", min_value=0, max_value=1),
            "btc_defi_ratio": st.column_config.ProgressColumn("BTC DeFi Ratio", min_value=0, max_value=1),
        },
        selection_mode="single-row",
        on_select="rerun",
        key="wallet_table"
    )

    # Show details for selected row
    if event and event.selection and event.selection.rows:
        selected_idx = event.selection.rows[0]
        selected_address = df_display.iloc[selected_idx]["address"]

        st.divider()
        render_wallet_details(selected_address)

    # Download button
    st.divider()
    csv = df_display.to_csv(index=False)
    st.download_button(
        "📥 CSV 다운로드",
        csv,
        file_name="eth_stakers_filtered.csv",
        mime="text/csv"
    )


def render_data_source_tab(df: pd.DataFrame):
    """Render the data source documentation tab."""
    st.subheader("📚 데이터 출처 및 수집 방법")

    # Data summary
    st.markdown("### 📊 데이터 요약")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 지갑 수", f"{len(df):,}")
    with col2:
        if "vault_name" in df.columns:
            st.metric("데이터 소스 수", f"{df['vault_name'].nunique()}")
    with col3:
        st.metric("수집 시점", "2025년 12월")

    st.divider()

    # Vault distribution
    if "vault_name" in df.columns:
        st.markdown("### 🏛️ 볼트별 지갑 분포")
        vault_counts = df["vault_name"].value_counts()
        for vault, count in vault_counts.items():
            pct = count / len(df) * 100
            st.write(f"- **{vault}**: {count:,}개 ({pct:.1f}%)")

    st.divider()

    # Data sources description
    st.markdown("### 📍 데이터 소스 설명")

    st.info("**공통 필터 조건**: 모든 소스에서 `0.05 ETH ≤ 잔고 < 1000 ETH` 범위의 주소만 추출")

    st.markdown("""
    | 소스 | 체인 | 프로토콜/토큰 | 컨트랙트 | 샘플링 |
    |------|------|--------------|----------|--------|
    | **lido_steth** | Ethereum | Lido stETH | `0xae7ab...7fe84` | 랜덤 1,000 |
    | **aave_mainnet** | Ethereum | AAVE V3 aWETH | `0x4d5F4...14E8` | 랜덤 1,000 |
    | **base_aave** | Base | AAVE V3 aWETH | `0xD4a0e...bb7` | 랜덤 1,000 |
    | **moonwell** | Base | Moonwell Flagship ETH | `0xa0E43...ff1` | 랜덤 1,000 |
    | **high_growth_eth** | Ethereum | High Growth ETH Vault | `0xc824A...9fD` | 전체 |
    """)

    st.divider()

    # Dune Queries
    st.markdown("### 🔍 Dune Analytics 쿼리")
    st.caption("모든 쿼리는 0.05 ETH 이상 ~ 1000 ETH 미만 잔고를 가진 주소를 대상으로 합니다.")

    with st.expander("🔷 1. Lido stETH 홀더 쿼리", expanded=False):
        st.code("""-- Lido stETH (0.05 <= balance < 1000, random 1000)
WITH steth_balances AS (
  SELECT "from" AS address, -CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_ethereum.evt_Transfer
  WHERE contract_address = 0xae7ab96520de3a18e5e111b5eaab095312d7fe84
  UNION ALL
  SELECT "to" AS address, CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_ethereum.evt_Transfer
  WHERE contract_address = 0xae7ab96520de3a18e5e111b5eaab095312d7fe84
),
holder_totals AS (
  SELECT address, SUM(amount) AS balance
  FROM steth_balances
  WHERE address != 0x0000000000000000000000000000000000000000
  GROUP BY address
  HAVING SUM(amount) >= 0.05 AND SUM(amount) < 1000
),
total_supply AS (SELECT SUM(balance) AS total FROM holder_totals)

SELECT address, balance, 'lido_steth' AS source,
  balance / (SELECT total FROM total_supply) * 100 AS vault_pct
FROM holder_totals ORDER BY RAND() LIMIT 1000""", language="sql")

    with st.expander("🔷 2. AAVE Mainnet aWETH 홀더 쿼리", expanded=False):
        st.code("""-- Aave Mainnet ETH (0.05 <= balance < 1000, random 1000)
WITH balances AS (
  SELECT "from" AS address, -CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_ethereum.evt_Transfer
  WHERE contract_address = 0x4d5F47FA6A74757f35C14fD3a6Ef8E3C9BC514E8
  UNION ALL
  SELECT "to" AS address, CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_ethereum.evt_Transfer
  WHERE contract_address = 0x4d5F47FA6A74757f35C14fD3a6Ef8E3C9BC514E8
),
holder_totals AS (
  SELECT address, SUM(amount) AS balance
  FROM balances
  WHERE address != 0x0000000000000000000000000000000000000000
  GROUP BY address
  HAVING SUM(amount) >= 0.05 AND SUM(amount) < 1000
),
total_supply AS (SELECT SUM(balance) AS total FROM holder_totals)

SELECT address, balance, 'aave_mainnet' AS source,
  balance / (SELECT total FROM total_supply) * 100 AS vault_pct
FROM holder_totals ORDER BY RAND() LIMIT 1000""", language="sql")

    with st.expander("🔷 3. Base AAVE aWETH 홀더 쿼리", expanded=False):
        st.code("""-- Base Aave ETH (0.05 <= balance < 1000, random 1000)
WITH balances AS (
  SELECT "from" AS address, -CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_base.evt_Transfer
  WHERE contract_address = 0xD4a0e0b9149BCee3C920d2E00b5dE09138fd8bb7
  UNION ALL
  SELECT "to" AS address, CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_base.evt_Transfer
  WHERE contract_address = 0xD4a0e0b9149BCee3C920d2E00b5dE09138fd8bb7
),
holder_totals AS (
  SELECT address, SUM(amount) AS balance
  FROM balances
  WHERE address != 0x0000000000000000000000000000000000000000
  GROUP BY address
  HAVING SUM(amount) >= 0.05 AND SUM(amount) < 1000
),
total_supply AS (SELECT SUM(balance) AS total FROM holder_totals)

SELECT address, balance, 'base_aave' AS source,
  balance / (SELECT total FROM total_supply) * 100 AS vault_pct
FROM holder_totals ORDER BY RAND() LIMIT 1000""", language="sql")

    with st.expander("🔷 4. High Growth ETH 볼트 쿼리", expanded=False):
        st.code("""-- High Growth ETH (0.05 <= balance < 1000, ALL - no random sampling)
WITH balances AS (
  SELECT "from" AS address, -CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_ethereum.evt_Transfer
  WHERE contract_address = 0xc824A08dB624942c5E5F330d56530cD1598859fD
  UNION ALL
  SELECT "to" AS address, CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_ethereum.evt_Transfer
  WHERE contract_address = 0xc824A08dB624942c5E5F330d56530cD1598859fD
),
holder_totals AS (
  SELECT address, SUM(amount) AS balance
  FROM balances
  WHERE address != 0x0000000000000000000000000000000000000000
  GROUP BY address
  HAVING SUM(amount) >= 0.05 AND SUM(amount) < 1000
),
total_supply AS (SELECT SUM(balance) AS total FROM holder_totals)

SELECT address, balance, 'high_growth_eth' AS source,
  balance / (SELECT total FROM total_supply) * 100 AS vault_pct
FROM holder_totals""", language="sql")

    with st.expander("🔷 5. Moonwell Flagship ETH 쿼리", expanded=False):
        st.code("""-- Moonwell Flagship ETH (0.05 <= balance < 1000, random 1000)
WITH balances AS (
  SELECT "from" AS address, -CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_base.evt_Transfer
  WHERE contract_address = 0xa0E430870c4604CcfC7B38Ca7845B1FF653D0ff1
  UNION ALL
  SELECT "to" AS address, CAST(value AS DOUBLE) / 1e18 AS amount
  FROM erc20_base.evt_Transfer
  WHERE contract_address = 0xa0E430870c4604CcfC7B38Ca7845B1FF653D0ff1
),
holder_totals AS (
  SELECT address, SUM(amount) AS balance
  FROM balances
  WHERE address != 0x0000000000000000000000000000000000000000
  GROUP BY address
  HAVING SUM(amount) >= 0.05 AND SUM(amount) < 1000
),
total_supply AS (SELECT SUM(balance) AS total FROM holder_totals)

SELECT address, balance, 'moonwell' AS source,
  balance / (SELECT total FROM total_supply) * 100 AS vault_pct
FROM holder_totals ORDER BY RAND() LIMIT 1000""", language="sql")

    st.divider()

    # Data collection method
    st.markdown("### 🔄 데이터 수집 흐름")

    st.markdown("""
    ```
    Dune Analytics (온체인 데이터)
        │
        ▼
    주소 목록 추출 (source, balance, vault_pct)
        │
        ▼
    DeBank API (포트폴리오 조회)
        │
        ├── 토큰 보유량
        ├── DeFi 포지션
        └── 프로토콜 사용 현황
        │
        ▼
    대시보드 분석
    ```
    """)

    st.divider()

    # Notes
    st.markdown("### ⚠️ 데이터 주의사항")
    st.warning("""
    - **수집 시점**: 2025년 12월 기준 데이터입니다. 실시간 데이터가 아닙니다.
    - **ETH 기준가**: $3,150 USD (수집 당시 기준)
    - **중복 주소**: 여러 소스에서 동일 주소가 포함될 수 있습니다.
    - **vaultPct**: 해당 볼트/프로토콜에서 해당 주소가 차지하는 비율입니다.
    """)

    # Links
    st.markdown("### 🔗 관련 링크")
    st.markdown("""
    - [DeBank](https://debank.com/) - 포트폴리오 데이터 소스
    - [Dune Analytics](https://dune.com/) - 온체인 데이터 쿼리
    - [Lido Finance](https://lido.fi/) - stETH 스테이킹
    - [AAVE](https://aave.com/) - DeFi 렌딩 프로토콜
    - [Moonwell](https://moonwell.fi/) - Base 체인 렌딩
    """)


ETH_PRICE_USD = 3150.0  # Data collection reference price (2025-12)


def main():
    """Main application entry point."""
    init_session_state()

    st.title("📊 ETH Stakers Portfolio Dashboard")
    st.caption("이더리움 스테이커들의 포트폴리오 분석 및 행동 편향성 탐색")

    # Load data with DeFi tokens from session state
    try:
        with st.spinner("데이터 로딩 중..."):
            eth_defi_tokens = tuple(st.session_state.get("eth_defi_tokens", []))
            df = load_data(eth_defi_tokens=eth_defi_tokens)
    except FileNotFoundError as e:
        st.error(f"데이터 디렉토리를 찾을 수 없습니다: {e}")
        st.info("data/results/ 폴더에 JSON 파일이 있는지 확인하세요.")
        return
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return

    # Get all protocols for filter
    all_protocols = get_all_protocols(df)

    # Render sidebar and get filter values
    filters = render_sidebar(df, all_protocols)

    # Apply filters
    df_filtered = apply_filters(
        df,
        wallet_days_range=filters["wallet_days_range"],
        total_usd_range=filters["total_usd_range"],
        eth_amount_range=filters["eth_amount_range"],
        eth_ratio_range=filters["eth_ratio_range"],
        usd_ratio_range=filters["usd_ratio_range"],
        btc_ratio_range=filters["btc_ratio_range"],
        eth_defi_ratio_range=filters["eth_defi_ratio_range"],
        usd_defi_ratio_range=filters["usd_defi_ratio_range"],
        btc_defi_ratio_range=filters["btc_defi_ratio_range"],
        protocols=filters["protocols"],
        protocol_require_all=True,  # AND logic: must use ALL selected protocols
        tokens=filters["tokens"],
        token_require_all=True,  # AND logic: must hold ALL selected tokens
        vaults=filters["vaults"],  # OR logic: any of selected vaults
        vault_share_range=filters["vault_share_range"]
    )

    # Update filter count in sidebar header (using placeholder from render_sidebar)
    filter_placeholder = filters.get("_filter_count_placeholder")
    if filter_placeholder:
        with filter_placeholder.container():
            st.markdown("### 📊 필터 결과")
            st.metric("필터된 지갑", f"{len(df_filtered):,} / {len(df):,}")
            st.divider()

    # Get data directory for bias analysis
    data_dir = find_data_directory()

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 개요",
        "🎯 편향 분석",
        "🔗 교차 분석",
        "📋 테이블",
        "📚 데이터 출처"
    ])

    with tab1:
        render_overview_tab(df, df_filtered, eth_price=ETH_PRICE_USD)

    with tab2:
        render_bias_analysis_tab(df_filtered, data_dir)

    with tab3:
        render_intersection_tab(df)

    with tab4:
        render_table_tab(df_filtered)

    with tab5:
        render_data_source_tab(df)


if __name__ == "__main__":
    main()
