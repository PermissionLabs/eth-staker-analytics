"""
Asset group configuration for ETH Stakers Portfolio Dashboard.
Users can edit these groups via the UI.
"""

# Default asset groups with tokens and display colors
DEFAULT_ASSET_GROUPS = {
    "ETH": {
        "tokens": [
            "ETH", "WETH", "stETH", "wstETH", "rETH",
            "cbETH", "mETH", "osETH", "eETH", "sfrxETH",
            "swETH", "ankrETH", "frxETH", "ETHx"
        ],
        "color": "#627EEA",
        "description": "Ethereum and liquid staking derivatives"
    },
    "BTC": {
        "tokens": [
            "BTC", "WBTC", "tBTC", "cbBTC", "sBTC",
            "renBTC", "HBTC", "imBTC", "pBTC"
        ],
        "color": "#F7931A",
        "description": "Bitcoin and wrapped variants"
    },
    "USD": {
        "tokens": [
            "USDT", "USDC", "DAI", "FRAX", "TUSD",
            "BUSD", "LUSD", "USDD", "GUSD", "USDP",
            "USD₮0", "sUSD", "MIM", "DOLA", "crvUSD"
        ],
        "color": "#26A17B",
        "description": "USD-pegged stablecoins"
    },
    "EUR": {
        "tokens": [
            "EURS", "EURT", "agEUR", "EURe", "EURC",
            "sEUR", "cEUR"
        ],
        "color": "#003399",
        "description": "EUR-pegged stablecoins"
    }
}

# Known DeFi protocols for filtering
KNOWN_PROTOCOLS = [
    "LIDO",
    "Rocket Pool",
    "Aave",
    "Compound",
    "Uniswap",
    "Curve",
    "Convex",
    "Balancer",
    "MakerDAO",
    "Eigenlayer",
    "Pendle",
    "Morpho",
    "Spark",
    "Kiln",
    "Frax",
    "Yearn",
    "1inch"
]

# ETH-related protocols for DeFi ratio calculation
ETH_DEFI_PROTOCOLS = [
    "LIDO",
    "Rocket Pool",
    "Kiln",
    "Eigenlayer",
    "Frax ETH",
    "Swell",
    "Stader",
    "Mantle Staked ETH"
]

# Chart color palette
CHART_COLORS = [
    "#627EEA",  # ETH blue
    "#F7931A",  # BTC orange
    "#26A17B",  # Tether green
    "#003399",  # EUR blue
    "#8B5CF6",  # Purple
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#F59E0B",  # Amber
]

# UI Configuration
UI_CONFIG = {
    "page_title": "ETH Stakers Portfolio Dashboard",
    "page_icon": "📊",
    "layout": "wide",
    "sidebar_width": 300,
    "max_venn_filters": 3
}
