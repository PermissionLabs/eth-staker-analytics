# ETH Staker Analytics Dashboard

이더리움 스테이커들의 포트폴리오 분석 대시보드

## Features

- 지갑 포트폴리오 분석 (ETH/USD/BTC 비율)
- DeFi 포지션 분석
- 프로토콜 사용 현황
- 볼트별 필터링
- 교차 분석 (Venn 다이어그램)

## Data Sources

| Source | Chain | Protocol | Sampling |
|--------|-------|----------|----------|
| lido_steth | Ethereum | Lido stETH | Random 1,000 |
| aave_mainnet | Ethereum | AAVE V3 aWETH | Random 1,000 |
| base_aave | Base | AAVE V3 aWETH | Random 1,000 |
| moonwell | Base | Moonwell Flagship ETH | Random 1,000 |
| high_growth_eth | Ethereum | High Growth ETH Vault | All |

**Filter**: 0.05 ETH ≤ balance < 1000 ETH

## Data Pipeline

```
Dune Analytics (On-chain) → Address List → DeBank API (Portfolio) → Dashboard
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Data**: Pandas
- **Data Sources**: Dune Analytics, DeBank API
