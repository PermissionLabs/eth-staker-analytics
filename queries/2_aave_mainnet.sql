-- Aave Mainnet ETH (0.05 <= balance < 1000, random 1000)
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
FROM holder_totals ORDER BY RAND() LIMIT 1000
