-- High Growth ETH (0.05 <= balance < 1000, ALL - no random sampling)
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
FROM holder_totals
