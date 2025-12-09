-- Lido stETH (0.05 <= balance < 1000, random 1000)
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
FROM holder_totals ORDER BY RAND() LIMIT 1000
