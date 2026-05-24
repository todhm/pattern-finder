-- Postgres initdb: create the alphafolio role + database alongside the
-- existing backtester DB. Runs once on first volume init.
--
-- Backtester (`backtester` user / `backtester` DB) is provisioned by the
-- official Postgres image via POSTGRES_USER/POSTGRES_DB env vars. This
-- script adds the second tenant so the three alphafolio services
-- (data, quant, portfolio) share a Postgres cluster but live in an
-- isolated database — no table-name collisions with backtester.

CREATE USER alphafolio WITH PASSWORD 'alphafolio';
CREATE DATABASE alphafolio OWNER alphafolio;
GRANT ALL PRIVILEGES ON DATABASE alphafolio TO alphafolio;
