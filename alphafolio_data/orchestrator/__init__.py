"""Airflow-style pipeline orchestrator hosted in alphafolio_data.

DAG executes data backfill + grade generation + backtest end-to-end.
Persists state in Postgres so re-runs resume from the first failed task.

- storage.py: DDL/CRUD for orch_runs / orch_tasks / orch_logs
- tasks.py:   Task implementations + DAG topology
- runner.py:  Async DAG executor with resume + cancel
"""
