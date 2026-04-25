# Trading System Implementation Plan

## Goal

Move the project from a research and dry-run execution baseline to a production-oriented trading system with:

- multi-symbol live portfolio execution
- event-driven market data and order lifecycle processing
- asynchronous fill/account reconciliation
- monitoring, alerts, audit trail, and operator reporting
- systematic walk-forward and multi-symbol stability validation

## Principles

- Keep signal generation deterministic by default; LLM is used for feature extraction, review, and operator assistance.
- Build production capabilities as reusable infrastructure, not one-off scripts.
- Prefer narrow, testable milestones that can run locally end to end.

## Phase 1: Operational Foundations

Status: completed

Deliverables:

- event model and in-process event bus
- JSONL audit journal for live execution events
- rule-based alert sink for rejected orders, risk halts, service failures, and drawdown alerts
- CLI/Web output consistency for research and paper pipelines
- walk-forward validation entrypoint

Exit criteria:

- live service emits machine-readable event and alert logs
- CLI and Web produce aligned result artifacts
- walk-forward can be run from a script and produces summary files

## Phase 2: Multi-Symbol Validation

Status: in progress

Deliverables:

- multi-symbol walk-forward runner
- per-symbol and aggregate fold summaries
- validation report with positive fold rate, drawdown distribution, turnover, and fee drag

Exit criteria:

- a stock pool can be validated in one command
- outputs support filtering weak symbols before live deployment

## Phase 3: Portfolio Live Execution

Status: in progress

Deliverables:

- portfolio live engine for multiple symbols
- portfolio-level position sizing and exposure caps
- portfolio-level live risk manager
- per-symbol and aggregate execution state snapshots

Exit criteria:

- live dry-run can manage more than one symbol in a single cycle
- portfolio constraints are enforced before order submission

## Phase 4: Event-Driven Data and Broker Integration

Status: in progress

Deliverables:

- market data feed abstraction for polling and streaming
- event-driven service loop on top of the event bus
- broker callback ingestion path for async order/fill/account updates

Exit criteria:

- live flow no longer depends only on periodic full snapshot polling
- order/fill/account updates can be processed asynchronously

## Phase 5: Reconciliation and Operator Controls

Status: pending

Deliverables:

- broker/account reconciliation module
- mismatch detection and alerting
- operator-visible health, lag, and execution status summary

Exit criteria:

- local state and broker state can be compared deterministically
- mismatches are journaled and surfaced as alerts

## Phase 6: Monitoring and Reporting

Status: pending

Deliverables:

- execution and risk summary report artifacts
- alert severity classification and compact operator report
- result retention and run catalog improvements

Exit criteria:

- each live cycle produces a compact operational summary
- operator can inspect recent alerts, fills, exposure, and drawdown without reading raw CSVs
