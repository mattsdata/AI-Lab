# ADR-0001: Two-layer architecture (packages and frameworks)

- **Status:** Accepted
- **Date:** 2026-11-03
- **Author:** Mathew Varghese

## Context

Applied AI work repeats the same technical tasks across industries: detecting objects,
retrieving documents, forecasting series. Each domain also has its own vocabulary and
rules, such as portfolios in finance or stations in climate science. If domain logic
leaks into shared code, reuse stops working and every new domain means rewriting shared parts.

## Decision

1. **Packages** implement one technical capability each and contain no domain terms.
2. **Frameworks** combine packages for one domain and own its vocabulary and rules.
3. **Dependencies point one way only:** frameworks may import packages, never the reverse.
   CI enforces this with an import-linter contract.
4. Cross-cutting concerns (provenance, uncertainty, privacy, configuration) live in
   the shared core, so every framework applies them the same way.

## Consequences

**Positive**
- A new domain is a new framework, not a fork of existing code.
- Packages can be tested and versioned independently.
- Users install only what they need, for example `ai-lab[finance]`.

**Trade-offs**
- Some upfront design is needed before the first framework is written.
- Shared interfaces must be stable, so changes to packages need care.

## Revisit when

A package grows a second, unrelated responsibility. It should then be split.
