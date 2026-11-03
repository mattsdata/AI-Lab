# ADR-0003: Extract shared code into a package on its second use

- **Status:** Accepted
- **Date:** 2026-11-03
- **Author:** Mathew Varghese

## Context

ADR-0001 puts reusable, domain-free code in packages. Some finance code, such as forward
returns or rolling statistics, is generic time-series logic that the climate framework will
also need. A `timeseries` package could be created now, but its interface would be guessed
from a single user.

## Decision

Generic code starts inside the first framework that needs it. When a **second** framework
needs the same capability, it is extracted into a package, with an interface shaped by both
real uses. The candidates are marked in module docstrings.

## Consequences

**Positive**
- Package interfaces are based on two real cases, not speculation.
- No empty or premature packages to maintain.

**Trade-offs**
- One extraction refactor per shared capability. Tests in the first framework protect it.
- The dependency rule (packages never import frameworks) still holds throughout, because
  code moves *down* into packages, never up.
