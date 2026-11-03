# ADR-0007: Heritage tools propose candidates, demand evidence and protect locations

- **Status:** Accepted
- **Date:** 2026-11-03
- **Author:** Mathew Varghese

## Context

Elevation models, especially LiDAR, reveal low earthworks that may be the remains of walls,
enclosures, forts or settlements. An algorithm that finds regular shapes is useful for
prospection, but it cannot tell a medieval bank from a modern field boundary, and publishing
the exact location of an unexcavated site invites looting and illegal metal detecting.

## Decision

1. **Candidates, not discoveries.** Detection output is always an *unverified candidate* with
   a geometric regularity score and a plain interpretation. The score measures shape only.
2. **Status follows evidence.** A record becomes *supported* only with cited documentary
   evidence (texts, historical maps, place names) and *confirmed* only with field survey or
   excavation. Software alone can never confirm a site. Every evidence item needs a source.
3. **Locations are protected by default.** Public outputs snap coordinates to the centre of a
   grid square (about 5 km). Exact coordinates require a confirmed status and an explicit
   opt-in. Snapping (not random noise) prevents recovery by averaging repeated outputs.
4. **Report, do not dig.** Outputs name the responsible authorities (ASI / state archaeology
   in India; HER / Historic England and national bodies in the UK) and the protecting law.
5. **Change over time is a range.** Degradation over 10, 500 and 1000 years is modelled with
   hillslope diffusion across a low-to-high diffusivity range, and stated as a natural-process
   estimate that ignores ploughing, building and stone robbing.

## Consequences

- Results are less headline-friendly ("candidate", "supported") but defensible and safe.
- Specialists and authorities stay in the loop; the tool speeds up their work instead of
  replacing it.
- Thresholds and the protection precision are parameters, so projects can follow the
  guidance of their own heritage authority.
