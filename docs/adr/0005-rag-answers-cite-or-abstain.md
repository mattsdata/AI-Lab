# ADR-0005: RAG answers must cite verifiable sources or abstain

- **Status:** Accepted
- **Date:** 2026-11-03
- **Author:** Mathew Varghese

## Context

Language models produce fluent answers whether or not the evidence supports them. In credit,
compliance or clinical work, an uncited answer cannot be checked, and a confident wrong answer
(an invented debt figure, a missed lawsuit) is worse than no answer.

## Decision

Every retrieval-augmented answer in AI Lab follows four rules:

1. **Answer only from the numbered sources shown.** The system prompt forbids outside knowledge.
2. **Cite every claim as [n], and verify the citations.** Numbers that do not match a source
   shown are recorded as invalid; such an answer is flagged *ungrounded* in reports.
3. **Abstain explicitly.** The model must reply `NOT FOUND` when the sources do not contain the
   answer, and the pipeline abstains without calling the model when retrieval finds nothing.
4. **Measure it.** Each use case has an evaluation set that includes *unanswerable* questions,
   scored for retrieval hit rate, MRR, citation precision, abstention accuracy and expected terms.

Quotes shown to users are cut from the source by character offsets, never paraphrased.

## Consequences

**Positive**
- Every statement in a generated memo can be traced to the filing in one click.
- Hallucination is measured, not assumed away: a system that never abstains fails the set.
- Models, prompts, chunk sizes and retrievers can be swapped and compared on the same tests.

**Trade-offs**
- Conservative systems sometimes abstain when a looser one would have answered correctly
  (the extractive baseline does this for broad questions); that is the intended direction of error.
- Evaluation sets must be written and maintained by people who know the documents.
