# ==================================================================================================
# 📜 ailab-finance :: examples :: annual_report
# --------------------------------------------------------------------------------------------------
#  File         : frameworks/finance/examples/annual_report.py
#  Author       : Mathew Varghese
#  Created      : 2026-11-03
#  Version      : V1.0
#  License      : MIT (see LICENSE)
# --------------------------------------------------------------------------------------------------
#  🎯 PURPOSE
#    Download a company's latest 10-K from SEC EDGAR and write a review memo answering the
#    standard analyst questions, each answer citing the report's own words.
#
#  🔧 WHAT IT DOES
#    EDGAR filing list -> primary document -> text -> chunks -> retrieval -> answers with
#    verified citations -> Markdown memo (written atomically) with a list of uncovered topics
#
#  🧠 DESIGN NOTES
#    - --llm extractive (default) runs with no model at all and never paraphrases; --llm ollama
#      uses a free local model; --llm openai-compatible works with hosted or self-hosted APIs
#      (key from the LLM_API_KEY environment variable).
#    - Requires a SEC User-Agent ("Name email@domain") via --sec-agent or SEC_USER_AGENT.
#
#  USAGE
#    export SEC_USER_AGENT="Your Name you@example.com"
#    python frameworks/finance/examples/annual_report.py MSFT
#    python frameworks/finance/examples/annual_report.py MSFT --llm ollama --model llama3.1
#
#  📂 VERSION HISTORY
#
#    V1.0 (2026-11-03) - Mathew Varghese
#      - Initial demo: EDGAR 10-K to cited review memo.
# ==================================================================================================
"""Annual report analyser: latest 10-K to a cited review memo."""

# ========================================
# 📂 Imports
# ========================================
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ailab_nlp import ExtractiveAnswerer, LlmAnswerer, create_llm, html_to_text
from ailab_nlp.rag import Answerer
from mattslib import atomic_write

from ailab_finance import EdgarClient, analyse_report, findings_markdown


# ========================================
# 📂 Pipeline
# ========================================
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--sec-agent", default=os.environ.get("SEC_USER_AGENT"))
    parser.add_argument(
        "--llm", default="extractive", choices=["extractive", "ollama", "openai-compatible"]
    )
    parser.add_argument("--model", default="llama3.1")
    parser.add_argument("--base-url", help="for --llm openai-compatible")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.sec_agent:
        parser.error('set --sec-agent or SEC_USER_AGENT, e.g. "Your Name you@example.com"')

    edgar = EdgarClient(args.sec_agent)
    filing = edgar.filings(args.ticker, "10-K", limit=1)[0]
    print(f"{filing.ticker} 10-K filed {filing.filed}: {filing.url}")
    text = html_to_text(edgar.filing_document(filing))

    answerer: Answerer = ExtractiveAnswerer()
    if args.llm == "ollama":
        answerer = LlmAnswerer(create_llm("ollama", model=args.model))
    elif args.llm == "openai-compatible":
        if not args.base_url:
            parser.error("--base-url is required for --llm openai-compatible")
        answerer = LlmAnswerer(
            create_llm("openai-compatible", model=args.model, base_url=args.base_url)
        )

    source = f"{filing.ticker} 10-K filed {filing.filed}"
    findings = analyse_report(text, source, answerer=answerer)
    memo = findings_markdown(findings, source)
    output = args.output or Path(f"{filing.ticker}-10K-{filing.filed}-review.md")
    atomic_write(output, memo + f"\n\nSource: {filing.url}\n")
    answered = sum(not f.answer.abstained for f in findings)
    print(f"{answered}/{len(findings)} topics answered with citations. Memo: {output}")


if __name__ == "__main__":
    main()
