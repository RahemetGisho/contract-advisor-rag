"""Per-query token and cost tracking.

Pricing is approximate and provider/model-specific -- it is not
pulled live from any pricing API (no such stable API exists across
providers), so it is a config-driven estimate, not a billing-accurate
figure. The point is to make cost a visible, tracked number on every
request, not to replace your provider's actual invoice.
"""

from dataclasses import dataclass

# USD per 1,000 tokens. Update these if you switch providers/models --
# intentionally kept as a plain, editable table rather than hidden
# inside a function.
PRICING_PER_1K_TOKENS = {
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "mock": {"input": 0.0, "output": 0.0},
}


@dataclass
class CostEstimate:
    model_name: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


def estimate_cost(
    model_name: str, input_tokens: int, output_tokens: int
) -> CostEstimate:
    rates = PRICING_PER_1K_TOKENS.get(model_name)
    if rates is None:
        # Unknown model: don't silently report $0 -- that would hide
        # cost rather than surface it. Fall back to a conservative
        # placeholder rate and make the uncertainty visible in logs,
        # rather than crashing a request over a pricing-table gap.
        rates = {"input": 0.001, "output": 0.001}

    cost = (input_tokens / 1000) * rates["input"] + (output_tokens / 1000) * rates[
        "output"
    ]
    return CostEstimate(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=round(cost, 6),
    )
