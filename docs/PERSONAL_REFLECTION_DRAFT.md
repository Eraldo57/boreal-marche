# Personal reflection — draft for owner verification

Eraldo Ibrushi · Vanier capstone · 5 November 2026

This draft uses issues raised in the project discussion. It is not a verified account of every action I personally performed. I must review, correct and approve it before submitting it as my reflection. Implementation and document preparation received AI assistance, described in docs/AI_USE.md.

## From predictive scores to useful decisions

The question that most changed my interpretation of the project was why a customer near a 23% purchase probability would receive the same coupon as someone near 90%. A probability of buying is not a probability of being persuaded by a coupon. The existing utility function uses cost and margin assumptions, but it does not estimate the purchases that would happen without an offer. I therefore cannot defend its utility as measured incremental profit.

Making the formula visible helped me distinguish the theoretical 20% break-even from a threshold selected on observed June outcomes. The updated model selects 20.5%; that number is an empirical optimum under assumptions, not a universal business rule. A proper next step would compare coupon and control outcomes with real costs, margins, redemption and a budget constraint.

I also questioned the basic recency baseline. A stronger model should demonstrate what it adds over a simple, understandable rule. The current model improves average precision, but the equal-count recall comparison remains close. That taught me to separate a ranking advantage from the apparent simplicity of a yes/no flag, and to report both rather than presenting one favourable metric.

Feature questions exposed another practical lesson. A zero standard deviation may mean too little history, rather than perfectly regular behaviour. “Single item” is defined by invoice lines, not necessarily a quantity of one. Category entropy depends on weak description-token categories. Being able to explain these definitions matters more than reciting the feature names.

<!-- PAGEBREAK -->

# Personal reflection — evidence and responsibility

Draft for owner review · page 2

## Explaining results without overstating them

For basket recommendations, I needed a concrete example: the white holder is A and the red holder is B. Confidence describes how frequently B occurred among baskets containing A; lift compares that with B’s overall frequency. The half-basket test reports whether any of five suggestions recovers a hidden item. It does not say that every fifth suggestion is correct or that sales increased by the same percentage.

The project also showed why labels and denominators matter. A product-description prefix can be part of the recorded name even when it resembles a quantity. Rule-firing coverage and catalogue coverage answer different questions. FP-Growth may be the exported algorithm while Apriori is faster in the measured run. I should explain the observed evidence instead of repeating general claims about algorithms.

French assistant questions made the distinction between routing and answer quality visible. Selecting the intended tool is useful, but an answer can still be wrong or incomplete. Citations and fluent French are not enough. The saved correctness assessment used one AI rater and development questions. Independent bilingual evaluation would give a more credible measure of readiness.

## What I would carry forward

The completion work reduced the serving contract from 25 to 12 inputs after a development-period audit, while preserving the original model and its reports. The new September result reuses an inspected test period, so I should acknowledge that limit. I would freeze the method and evaluate a genuinely later period before making stronger generalisation claims.

I would also separate implementation from operational proof. A local test suite and Docker stack are valuable, but they do not establish cloud execution, a public deployment, authenticated access or functioning privacy-rights procedures. Those need their own evidence. My remaining responsibilities include reviewing the reflection and AI attribution, supplying authorised deployment access, recording the required demonstration and being able to explain and debug the submitted code.

AI assistance accelerated inspection, implementation and preparation. My responsibility is to verify the work, understand its assumptions and defend only what the evidence supports. Before submission, I must replace any statement that does not accurately describe my own contribution and experience.
