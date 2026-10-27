# Canonical feature inventory

Contract 1.0; source of truth: src/contracts.py, FEATURE_NAMES. All features use events strictly before the cutoff. Order below is serialization/API order.

| # | Name | Definition |
|---|---|---|
| 1 | recency_days | Whole days since latest positive invoice |
| 2 | frequency | Distinct positive invoices |
| 3 | monetary_total | Sum of positive line totals, GBP |
| 4 | monetary_mean | Mean positive invoice value, GBP |
| 5 | monetary_std | Sample standard deviation of invoice values; one invoice → 0 |
| 6 | days_since_first_purchase | Whole days since first positive invoice |
| 7 | avg_days_between_orders | Mean whole-day invoice gap; one invoice → tenure |
| 8 | std_days_between_orders | Sample gap deviation; fewer than two gaps → 0 |
| 9 | is_one_time_buyer | Exactly one positive invoice → 1 |
| 10 | avg_basket_size | Mean number of transaction lines per positive invoice, not units |
| 11 | avg_basket_value | Mean invoice value; currently redundant with monetary_mean |
| 12 | max_basket_value | Maximum invoice value, GBP |
| 13 | avg_unique_items_per_basket | Mean distinct StockCodes per invoice |
| 14 | single_item_basket_ratio | Fraction of invoices with exactly one line |
| 15 | top_category_share | Largest spend share by first description token |
| 16 | n_unique_categories | Distinct first-token proxy categories |
| 17 | category_entropy | Base-2 Shannon entropy of category spend shares |
| 18 | has_ever_cancelled | Any C-prefixed historical invoice → 1 |
| 19 | cancellation_rate | Cancelled invoices / all distinct historical event invoices |
| 20 | return_rate | Absolute negative units / all absolute event units |
| 21 | weekend_purchase_ratio | Positive invoices on Saturday/Sunday / positive invoices |
| 22 | morning_purchase_ratio | Positive invoices before noon / positive invoices |
| 23 | country_code | Majority country; map fitted at earliest training cutoff; unseen → −1 |
| 24 | is_domestic | Majority country is United Kingdom → 1, not Québec |
| 25 | n_countries | Distinct countries in positive purchase history |

Fixed outlier caps precede these calculations. Ratios are bounded 0–1, binary flags are 0/1, unknown country alone may be −1. Structural defaults above are explicit. Remaining missing numeric values are left for a median imputer fitted only on the estimator's training rows.

The API requires all 25 finite numbers and rejects missing/extra keys, strings, booleans, infinities, NaNs and invalid ranges. The saved historical feature store supports the demo; it is not a live feature-computation service. Feature necessity, category taxonomy, ordinal-country encoding and redundant monetary features require further study.

