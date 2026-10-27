# Data card — measured local dataset

Updated 2026-09-30. Public UCI data; fictional Boréal Marché narrative, not client data.

## Source and provenance

[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii), Daqing Chen (2012), DOI 10.24432/C5CG6D, CC BY 4.0. The original retailer is UK-based and sells giftware, including wholesale purchases. The workbook spans December 2009–December 2011. Prices remain GBP.

The exact workbook URL and SHA-256 are pinned in configs/pipeline.yaml. Ingestion fails closed on a mismatch. The pin was measured from the official download, not supplied as a publisher-signed checksum; see reports/current_run/SOURCE_VERIFICATION.md.

## Measured cleaning

Source: data/processed/manifest.json and cleaning_report.json. Counts are sequential rule effects, not overlapping raw-condition counts.

| Stage | Removed | Modified without removal |
|---|---:|---:|
| Original workbook, both sheets | — | 1,067,371 input rows |
| Invalid schema / zero quantity | 0 | 0 |
| Non-positive price | 6,207 | 0 |
| Missing customer ID | 236,871 | 0 |
| Service SKUs | 3,648 | 0 |
| Description canonicalization | 0 | 820,645 |
| Fixed extreme-value caps | 0 | 9 |
| Purchase/cancellation/return flags | 0 | 820,645 |
| Final event ledger | — | 820,645 output rows |

The output includes 802,679 purchase rows and 17,966 cancellation/return rows. Cancellation and return flags overlap; do not add them as disjoint counts. Percentages and per-rule denominators are saved in the cleaning report.

## Schema and treatment

Original columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country. Added: WasCapped, IsCancellation, IsReturn, IsPurchase, LineTotal. Negative quantities and C invoices are preserved in the event ledger but excluded from purchase labels, revenue features and baskets.

Fixed caps are abs(Quantity) ≤ 10,000 and Price ≤ GBP 1,000. They preserve sign and invoice presence, flag affected rows, and do not learn thresholds from future data. This is a transparent engineering sensitivity treatment, **not** proof that capped wholesale orders are errors. The raw workbook remains intact; uncapped sensitivity analysis is future work. Exact duplicate rows are retained because the source supplies no stable line identifier proving accidental duplication.

Head A counts distinct StockCodes for basket variety; Head B intentionally identifies items by canonical lowercase description. Different SKUs can merge or description variants can split products. The assistant catalogue (`src/catalogue.py`) keys products by StockCode, names each by its most frequent description and keeps the other variants for search.

## Limits

Customer identifiers are pseudonymous source identifiers, not proof of anonymization. Rows without usable IDs are excluded, causing selection bias. Geographic/temporal/wholesale patterns may not transfer to Québec grocery shoppers. The first description token is only a category proxy. No demographic fairness inference is supported.

The authored policy documents describe a fictional retailer's rules. Product facts come only from the catalogue export of UCI purchases: 4,624 StockCodes with median observed unit price in GBP. It has no categories, allergens or dietary tags because the source has none.
