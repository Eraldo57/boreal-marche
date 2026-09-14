# Data Card — Boréal Marché

---

## ⚠️ Provenance Disclosure

**This application uses the UCI Online Retail II dataset, a publicly available dataset
re-skinned under the name "Boréal Marché" for a pedagogical capstone project at Vanier
College (course 420-977-VA). No real Boréal Marché company or customer data exists.
Presenting this public data as proprietary client data would be an academic integrity
violation.**

---

## Source

| Field | Value |
|---|---|
| **Dataset name** | Online Retail II |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/502/online+retail+ii |
| **Original authors** | Daqing Chen, School of Engineering, London South Bank University |
| **Year** | 2019 (deposited); data spans December 2009 – December 2011 |
| **Licence** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Citation** | Chen, D. (2019). Online Retail II. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D |

---

## Description

The dataset contains all transactions occurring between 01/12/2009 and 09/12/2011 for a
UK-based and registered non-store online retail company. The company mainly sells unique
all-occasion gifts. Many customers of the company are wholesalers.

---

## Schema

| Column | Type | Description | Example |
|---|---|---|---|
| `Invoice` | string | 6-digit invoice number. Prefix "C" indicates a cancellation. | `489434`, `C489449` |
| `StockCode` | string | 5-digit product code. Some codes are service charges (POST, DOT, M, etc.). | `85048`, `POST` |
| `Description` | string | Product name (free text, English). | `"15CM CHRISTMAS GLASS BALL 20 LIGHTS"` |
| `Quantity` | integer | Quantity purchased per transaction line. Negative for cancellations. | `12`, `-2` |
| `InvoiceDate` | datetime | Date and time of the invoice. | `2009-12-01 07:45:00` |
| `Price` | float | Unit price in GBP (£). | `6.95` |
| `Customer ID` | float | 5-digit customer identifier. May be null for unregistered purchases. | `13085.0`, `NaN` |
| `Country` | string | Country of the customer. | `"United Kingdom"`, `"France"` |

---

## Size

| Metric | Value |
|---|---|
| **Raw rows** | 1,067,371 |
| **Raw columns** | 8 |
| **File size (Excel)** | ~45 MB |
| **Unique invoices** | ~28,000 |
| **Unique customers** | ~5,942 (with Customer ID) |
| **Unique stock codes** | ~4,631 |
| **Countries** | 43 |
| **Date range** | 2009-12-01 to 2011-12-09 |

---

## Known Quality Issues

| Issue | Rows affected | Treatment |
|---|---|---|
| **Null Customer ID** | ~243,000 (22.8%) | Dropped — cannot build customer-level features without an identifier |
| **Cancellations** | ~16,000 invoices with "C" prefix | Dropped from positive transactions; used to compute `cancellation_rate` and `has_ever_cancelled` features |
| **Non-positive quantities** | ~11,000 rows | Dropped — represent returns or data errors |
| **Non-positive prices** | ~2,500 rows | Dropped — free samples or errors |
| **Service SKUs** | ~1,500 rows (POST, DOT, BANK CHARGES, M, etc.) | Dropped — not real products |
| **Description inconsistencies** | Widespread | Canonicalised: lowercased, whitespace stripped and collapsed |
| **Duplicate descriptions for different SKUs** | Minor | Accepted — description is used for display, SKU for identity |
| **Extreme outliers** | ~200 rows with Quantity > 10,000 or Price > £1,000 | Investigated individually; genuine wholesale orders are retained, obvious errors are dropped |

---

## Re-skinning as Boréal Marché

The dataset is presented in this project under the fictional name "Boréal Marché" to
create a realistic capstone scenario. The re-skinning involves:

1. **Company name:** "UK-based online retailer" → "Boréal Marché" (fictional Québec
   online grocery and specialty-goods retailer)
2. **Currency:** GBP → CAD (for presentation only; no exchange-rate conversion applied
   to the underlying values)
3. **Context:** The business problems (retention, recommendation, customer support) are
   adapted to a Québec market context, including bilingual (French/English) requirements

**No values in the dataset are altered.** The re-skinning is a narrative wrapper only.

---

## Bias Considerations

1. **Geographic bias:** ~90% of transactions are from the United Kingdom. The model's
   understanding of "domestic" vs. "international" customer behaviour is dominated by
   UK patterns, which may not generalise to a Québec context.

2. **Temporal bias:** The data spans 2009–2011. Consumer purchasing patterns, especially
   in e-commerce, have evolved substantially since then.

3. **Wholesale bias:** Many customers are wholesalers who buy in bulk. Their purchasing
   patterns (high frequency, large quantities, narrow product range) differ from
   individual consumers. The model may over-weight wholesale behaviour.

4. **Missing-customer bias:** 22.8% of transactions lack a Customer ID. These
   unregistered purchases are dropped entirely, which may systematically exclude
   certain customer types (e.g., one-time buyers, privacy-conscious customers).

5. **Product-category proxy:** The dataset is dominated by gift items. A real grocery
   retailer would have very different basket dynamics (perishables, staples, seasonal
   produce).

6. **No demographic data:** The dataset contains no age, gender, ethnicity, or income
   data. While this limits the fairness audit, it also means the model cannot directly
   discriminate on protected characteristics. Indirect discrimination through geographic
   features remains possible.

---

## Ethical Use

This dataset should be used only for:
- Educational purposes (coursework, capstone projects)
- Research into customer analytics methodologies
- Demonstrations of ML engineering practices

It should NOT be used to:
- Make decisions about real individuals
- Train models deployed on real customer data without fresh validation
- Present as proprietary data in a commercial setting
