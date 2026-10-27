# Source and checksum verification

The original pinned digest failed against the official UCI workbook. Ingestion was
changed to reject mismatches **before** reading/cleaning data, and this failure was
observed in `logs/ingest_local.log`. No processed artifact was produced by that attempt.

Official catalogue: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Official archive URL: https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
The archive request timed out. The official direct workbook succeeded:
https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx

Executed download:
```sh
curl -fL --connect-timeout 20 --max-time 180 --retry 2 'https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx' -o data/raw/online_retail_II.xlsx
```

Measured SHA-256, now pinned in configuration:
`bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`.
This is a trust-on-first-verified-official-download pin, not a UCI-signed digest.
Subsequent downloads must match it. Workbook sheets/schema/counts are checked during
the actual ingestion; the resulting manifest contains the raw and processed hashes.

Original prices remain **GBP**. The separately authored fictional grocery catalogue
uses CAD and is not a product mapping for the UCI giftware transactions.
