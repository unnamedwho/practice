# Data quality report

Source: UCI Online Retail (ID 352), local file `online_retail_mirror.parquet`. Source SHA-256: `0663e4d42cda4401cd76ee44bb278f3b23f02f576399925d4701a6cf07a9591c`.

| Measure | Before | Retained after cleaning |
|---|---:|---:|
| Rows | 541,909 | 536,638 |
| Exact duplicates | 5,268 | 0 |
| Invalid rows excluding duplicates | 3 | 0 |
| Returns/cancellations | — | 10,587 |
| Regular sale rows | — | 526,051 |
| Review flags | — | 5,367 |

Missing values by column before cleaning: `{"InvoiceNo": 0, "StockCode": 0, "Description": 1454, "Quantity": 0, "InvoiceDate": 0, "UnitPrice": 0, "CustomerID": 135080, "Country": 0}`.

Missing values among retained rows: `{"InvoiceNo": 0, "StockCode": 0, "Description": 1454, "Quantity": 0, "InvoiceDate": 0, "UnitPrice": 0, "CustomerID": 135034, "Country": 0}`.

Supplied invalid CustomerID values: 0; missing CustomerID values: 135,080. A missing ID is accepted. A supplied nonintegral, nonfinite, nonpositive, out-of-exact-range, or nonnumeric ID is marked `customer_id_invalid` in the audit, converted to missing for numeric operations, and its invoice is excluded from modeling pending review. No identity is invented. Quantities beyond the exactly representable integer range (2^53) are invalid. Nonfinite quantity rows: 0; nonfinite unit-price rows: 0; line-amount overflows: 0. These cannot enter valid sale or return categories. 0 invoices with nonfinite aggregate values are excluded from ML.

Invoice consistency: 0 invoices with multiple known customer IDs, 0 with multiple countries, 42 with multiple timestamps, 0 spanning more than 24 hours, and 0 excluded from modeling. Different line times within 24 hours are allowed. The invoice date is the latest line timestamp, assuming the input snapshot contains the complete invoice. Conflicting customer or country values are not silently chosen; those invoices require review and are excluded from the model.

Exact duplicate removal is an **assumption**: identical lines may be legitimate repeated items. Keeping the 5,268 otherwise valid duplicate lines changes sale total from £10,631,048.69 to £10,655,622.43, and signed net total from £9,737,068.97 to £9,758,809.95. The ML sensitivity is documented in the ML report. Raw data remains untouched, and all rows remain in the audit Parquet.
