# ML evaluation report

This educational prototype models complete sale invoices using public retail data. The Isolation Forest has 200 trees, seed 42, and contamination 0.02. All preprocessing is fitted only on 16,581 training invoices. Training ends at **2011-10-26 12:28:00** and the 4,144 holdout invoices begin at **2011-10-26 12:31:00**; no timestamp appears in both subsets.

| Holdout comparison | Invoices |
|---|---:|
| Isolation Forest flags | 119 |
| Amount-rule flags | 92 |
| Both | 64 |
| Model only | 55 |
| Rule only | 28 |

The amount rule is `total_gbp > £4,107.42`, with the 99th-percentile threshold learned from training invoices. The model flags 332 training invoices. These counts compare review queues, not predictive accuracy.

Sensitivity using the **same fitted forest** and different thresholds derived only from training scores:

| Assumed contamination | Training flags | Holdout flags |
|---:|---:|---:|
| 0.01 | 166 | 71 |
| 0.02 | 332 | 119 |
| 0.05 | 829 | 193 |

If otherwise valid exact duplicate lines are retained, the fixed trained model flags 451 invoices versus 451 after deduplication. 0 flags appear only with duplicate lines and 0 only after deduplication. In the alternate holdout, the fixed model flags 119 invoices and the same amount rule flags 92. This changes inputs to one fixed model; it is not a retraining experiment.

Top invoices for manual review:
- 573585: £16,874.58, 1114 lines, score 0.090
- 580729: £9,020.83, 721 lines, score 0.088
- 537237: £7,335.45, 597 lines, score 0.087
- 538524: £5,344.98, 490 lines, score 0.082
- 580730: £6,870.20, 662 lines, score 0.082

The listed amounts and line counts are observable characteristics, not confirmed reasons for a flag. Scores indicate statistical rarity, not fraud. There are no verified anomaly labels, so precision, recall, F1 and fraud-detection accuracy cannot be calculated. Returns are audited separately and are outside this sale-invoice model. A chronological holdout measures flag stability, not classification quality.
