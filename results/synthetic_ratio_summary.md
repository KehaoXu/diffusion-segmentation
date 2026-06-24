# Synthetic Ratio Summary

Aggregated from existing `outputs/**/eval_metrics.json` files. No new training runs are included.

| Synthetic ratio | Dice mean +/- std | Delta Dice | IoU | Precision | Recall | Best Dice |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.5701 +/- 0.0264 | +0.0000 | 0.4486 | 0.6373 | 0.6021 | 0.5932 |
| 0.1 | 0.5801 +/- 0.0146 | +0.0100 | 0.4561 | 0.6236 | 0.6277 | 0.6017 |
| 0.2 | 0.5751 +/- 0.0077 | +0.0050 | 0.4533 | 0.6358 | 0.6127 | 0.5880 |
| 0.3 | 0.5866 +/- 0.0056 | +0.0164 | 0.4637 | 0.6615 | 0.6111 | 0.5918 |
| 0.4 | 0.5819 +/- 0.0083 | +0.0117 | 0.4598 | 0.6550 | 0.6059 | 0.5909 |
| 0.5 | 0.5818 +/- 0.0062 | +0.0116 | 0.4599 | 0.6464 | 0.6077 | 0.5913 |
| 0.6 | 0.5835 +/- 0.0129 | +0.0134 | 0.4611 | 0.6644 | 0.6034 | 0.6085 |
| 0.7 | 0.5841 +/- 0.0087 | +0.0140 | 0.4623 | 0.6646 | 0.6054 | 0.5950 |
| 0.8 | 0.5866 +/- 0.0075 | +0.0165 | 0.4652 | 0.6740 | 0.6015 | 0.6030 |
| 0.9 | 0.5891 +/- 0.0060 | +0.0189 | 0.4663 | 0.6586 | 0.6129 | 0.5981 |

Takeaway: synthetic augmentation gives modest average gains over the real-only baseline in these runs. The result is best presented as an applied ML experiment on reproducible evaluation and ratio tuning, not as a new generative modeling method.
