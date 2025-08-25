# Concern Audit Report

- Root: `C:/Users/Kaito_Tanaka/Desktop/private_file/kancolle_data/kancolle_program`
- Generated: 2025-08-26 00:46:13

## Summary
- Total LOC: **524**
- Concerns: **4** (including `unclassified`)
- Entropy H: **1.21**  |  CC p95: **10.6**  |  Fan-out avg: **8.7**  |  IO avg: **0.13**
- SKDI: **0.23** → Keep (≤0.40)

## Per Concern
| Concern | Files | LOC | CC p95 | Fan-out avg | IO avg |
|---|---:|---:|---:|---:|---:|
| `runner` | 3 | 343 | 10.6 | 11.7 | 0.17 |
| `region` | 1 | 113 | 10.9 | 10.0 | 0.17 |
| `assets` | 1 | 53 | 4.0 | 3.0 | 0.04 |
| `unclassified` | 1 | 15 | 2.0 | 4.0 | 0.07 |

## Notes
- CC/IO/Fan-out は簡易推定です。厳密な解析には radon / import-linter 等の導入を検討してください。
- ファイル先頭に `# concern: <name>` と書くと手動分類できます。