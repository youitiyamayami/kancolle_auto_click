# Concern Audit Report

- Root: `C:/Users/Kaito_Tanaka/Desktop/private_file/kancolle_data/kancolle_program`
- Generated: 2025-08-30 20:55:42

## Summary
- Total LOC: **1022**
- Concerns: **5** (including `unclassified`)
- Entropy H: **1.68**  |  CC p95: **10.0**  |  Fan-out avg: **8.7**  |  IO avg: **0.10**
- SKDI: **0.40** → Keep (≤0.40)

## Per Concern
| Concern | Files | LOC | CC p95 | Fan-out avg | IO avg |
|---|---:|---:|---:|---:|---:|
| `runner` | 3 | 386 | 9.9 | 10.3 | 0.10 |
| `unclassified` | 4 | 334 | 7.0 | 6.5 | 0.06 |
| `region` | 1 | 113 | 10.9 | 10.0 | 0.17 |
| `assets` | 1 | 112 | 7.0 | 14.0 | 0.30 |
| `config` | 1 | 77 | 6.8 | 6.0 | 0.01 |

## Notes
- CC/IO/Fan-out は簡易推定です。厳密な解析には radon / import-linter 等の導入を検討してください。
- ファイル先頭に `# concern: <name>` と書くと手動分類できます。