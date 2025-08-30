# Concern Audit Report

- Root: `C:/Users/Kaito_Tanaka/Desktop/private_file/kancolle_data/kancolle_program`
- Generated: 2025-08-30 23:23:43

## Summary
- Total LOC: **1454**
- Concerns: **7** (including `unclassified`)
- Entropy H: **2.50**  |  CC p95: **11.0**  |  Fan-out avg: **7.3**  |  IO avg: **0.08**
- SKDI: **0.46** → Gray zone (0.40–0.60)

## Per Concern
| Concern | Files | LOC | CC p95 | Fan-out avg | IO avg |
|---|---:|---:|---:|---:|---:|
| `unclassified` | 6 | 334 | 6.6 | 4.3 | 0.04 |
| `config` | 1 | 273 | 21.7 | 8.0 | 0.02 |
| `runner` | 3 | 254 | 10.6 | 9.0 | 0.13 |
| `decision` | 1 | 208 | 8.2 | 8.0 | 0.03 |
| `ui` | 1 | 160 | 4.7 | 9.0 | 0.03 |
| `region` | 1 | 113 | 10.9 | 10.0 | 0.17 |
| `assets` | 1 | 112 | 7.0 | 14.0 | 0.30 |

## Notes
- CC/IO/Fan-out は簡易推定です。厳密な解析には radon / import-linter 等の導入を検討してください。
- ファイル先頭に `# concern: <name>` と書くと手動分類できます。