# LeRobot v3 Config Benchmark

- config: `config/data/lerobot_v3_octo25.yaml`
- output_format: `stage1`
- num_sources: `24`
- total_episodes: `639138`
- train_episodes: `639090`
- val_episodes: `48`
- train_samples: `24048868`
- val_samples: `13527`

## Best by batch size

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 8 | 623.9 | 0.2051 | 49.1775 | 48.00 |
| 256 | 8 | 1289.7 | 0.1985 | 54.2066 | 96.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 4 | 156.6 | 0.8172 | 49.0166 | 48.00 | 2615.5 |
| 128 | 8 | 623.9 | 0.2051 | 49.1775 | 48.00 | 3081.9 |
| 256 | 4 | 174.2 | 1.4696 | 53.8508 | 96.00 | 3622.2 |
| 256 | 8 | 1289.7 | 0.1985 | 54.2066 | 96.00 | 4082.7 |
