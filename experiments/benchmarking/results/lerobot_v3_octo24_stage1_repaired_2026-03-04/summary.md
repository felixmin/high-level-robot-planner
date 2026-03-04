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
| 32 | 8 | 287.4 | 0.1113 | 45.8177 | 12.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 4 | 92.0 | 0.3476 | 46.3562 | 12.00 | 2774.0 |
| 32 | 8 | 287.4 | 0.1113 | 45.8177 | 12.00 | 3303.6 |
| 32 | 12 | 96.6 | 0.3312 | 46.1668 | 12.00 | 3820.4 |
| 32 | 16 | 107.1 | 0.2987 | 47.3764 | 12.00 | 4342.5 |
| 32 | 20 | 135.4 | 0.2364 | 47.7137 | 12.00 | 4881.6 |
| 32 | 24 | 153.0 | 0.2091 | 46.4435 | 12.00 | 5396.5 |
