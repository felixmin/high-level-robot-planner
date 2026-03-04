# LeRobot v3 Config Benchmark

- config: `config/data/lerobot_v3_octo24_libero.yaml`
- output_format: `stage1`
- num_sources: `25`
- total_episodes: `640831`
- train_episodes: `640781`
- val_episodes: `50`
- train_samples: `24320393`
- val_samples: `13774`

## Best by batch size

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 67.2 | 0.4759 | 50.2043 | 12.00 |
| 64 | 16 | 3980966.0 | 0.0000 | 54.2454 | 24.00 |
| 128 | 16 | 4681357.6 | 0.0000 | 59.4980 | 48.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 67.2 | 0.4759 | 50.2043 | 12.00 | 2618.7 |
| 64 | 16 | 3980966.0 | 0.0000 | 54.2454 | 24.00 | 3148.4 |
| 128 | 16 | 4681357.6 | 0.0000 | 59.4980 | 48.00 | 3625.2 |
