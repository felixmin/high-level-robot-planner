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
| 32 | 16 | 223.8 | 0.1430 | 51.9950 | 12.00 |
| 64 | 16 | 437.0 | 0.1465 | 52.7753 | 24.00 |
| 128 | 16 | 427.4 | 0.2995 | 54.6427 | 48.00 |
| 256 | 16 | 418.3 | 0.6120 | 59.7940 | 96.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 16 | 223.8 | 0.1430 | 51.9950 | 12.00 | 2625.5 |
| 64 | 16 | 437.0 | 0.1465 | 52.7753 | 24.00 | 3168.3 |
| 128 | 16 | 427.4 | 0.2995 | 54.6427 | 48.00 | 3631.8 |
| 256 | 16 | 418.3 | 0.6120 | 59.7940 | 96.00 | 4166.7 |
