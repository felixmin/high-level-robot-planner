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
| 64 | 16 | 248.1 | 0.2580 | 51.5568 | 24.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 115.4 | 0.5546 | 46.0263 | 24.00 | 3093.6 |
| 64 | 8 | 214.0 | 0.2991 | 47.1162 | 24.00 | 3584.9 |
| 64 | 12 | 211.1 | 0.3032 | 51.4466 | 24.00 | 4089.0 |
| 64 | 16 | 248.1 | 0.2580 | 51.5568 | 24.00 | 4533.2 |
