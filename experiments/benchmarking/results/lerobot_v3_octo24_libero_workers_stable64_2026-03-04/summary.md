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
| 64 | 16 | 435.1 | 0.1471 | 52.8600 | 24.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 127.0 | 0.5041 | 46.1930 | 24.00 | 2639.7 |
| 64 | 8 | 240.5 | 0.2661 | 47.8356 | 24.00 | 3122.4 |
| 64 | 12 | 322.0 | 0.1988 | 51.5261 | 24.00 | 3456.9 |
| 64 | 16 | 435.1 | 0.1471 | 52.8600 | 24.00 | 3674.2 |
