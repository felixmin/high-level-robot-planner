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
| 64 | 8 | 801.9 | 0.0798 | 47.9813 | 24.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 281.2 | 0.2276 | 46.9122 | 24.00 | 2620.5 |
| 64 | 8 | 801.9 | 0.0798 | 47.9813 | 24.00 | 3165.2 |
| 64 | 12 | 246.6 | 0.2595 | 47.6122 | 24.00 | 3686.7 |
