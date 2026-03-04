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
| 32 | 8 | 168.2 | 0.1903 | 46.2124 | 12.00 |
| 64 | 8 | 1921806.5 | 0.0000 | 48.1605 | 24.00 |
| 128 | 8 | 2776693.5 | 0.0000 | 50.2620 | 48.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 168.2 | 0.1903 | 46.2124 | 12.00 | 3087.7 |
| 64 | 8 | 1921806.5 | 0.0000 | 48.1605 | 24.00 | 3272.5 |
| 128 | 8 | 2776693.5 | 0.0000 | 50.2620 | 48.00 | 3743.4 |
