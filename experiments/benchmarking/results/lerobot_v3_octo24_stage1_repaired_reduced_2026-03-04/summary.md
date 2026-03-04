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
| 64 | 8 | 482.8 | 0.1326 | 47.0071 | 24.00 |

## Full results

| batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb | rss_after_run_mb |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 4 | 133.6 | 0.4791 | 46.1174 | 24.00 | 2616.3 |
| 64 | 8 | 482.8 | 0.1326 | 47.0071 | 24.00 | 3149.2 |
| 64 | 12 | 262.5 | 0.2438 | 48.4061 | 24.00 | 3606.2 |
