# LeRobot v3 Dataloader Benchmark

| scenario | batch_size | workers | samples/s | mean_batch_s | first_batch_s | payload_mb |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage1_single_nyu | 8 | 0 | 105.2 | 0.0761 | 0.3470 | 0.42 |
| stage1_single_nyu | 8 | 2 | 910.4 | 0.0088 | 0.0559 | 0.42 |
| stage2_nyu | 8 | 0 | 141.9 | 0.0564 | 0.0675 | 0.43 |
| stage2_nyu | 8 | 2 | 1016.7 | 0.0079 | 0.1297 | 0.43 |
