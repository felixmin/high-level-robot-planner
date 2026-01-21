"""
Benchmark flow supervision cost (RAFT teacher + FlowDecoder).

Usage (GPU recommended):
  PYTHONPATH=packages python scripts/bench_flow_supervision.py --device cuda --raft raft_large
"""

import argparse
import time

import torch


def _timeit(fn, *, iters: int, warmup: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--raft", default="raft_small", choices=["raft_small", "raft_large"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--grid", type=int, default=8, help="effective grid size (h=w)")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim-head", type=int, default=64)
    parser.add_argument("--decoder-depth", type=int, default=4)
    parser.add_argument("--num-flow-updates", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available")

    from laq.models.flow import RAFTTeacher, FlowDecoder

    torch.manual_seed(0)

    b = args.batch_size
    h = w = args.image_size

    # Video frames in [0, 1]
    frame1 = torch.rand(b, 3, 1, h, w, device=device)
    frame2 = torch.rand(b, 3, 1, h, w, device=device)

    teacher = RAFTTeacher(args.raft, chunk_size=args.chunk_size, num_flow_updates=args.num_flow_updates).to(device)

    def run_teacher():
        teacher.compute_flow(frame1, frame2)

    teacher_s = _timeit(run_teacher, iters=args.iters, warmup=args.warmup, device=device)

    # FlowDecoder benchmark (synthetic tokens)
    gh = gw = args.grid
    context_tokens = torch.randn(b, 1, gh, gw, args.dim, device=device)
    action_tokens = torch.randn(b, 1, gh, gw, args.dim, device=device)
    attn_bias = torch.zeros(args.heads, gh * gw, gh * gw, device=device)

    decoder = FlowDecoder(
        dim=args.dim,
        depth=args.decoder_depth,
        heads=args.heads,
        dim_head=args.dim_head,
        image_size=(h, w),
        effective_grid_size=(gh, gw),
    ).to(device)
    decoder.eval()

    def run_decoder():
        decoder(context_tokens, action_tokens, attn_bias)

    decoder_s = _timeit(run_decoder, iters=args.iters, warmup=args.warmup, device=device)

    print(f"RAFT teacher: {teacher_s * 1000:.2f} ms/iter")
    print(f"FlowDecoder: {decoder_s * 1000:.2f} ms/iter")


if __name__ == "__main__":
    main()

