# speed_benchmark

**Status:** tentative
**Code:** branch `exp/speed-benchmark`, PR #<n>

## Question

Which device and level of parallelism finishes training runs fastest on this machine (M1 Pro, 6 performance + 2
efficiency cores, 16 GB)? Until now every run trained on MPS, one job at a time.

## Protocol deviations

None to the science protocol. This adds the `device` key (`docs/protocol.md`, Device row; the default `auto` keeps MPS).

## Run set

`configs/experiment/speed_benchmark.yaml`: shred, flow_trunk_shred_w256, flow_trunk_shred_div; clean training, fixed
100 epochs (no early stopping), seed 0, one clean eval condition. Workload per setup: 6 identical jobs = 18 runs.
Setups (device, N parallel jobs × T threads each via `OMP_NUM_THREADS`), 2 repeats, run round-robin:

```
experiments/speed_benchmark/bench.sh     # -> outputs/speed_benchmark/timings.csv
```

## Results

Makespan for the 18 runs; throughput = 18 / median makespan.

| Setup | Repeat 1 (s) | Repeat 2 (s) | Runs / min | vs. current (mps_1) |
|---|---|---|---|---|
| hybrid: 2 MPS jobs + 4 CPU jobs × 1 thread | 49.5 | 51.1 | 21.5 | 4.7× |
| cpu 6 × 1 | 52.5 | 53.2 | 20.4 | 4.5× |
| cpu 3 × 2 | 77.6 | 75.6 | 14.1 | 3.1× |
| mps 3 jobs | 96.2 | 85.0 | 11.9 | 2.6× |
| cpu 2 × 3 | 98.4 | 88.5 | 11.6 | 2.5× |
| mps 2 jobs | 137.8 | 128.5 | 8.1 | 1.8× |
| cpu 1 × 6 | 149.3 | 140.9 | 7.4 | 1.6× |
| cpu 1 × 1 | 201.9 | 192.7 | 5.5 | 1.2× |
| mps 1 job (what we ran so far) | 249.2 | 222.6 | 4.6 | 1.0× |

Median per-run `wall_time_s` for a single job (no contention): shred 4.1 s on CPU (6 threads) vs 8.0 s on MPS;
flow_trunk_shred_w256 4.8 vs 7.9 s; flow_trunk_shred_div 11.7 vs 19.2 s. CPU beats MPS even for one job.

Same seed, same device: the same test error in every repeat (to 1e-6). CPU vs MPS at 100 epochs:
shred 0.0104 vs 0.0086, flow_trunk_shred_w256 0.2283 vs 0.2291, flow_trunk_shred_div 0.2303 vs 0.2302.

## Conclusion

MPS is the slowest choice here: the models are too small to use the GPU. The fastest setups are **6 single-thread
CPU jobs** (20.4 runs/min) and the hybrid (21.5 runs/min), both ≈ 4.5–4.7× faster than one MPS job. The hybrid wins
by only 5 %, and needs two device groups, so the practical recommendation is **6 CPU jobs × 1 thread**.

Caveats:
- The device changes the numbers (shred differs by ~20 % at 100 epochs), so never mix devices within one comparison;
  record the device (it is in every `result.json`).
- Parallelism only helps when a run set splits into ≥ 6 jobs. `run.py` runs a job's runs one after the other.
- Measured at 100 epochs; real runs use 1000–5000, so per-job start-up (data load) counts for even less there.
- 2 repeats only; the top two do not overlap but are 5 % apart.

Recipe (split by seed, up to 6 jobs at once, so 3 seeds use only 3 cores; each job writes its own `seed<i>/` folder; one final call with all seeds skips everything
and rewrites the full `summary.json`):

```
printf '%s\n' 0 1 2 | OMP_NUM_THREADS=1 xargs -P 6 -I{} \
  uv run python scripts/run.py +experiment=<name> device=cpu 'eval.seeds=[{}]'
uv run python scripts/run.py +experiment=<name> device=cpu
```
