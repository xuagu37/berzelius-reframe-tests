# Berzelius ReFrame tests

This repository contains ReFrame tests intended to run from a Berzelius login
node and submit their workloads to compute nodes through Slurm.

## Install ReFrame

```bash
module load Miniforge3
mamba create -n reframe-hpc python=3.11
mamba activate reframe-hpc
pip install reframe-hpc
```

If pip repeatedly warns that `pypi.ngc.nvidia.com` cannot be resolved, install
from the standard Python package index only:

```bash
PIP_CONFIG_FILE=/dev/null \
pip install --index-url https://pypi.org/simple reframe-hpc
```

Verify the installation:

```bash
reframe -V
```

## Configure the Slurm account

Use `projinfo` to find the project account, then export it before running
ReFrame:

```bash
projinfo
export RFM_ACCOUNT=<project-account>
```

If `RFM_ACCOUNT` is not set, the configuration does not emit an `-A` option.
This may work for users with a default account, but setting it explicitly is
recommended.

## Hardware contexts

The configuration exposes separate ReFrame partitions so that measurements from
different GPU types are never mixed accidentally:

| ReFrame system/partition | Berzelius nodes | Pilot enabled |
| --- | --- | --- |
| `berzelius-ampere:a100_40` | A100 40 GB, `-C thin` | yes |
| `berzelius-ampere:a100_80` | A100 80 GB, `-C fat` | no |
| `berzelius-hopper:h200` | H200 141 GB | yes |

The A100 80 GB context is intentionally excluded from the initial pilot because
fat-node GPU-hours cost twice as much. It can be enabled later after the pilot
method is stable.

## Run the GPU smoke test

List the concrete test cases first:

```bash
reframe -C config/berzelius.py -c checks -R \
    -n NvidiaSmiCheck -lC
```

Generate and inspect the job script without submitting it:

```bash
reframe -C config/berzelius.py -c checks -R \
    -n NvidiaSmiCheck \
    --prefix="$PWD/rfm-runs" --dry-run
find rfm-runs/stage -name rfm_job.sh -print
```

Submit the test:

```bash
reframe -C config/berzelius.py -c checks -R \
    -n NvidiaSmiCheck \
    --prefix="$PWD/rfm-runs" -r
```

On an Ampere login node, the `+pilot_gpu` selector resolves to `a100_40`. On a
Hopper login node, it resolves to `h200`. Run the smoke test once from each
login environment if the project has access to both systems.

## Run the single-GPU GEMM/HBM pilot

The pilot is implemented in `checks/gpu_gemm_hbm.py`. It compiles a small CUDA
program with the NSC CUDA build environment and measures:

- FP16 tensor-core GEMM with FP32 accumulation, in TFLOP/s;
- a STREAM-style device-memory triad, in decimal GB/s;
- the fifth percentile and coefficient of variation for both metrics;
- numerical correctness for both kernels.

The default CUDA module is
`buildenv-gcccuda/12.1.1-gcc12.3.0`. Confirm that it is available:

```bash
module avail buildenv-gcccuda
```

If a different installed module must be used, select it without editing the
configuration:

```bash
export RFM_CUDA_MODULE=<installed-buildenv-gcccuda-module>
```

List the pilot cases:

```bash
reframe -C config/berzelius.py -c checks -R \
    -n GpuGemmHbmPilot -lC
```

Generate the job script without submitting it:

```bash
reframe -C config/berzelius.py -c checks -R \
    -n GpuGemmHbmPilot \
    --prefix="$PWD/rfm-runs" --dry-run
find rfm-runs/stage -name rfm_job.sh -print
```

Run one observation and save a uniquely named ReFrame report:

```bash
mkdir -p rfm-runs/reports
reframe -C config/berzelius.py -c checks -R \
    -n GpuGemmHbmPilot \
    --prefix="$PWD/rfm-runs" \
    --report-file="$PWD/rfm-runs/reports/report-{sessionid}.json" \
    --performance-report -r
```

There are deliberately no performance reference thresholds yet. A successful
run must pass the numerical sanity checks, but its performance values are only
recorded. ReFrame also preserves `pilot_samples.csv` in the test output
directory; this contains the raw per-sample timing and throughput values.

Find the resulting artifacts with:

```bash
find rfm-runs/output -type f \
    \( -name 'pilot_samples.csv' -o -name 'rfm_job.out' \) -print
```

### Initial collection plan

1. Run one observation on A100 40 GB and inspect the output and raw CSV.
2. Run one observation on H200 and confirm the same check works unchanged.
3. Run three independent observations on each system as a shake-down set.
4. If the results look plausible and the within-run CV is acceptable, collect
   ten independent observations per system over several days.

Do not submit all ten observations simultaneously. Independent allocations at
different times are more useful for this pilot than many back-to-back samples
inside one Slurm job.
