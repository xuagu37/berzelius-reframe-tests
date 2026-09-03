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

## Quick start

The `berzelius-tests` runner wraps the longer ReFrame commands. On its first
real run it asks for the Slurm project account and remembers it in
`berzelius.env`.

Project settings are kept in the visible `berzelius.env` file. For example:

```bash
RFM_ACCOUNT=nsc
RFM_ENV=reframe-hpc
RFM_CUDA_MODULE=
```

The runner loads this file automatically; there is no need to run
`source berzelius.env` manually. Values explicitly exported in the current
shell take precedence over values in the file.

```bash
# Smoke check followed by one GEMM/HBM observation:
./berzelius-tests first-run

# Normal subsequent observations:
./berzelius-tests run

# `run` is the default, so this is equivalent:
./berzelius-tests

# Find all preserved reports and raw samples:
./berzelius-tests results
```

Each observation gets a timestamped directory under `rfm-runs/sessions`, so a
later run does not overwrite an earlier `pilot_samples.csv`. Other available
actions are:

```bash
./berzelius-tests list
./berzelius-tests dry-run
./berzelius-tests smoke
./berzelius-tests configure
./berzelius-tests help
```

With `RFM_ENV=reframe-hpc`, the runner loads `Miniforge3` only when `mamba` is
not already available, activates the named environment only when needed, and
then calls `reframe` directly.

## Configure the Slurm account

Use the runner to find and save the project account:

```bash
./berzelius-tests configure
```

The account is stored in `berzelius.env`. You can also export `RFM_ACCOUNT`
before running the tests; an exported value takes precedence over the file.

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

By default, the runner does not load a CUDA module and uses CUDA from the
system or current job environment. To load a specific build environment,
confirm that it is available:

```bash
module avail buildenv-gcccuda
```

Then set it in `berzelius.env` without editing the ReFrame configuration:

```bash
RFM_CUDA_MODULE=<installed-buildenv-gcccuda-module>
```

Leave `RFM_CUDA_MODULE=` empty to continue using the system/default CUDA.

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
