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

## Run the GPU smoke test

List the concrete test cases first:

```bash
reframe -C config/berzelius.py -c checks -R -lC
```

Generate and inspect the job script without submitting it:

```bash
reframe -C config/berzelius.py -c checks -R \
    --prefix="$PWD/rfm-runs" --dry-run
find rfm-runs/stage -name rfm_job.sh -print
```

Submit the test:

```bash
reframe -C config/berzelius.py -c checks -R \
    --prefix="$PWD/rfm-runs" -r
```

