import sys

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import performance_function, sanity_function


@rfm.simple_test
class SmallDeterministicAiCanary(rfm.RunOnlyRegressionTest):
    """Measure a reproducible single-GPU transformer training workload."""

    descr = 'Small deterministic BF16 transformer training canary'
    valid_systems = ['+pilot_gpu']
    valid_prog_environs = ['default']
    tags = {'pilot', 'performance', 'ai', 'transformer', 'single_gpu'}

    # ReFrame and PyTorch deliberately share the reframe-hpc environment.
    # Using the absolute interpreter path avoids relying on Conda activation in
    # the non-interactive Slurm job shell.
    sourcesdir = 'src'
    executable = sys.executable
    executable_opts = [
        'transformer_canary.py',
        '--seed=1729',
        '--vocab-size=8192',
        '--sequence-length=512',
        '--batch-size=8',
        '--dataset-batches=8',
        '--layers=8',
        '--hidden-size=768',
        '--heads=12',
        '--ffn-size=3072',
        '--warmup-steps=20',
        '--measured-steps=100',
        '--learning-rate=0.0003',
        '--output=canary_samples.csv',
    ]

    num_tasks = 1
    num_tasks_per_node = 1
    time_limit = '10m'
    extra_resources = {
        'gpu': {
            'count': 1,
        }
    }
    env_vars = {
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'PYTHONHASHSEED': '0',
        'PYTHONNOUSERSITE': '1',
    }
    keep_files = ['canary_samples.csv']
    # Collect observations first. Add references only after separating the
    # baseline data into training and evaluation sets.
    reference = {}

    @sanity_function
    def validate_training(self):
        initial_loss = sn.extractsingle(
            r'^INITIAL_LOSS=(\S+)$', self.stdout, 1, float
        )
        final_loss = sn.extractsingle(
            r'^FINAL_LOSS=(\S+)$', self.stdout, 1, float
        )
        gradient_norm = sn.extractsingle(
            r'^GRADIENT_NORM=(\S+)$', self.stdout, 1, float
        )
        return sn.all([
            sn.assert_found(r'^CANARY_SANITY_PASS$', self.stdout),
            sn.assert_found(r'^CUDA_AVAILABLE=True$', self.stdout),
            sn.assert_found(r'^BF16_SUPPORTED=True$', self.stdout),
            sn.assert_found(
                r'^DETERMINISTIC_ALGORITHMS=True$', self.stdout
            ),
            sn.assert_found(r'^GPU_NAME=\S.*$', self.stdout),
            sn.assert_found(r'^TORCH_VERSION=\S+$', self.stdout),
            sn.assert_found(r'^TORCH_CUDA_VERSION=\S+$', self.stdout),
            sn.assert_lt(final_loss, initial_loss),
            sn.assert_gt(gradient_norm, 0.0),
        ])

    @performance_function('tokens/s')
    def median_tokens_per_second(self):
        return sn.extractsingle(
            r'^TOKENS_PER_SECOND_MEDIAN=(\S+)$', self.stdout, 1, float
        )

    @performance_function('tokens/s')
    def p05_tokens_per_second(self):
        return sn.extractsingle(
            r'^TOKENS_PER_SECOND_P05=(\S+)$', self.stdout, 1, float
        )

    @performance_function('ms')
    def median_step_ms(self):
        return sn.extractsingle(
            r'^STEP_MEDIAN_MS=(\S+)$', self.stdout, 1, float
        )

    @performance_function('ms')
    def p95_step_ms(self):
        return sn.extractsingle(
            r'^STEP_P95_MS=(\S+)$', self.stdout, 1, float
        )

    @performance_function('%')
    def step_cv_percent(self):
        return sn.extractsingle(
            r'^STEP_CV_PERCENT=(\S+)$', self.stdout, 1, float
        )

    @performance_function('GiB')
    def max_allocated_memory_gib(self):
        return sn.extractsingle(
            r'^MAX_ALLOCATED_MEMORY_GIB=(\S+)$', self.stdout, 1, float
        )

    @performance_function('%')
    def loss_decrease_percent(self):
        return sn.extractsingle(
            r'^LOSS_DECREASE_PERCENT=(\S+)$', self.stdout, 1, float
        )
