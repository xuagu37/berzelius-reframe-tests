import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import (
    performance_function,
    run_before,
    sanity_function,
)


@rfm.simple_test
class GpuGemmHbmPilot(rfm.RegressionTest):
    """Collect threshold-free single-GPU compute and HBM observations."""

    descr = 'Single-GPU tensor-core GEMM and HBM-bandwidth pilot'
    valid_systems = ['+pilot_gpu']
    valid_prog_environs = ['cuda']
    tags = {'pilot', 'performance', 'component', 'gpu', 'single_gpu'}

    sourcepath = 'gpu_gemm_hbm.cu'
    build_system = 'SingleSource'
    executable_opts = [
        '--gemm-n=16384',
        '--gemm-warmup=100',
        '--gemm-samples=50',
        '--hbm-mib=1024',
        '--hbm-warmup=20',
        '--hbm-samples=50',
        '--hbm-kernels-per-sample=20',
        '--output=pilot_samples.csv',
    ]

    num_tasks = 1
    num_tasks_per_node = 1
    time_limit = '10m'
    extra_resources = {
        'gpu': {
            'count': 1,
        }
    }
    keep_files = ['pilot_samples.csv']
    # Baseline collection is observational: add references only after the
    # baseline data have been separated into training and evaluation sets.
    reference = {}

    @run_before('compile')
    def set_cuda_build_options(self):
        self.build_system.cxxflags = [
            '-O3',
            '-std=c++17',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_90,code=sm_90',
        ]
        self.build_system.ldflags = ['-lcublas']

    @sanity_function
    def validate_result(self):
        return sn.all([
            sn.assert_found(r'^SANITY_PASS$', self.stdout),
            sn.assert_found(r'^GPU_NAME=\S.*$', self.stdout),
            sn.assert_found(r'^GPU_UUID=[0-9a-f-]+$', self.stdout),
            sn.assert_found(r'^GEMM_MEDIAN_TFLOPS=\S+$', self.stdout),
            sn.assert_found(r'^HBM_MEDIAN_GBPS=\S+$', self.stdout),
        ])

    @performance_function('TFLOP/s')
    def gemm_median_tflops(self):
        return sn.extractsingle(
            r'^GEMM_MEDIAN_TFLOPS=(\S+)$', self.stdout, 1, float
        )

    @performance_function('TFLOP/s')
    def gemm_p05_tflops(self):
        return sn.extractsingle(
            r'^GEMM_P05_TFLOPS=(\S+)$', self.stdout, 1, float
        )

    @performance_function('%')
    def gemm_cv_percent(self):
        return sn.extractsingle(
            r'^GEMM_CV_PERCENT=(\S+)$', self.stdout, 1, float
        )

    @performance_function('GB/s')
    def hbm_median_gbps(self):
        return sn.extractsingle(
            r'^HBM_MEDIAN_GBPS=(\S+)$', self.stdout, 1, float
        )

    @performance_function('GB/s')
    def hbm_p05_gbps(self):
        return sn.extractsingle(
            r'^HBM_P05_GBPS=(\S+)$', self.stdout, 1, float
        )

    @performance_function('%')
    def hbm_cv_percent(self):
        return sn.extractsingle(
            r'^HBM_CV_PERCENT=(\S+)$', self.stdout, 1, float
        )
