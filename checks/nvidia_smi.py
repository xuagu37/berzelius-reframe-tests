import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import sanity_function


@rfm.simple_test
class NvidiaSmiCheck(rfm.RunOnlyRegressionTest):
    descr = 'Verify that a scheduled NVIDIA GPU is visible'
    valid_systems = ['+gpu']
    valid_prog_environs = ['*']

    executable = 'nvidia-smi'
    executable_opts = [
        '--query-gpu=name,uuid',
        '--format=csv,noheader',
    ]

    num_tasks = 1
    time_limit = '5m'
    extra_resources = {
        'gpu': {
            'count': 1,
        }
    }

    @sanity_function
    def validate_gpu(self):
        return sn.assert_found(r'NVIDIA', self.stdout)
