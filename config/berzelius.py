import os


account = os.getenv('RFM_ACCOUNT')
account_access = [f'-A {account}'] if account else []
cuda_module = os.getenv('RFM_CUDA_MODULE', '').strip()


def gpu_partition(name, description, features, access=None):
    return {
        'name': name,
        'descr': description,
        'scheduler': 'slurm',
        'launcher': 'srun',
        'access': account_access + (access or []),
        'environs': ['default', 'cuda'],
        'features': ['gpu', *features],
        'max_jobs': 1,
        'resources': [
            {
                'name': 'gpu',
                'options': ['--gpus={count}'],
            }
        ],
    }


site_configuration = {
    'systems': [
        {
            'name': 'berzelius-ampere',
            'descr': 'Berzelius Ampere (NVIDIA A100)',
            'hostnames': [r'berzelius[0-9]+'],
            'modules_system': 'lmod',
            'partitions': [
                gpu_partition(
                    'a100_40',
                    'A100 40 GB thin nodes',
                    ['a100', 'a100_40', 'thin', 'pilot_gpu'],
                    ['-C thin'],
                ),
                gpu_partition(
                    'a100_80',
                    'A100 80 GB fat nodes',
                    ['a100', 'a100_80', 'fat'],
                    ['-C fat'],
                ),
            ],
        },
        {
            'name': 'berzelius-hopper',
            'descr': 'Berzelius Hopper (NVIDIA H200)',
            'hostnames': [r'berzelius-hopper[0-9]+'],
            'modules_system': 'lmod',
            'partitions': [
                gpu_partition(
                    'h200',
                    'H200 141 GB GPU compute nodes',
                    ['h200', 'pilot_gpu'],
                ),
            ],
        },
    ],
    'environments': [
        {
            'name': 'default',
        },
        {
            'name': 'cuda',
            'modules': [cuda_module] if cuda_module else [],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'nvcc': 'nvcc',
        }
    ],
}
