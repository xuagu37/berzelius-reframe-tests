import os


account = os.getenv('RFM_ACCOUNT')
account_access = [f'-A {account}'] if account else []


def gpu_partition(description):
    return {
        'name': 'gpu',
        'descr': description,
        'scheduler': 'slurm',
        'launcher': 'srun',
        'access': account_access,
        'environs': ['default'],
        'features': ['gpu'],
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
            'partitions': [gpu_partition('A100 GPU compute nodes')],
        },
        {
            'name': 'berzelius-hopper',
            'descr': 'Berzelius Hopper (NVIDIA H200)',
            'hostnames': [r'berzelius-hopper[0-9]+'],
            'modules_system': 'lmod',
            'partitions': [gpu_partition('H200 GPU compute nodes')],
        },
    ],
    'environments': [
        {
            'name': 'default',
        }
    ],
}

