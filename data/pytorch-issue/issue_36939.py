import torch

if dist.is_nccl_available():
        DISTRIBUTED_TESTS_CONFIG['nccl'] = {
            'WORLD_SIZE': '2' if torch.cuda.device_count() == 2 else '3',
            'TEST_REPORT_SOURCE_OVERRIDE': 'dist-nccl'
        }