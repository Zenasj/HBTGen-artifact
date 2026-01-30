with FSDP.state_dict_type(
                        model,
                        StateDictType.SHARDED_STATE_DICT,
                ):
                    model_state = model.state_dict()
                    optim_state = FSDP.sharded_optim_state_dict(model, optimizer, group=shard_group)

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'