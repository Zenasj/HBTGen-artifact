import torch

if config.checkpoint_iters is not None and self.iter_num % config.checkpoint_iters == 0:
                model_state_dict = model.state_dict()
                sd = {'module': model_state_dict, 'iteration': self.iter_num}
                fs_storage_writer = FileSystemWriter("/tmp/checkpoint")
                save_state_dict(
                    state_dict=sd,
                    storage_writer=fs_storage_writer,
                )

if args.start_from_checkpoint:
        sd = {'module': model.state_dict(), 'iteration': 0}
        fs_storage_reader = torch.distributed.checkpoint.FileSystemReader("/tmp/checkpoint")
        load_state_dict(
            state_dict=sd,
            storage_reader=fs_storage_reader,
        )