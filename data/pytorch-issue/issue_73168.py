if not use_ddp or dist.get_rank() == 0:
        logger.info("begin save checkpoint")
        save_checkpoint(...)
if use_ddp:
        dist.barrier()