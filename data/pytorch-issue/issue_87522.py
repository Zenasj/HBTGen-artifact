import torch

if torch._dynamo.config.log_code:
    log.info("...")