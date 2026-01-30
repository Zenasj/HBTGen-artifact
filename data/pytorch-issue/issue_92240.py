__all__ = [
    'script',
    'trace',
    'script_if_tracing',
    'trace_module',
    'fork',
    'wait',
    'ScriptModule',
    'ScriptFunction',
    'CompilationUnit',
    'freeze',
    'optimize_for_inference',
    'enable_onednn_fusion',
    'export_opnames',
    'onednn_fusion_enabled',
    'set_fusion_strategy',
    'strict_fusion',
    'save',
    'load',
    'ignore',
    'unused',
    'isinstance',
    'Attribute',
    'Error',
    'Future',
    'annotate',
]

import torch.jit

torch.jit.save  # Pylance: "save" is not exported from module "torch.jit.jit"