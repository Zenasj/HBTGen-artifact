def autodoc_skip_member(app, what, name, obj, skip, options):
    exclusions = ('distributed', 'others?')
    if name in exclusions:
        spec = find_spec('.' + name, 'torch') # checks if submodule exists, returns None otherwise
        exclude = spec is None
    else:
        exclude = False
    return skip or exclude

try:
    import torchvision
    from torch.distributed import init_process_group
except ImportError:
    # Suppress autodoc import warnings.
    suppress_warnings = ['autodoc.import_object']