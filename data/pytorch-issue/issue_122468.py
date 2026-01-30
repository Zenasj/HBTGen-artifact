import torch

from importlib.metadata import entry_points
discovered_plugins = entry_points(group='torch.plugins')
for plugin in discovered_plugins:
    try:
        plugin.load()
    except IndexError:
        pass

from importlib.metadata import entry_points
discovered_plugins = entry_points(group='torch.plugins')
for plugin in discovered_plugins:
    try:
        plugin.load()
    except IndexError:
        pass