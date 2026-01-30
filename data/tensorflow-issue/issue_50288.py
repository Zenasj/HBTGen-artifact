import re
from tensorflow.python.keras.utils.generic_utils import to_snake_case
from tensorflow.python.keras.engine.base_layer import Layer

layers = Layer.__subclasses__()


def to_snake_case_proposed(name):
    name = name.replace('ReLU', 'Relu')
    intermediate = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
    # If the class is private the name starts with "_" which is not secure
    # for creating scopes. We prefix the name with "private" in this case.
    if insecure[0] != '_':
        return insecure
    return 'private' + insecure


for layer in sorted(layers, key=lambda x: x.__name__):
    current = to_snake_case(layer.__name__)
    proposed = to_snake_case_proposed(layer.__name__)
    if current != proposed:
        print(current, proposed)