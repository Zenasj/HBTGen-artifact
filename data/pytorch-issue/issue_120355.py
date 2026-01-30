import os
current_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(current_dir, 'dynamo_expected_failures')
for name in dynamo_expected_failures:
    path = os.path.join(directory, name)
    with open(path, 'w') as fp:
        pass