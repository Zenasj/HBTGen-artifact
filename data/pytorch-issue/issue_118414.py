import glob

exclude_patterns = [
    ...
]

for pattern in exclude_patterns:
    for filepath in glob.glob(pattern, recursive=True):
        if filepath.endswith('.py'):
            with open(filepath, 'r+') as f:
                content = f.read()
                f.seek(0, 0)
                f.write('# mypy: ignore-errors\n\n' + content)