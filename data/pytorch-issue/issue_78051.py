toml
[[linter]]
code = 'CALL_ONCE'
include_patterns = [
    'c10/**',
    'aten/**',
    'torch/csrc/**',
]
command = [
    'python3',
    'tools/linter/adapters/grep_linter.py',
    '--pattern=std::call_once',
    '--linter-name=CALL_ONCE',
    '--error-name=invalid call_once',
    '--replace-pattern=s/std::call_once/c10::call_once/',
    """--error-description=\
        Use of std::call_once is forbidden and should be replaced with c10::call_once\
    """,
    '--',
    '@{{PATHSFILE}}'
]