import re

# Read the error file
with open("error_file.txt", "r") as f:
    errors = f.readlines()

# Parse the lines with errors and error types
error_lines = {}
for error in errors:
    match = re.match(r"(.*):(\d+):\d+: error:.*\[(.*)\]", error)
    if match:
        file_path, line_number, error_type = match.groups()
        if file_path not in error_lines:
            error_lines[file_path] = {}
        error_lines[file_path][int(line_number)] = error_type

# Insert ignore comments in the source files
for file_path, lines in error_lines.items():
    with open(file_path, "r") as f:
        code = f.readlines()
    for line_number, error_type in sorted(lines.items(), key=lambda x: x[0], reverse=True):
        code[line_number - 1] = code[line_number - 1].rstrip() + f"  # type: ignore[{error_type}]\n"
    with open(file_path, "w") as f:
        f.writelines(code)