import os

def enum_files(root_dir: str):
    module_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py"):
                module_files.append(os.path.join(root, file))
    return module_files


def count_num(file_path, str_to_count):
    file=open(file_path)
    try:
        ctx = file.read()
        count_num = ctx.count(str_to_count)
    except:
        return 0
    return count_num


def count_str_in_dir(dir_path, str_to_count):
    files = enum_files(dir_path)
    for i in files:
        num = count_num(i, str_to_count)
        if num != 0:
            print(f"{i}: {str_to_count} num: {num}")

def count_skip_windows_uts():
    cur_dir = os.getcwd()
    dynamo_dir = os.path.join(cur_dir, "test", "dynamo")
    inductor_dir = os.path.join(cur_dir, "test", "inductor")
    export_dir = os.path.join(cur_dir, "test", "export")

    print("Skip UTs:")
    count_str_in_dir(dynamo_dir, "@skipIfWindows")
    count_str_in_dir(inductor_dir, "@skipIfWindows")
    count_str_in_dir(export_dir, "@skipIfWindows")
    
    print("Skip Blocks:")
    count_str_in_dir(dynamo_dir, "if not IS_WINDOWS:")
    count_str_in_dir(inductor_dir, "if not IS_WINDOWS:")
    count_str_in_dir(export_dir, "if not IS_WINDOWS:")

    print("Skip whole file:")
    count_str_in_dir(dynamo_dir, "if IS_WINDOWS and IS_CI:")
    count_str_in_dir(inductor_dir, "if IS_WINDOWS and IS_CI:")
    count_str_in_dir(export_dir, "if IS_WINDOWS and IS_CI:")

if __name__ == "__main__":
    count_skip_windows_uts()