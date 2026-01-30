import subprocess
import argparse
import pathlib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--torch-install-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "third_party" / "pytorch" / "torch",
    )
    args = parser.parse_args()

    out = subprocess.check_output(
        "find -type f -executable",
        cwd=args.torch_install_dir,
        shell=True,
    )

    executables = out.decode(encoding="utf-8").splitlines()
    problematic_executables: str = []
    for exe in executables:
        print(exe)
        if ".so" in exe or ".py" in exe:
            continue
        try:
            out = subprocess.check_output(
                exe, cwd=args.torch_install_dir, shell=True, stderr=subprocess.STDOUT
            )
        except subprocess.CalledProcessError as e:
            if "error while loading shared libraries" in e.output.decode(
                encoding="utf-8"
            ):
                problematic_executables.append(exe)
            else:
                print(e)

    with open("torch_executables_missing_rpath.txt", "w") as f:
        f.write("\n".join(problematic_executables))