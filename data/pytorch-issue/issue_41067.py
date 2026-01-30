for process in processes:
        process.wait()

import subprocess as sp

p = sp.Popen(["sleep", "100"])
p.kill()
p.wait()
print(f"done waiting for proc = {p.pid}, exit_code = {p.returncode}")

for process in processes:
        process.wait()