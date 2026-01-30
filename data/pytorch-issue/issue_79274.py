import torch

#13 17.58 Collecting torch==1.7.1+cu110
#13 17.59   Downloading https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp38-cp38-linux_x86_64.whl (1156.8 MB)
#13 45.83 ERROR: Exception:
#13 45.83 Traceback (most recent call last):
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/cli/base_command.py", line 186, in _main
#13 45.83     status = self.run(options, args)
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/commands/install.py", line 357, in run
#13 45.83     resolver.resolve(requirement_set)
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/legacy_resolve.py", line 177, in resolve
#13 45.83     discovered_reqs.extend(self._resolve_one(requirement_set, req))
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/legacy_resolve.py", line 333, in _resolve_one
#13 45.83     abstract_dist = self._get_abstract_dist_for(req_to_install)
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/legacy_resolve.py", line 282, in _get_abstract_dist_for
#13 45.83     abstract_dist = self.preparer.prepare_linked_requirement(req)
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/operations/prepare.py", line 480, in prepare_linked_requirement
#13 45.83     local_path = unpack_url(
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/operations/prepare.py", line 282, in unpack_url
#13 45.83     return unpack_http_url(
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/operations/prepare.py", line 164, in unpack_http_url
#13 45.83     unpack_file(from_path, location, content_type)
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/utils/unpacking.py", line 249, in unpack_file
#13 45.83     unzip_file(
#13 45.83   File "/root/venv/lib/python3.8/site-packages/pip/_internal/utils/unpacking.py", line 139, in unzip_file
#13 45.83     shutil.copyfileobj(fp, destfp)
#13 45.83   File "/usr/lib/python3.8/shutil.py", line 205, in copyfileobj
#13 45.83     buf = fsrc_read(length)
#13 45.83   File "/usr/lib/python3.8/zipfile.py", line 940, in read
#13 45.83     data = self._read1(n)
#13 45.83   File "/usr/lib/python3.8/zipfile.py", line 1016, in _read1
#13 45.83     data = self._decompressor.decompress(data, n)
#13 45.83 zlib.error: Error -3 while decompressing data: invalid code lengths set
#13 ERROR: executor failed running [/bin/sh -c pip install     --extra-index-url https://download.pytorch.org/whl/     -r /requirements-freeze.txt]: exit code: 2