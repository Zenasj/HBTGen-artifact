import os
import shutil
import sys
import zipfile
import boto3

s3 = boto3.client('s3')

zip_requirements = "/tmp/pytorch1.0.1_lambda_deps.zip"
pkgdir = '/tmp/sls-py-req'
sys.path.append(pkgdir)
if not os.path.exists(pkgdir):
    tempdir = '/tmp/_temp-sls-py-req'
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

    s3.download_file(
                YOUR_BUCKET,"pytorch1.0.1_lambda_deps.zip", zip_requirements)
    zipfile.ZipFile(zip_requirements, 'r').extractall(tempdir)
    os.remove(zip_requirements)
    os.rename(tempdir, pkgdir)  # Atomic
print("Deps extracted successfully !")

extra_compile_args = {"cxx": [
        '-Wl,-rpath,$ORIGIN'
    ]}