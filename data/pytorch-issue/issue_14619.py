# before
version = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion'])
# after
version = subprocess.check_output([compiler, '-dumpfullversion', '-dumpversion']).decode('utf-8')