import resource
import warnings

def check_fd_limit(min_limit):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < min_limit:
        warnings.warn(f"Current file descriptor limit ({soft}) is below the recommended minimum ({min_limit}). "
                      f"This may cause issues. Consider increasing the limit using 'ulimit -n {min_limit}' "
                      "before running the program.")