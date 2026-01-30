py
FUNCTIONALS_WITHOUT_ANNOTATION = (
        "adaptive_max_pool1d",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "fractional_max_pool2d",
        "fractional_max_pool3d",
        "max_pool1d",
        "max_pool2d",
        "max_pool3d",
        "gaussian_nll_loss",
        "upsample",
        "upsample_bilinear",
        "upsample_nearest",
    )

py
class P:
    def __iter__(self):
        raise RuntimeError

if __name__ == '__main__':
    a = P()
    if 1 in a:
        pass