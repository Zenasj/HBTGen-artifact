import random

import os
from tempfile import TemporaryDirectory

import numpy as np
import tensorflow as tf


def as_numpy(ds: tf.data.Dataset):
    return np.array([x.numpy() for x in ds])


def get_data(
    num_repeats=2,
    snap=False,
    preprocess_early=False,
    preprocess_late=False,
    del_rng=False,
):
    """
    Get numpy results from a data pipeline.

    The pipeline looks like:
        1. range
        2. add stateful random noise
        3. create `num_repeats` `cache`d or `snapshot`ted versions
        4. `flat_map` if num_repeats > 1

    Args:
        num_repeats: number of duplicates created in step 3 above.
        snap: use `snapshot` (otherwise use `cache`)
        preprocess_early: if True, we iterate over individually cached / snapshotted
            datasets prior to flat-mapping.
        preprocess_late: if True, we iterate over the `flat_map`ped dataset
        del_rng: if True, we delete the rng responsible for generating random noise in
            step 2. This will cause an error if this map function is called again,
            rather than using cached / snapshotted files on disk.

    Returns:
        Two iterations of the repeated dataset.
    """
    rng = tf.random.Generator.from_seed(0)
    dataset = tf.data.Dataset.range(10).map(
        lambda x: tf.cast(x, tf.float32) + rng.uniform(())
    )
    with TemporaryDirectory() as tmp_dir:
        paths = [os.path.join(tmp_dir, f"repeat-{i}") for i in range(num_repeats)]
        if snap:
            datasets = [
                dataset.apply(tf.data.experimental.snapshot(path)) for path in paths
            ]
        else:
            datasets = [dataset.cache(path) for path in paths]
        if preprocess_early:
            # iterate over datasets individually to force saving to file
            for ds in datasets:
                as_numpy(ds)
        if num_repeats == 1:
            (dataset,) = datasets
        else:
            dataset = tf.data.Dataset.from_tensor_slices(datasets).flat_map(lambda x: x)
        if preprocess_late:
            # iterate over concatenated dataset to force saving to file
            as_numpy(dataset)
        if del_rng:
            # this will cause an error is the original mapped dataset is called
            del rng
        return as_numpy(dataset), as_numpy(dataset)


class SnapshotTest(tf.test.TestCase):
    def test_consistent(self):
        base0, base1 = get_data()
        np.testing.assert_equal(base0, base1)

    def test_reproducible(self):
        base0, _ = get_data()
        s0, s1 = get_data()
        np.testing.assert_equal(s0, s1)
        np.testing.assert_equal(s0, base0)

    def test_snapshot(self):
        base0, _ = get_data()
        s0, s1 = get_data(snap=True)
        np.testing.assert_equal(s0, s1)
        np.testing.assert_equal(s0, base0)

    def test_preprocess_late(self):
        base0, _ = get_data()
        s0, s1 = get_data(snap=True, preprocess_late=True)
        np.testing.assert_equal(s0, s1)
        np.testing.assert_equal(s0, base0)

    def test_preprocess_late_del_rng(self):
        base0, _ = get_data()
        s0, s1 = get_data(snap=True, preprocess_late=True, del_rng=True)
        np.testing.assert_equal(s0, s1)
        np.testing.assert_equal(s0, base0)

    def test_preprocess_early(self):
        base0, _ = get_data()
        s0, s1 = get_data(snap=True, preprocess_early=True)
        np.testing.assert_equal(s0, s1)
        np.testing.assert_equal(s0, base0)

    def test_preprocess_early_del_rng(self):
        base0, _ = get_data()
        s0, s1 = get_data(snap=True, preprocess_early=True, del_rng=True)
        np.testing.assert_equal(s0, s1)
        np.testing.assert_equal(s0, base0)

    def test_preprocess_no_repeats(self):
        # preprocess_early is equivalent to preprocess_late here
        base0, _ = get_data(num_repeats=1)
        s0, s1 = get_data(snap=True, preprocess_early=True, num_repeats=1)
        np.testing.assert_equal(s0, s1)
        np.testing.assert_equal(s0, base0)

    def test_preprocess_del_rng_no_repeats(self):
        # preprocess_early is equivalent to preprocess_late here
        base0, _ = get_data(num_repeats=1)
        s0, s1 = get_data(snap=True, preprocess_early=True, num_repeats=1, del_rng=True)
        np.testing.assert_equal(s0, s1)
        np.testing.assert_equal(s0, base0)


if __name__ == "__main__":
    tf.test.main()

if preprocess_early:
    # iterate over datasets individually to force saving to file
    for ds in datasets:
        as_numpy(tf.data.Dataset.from_tensors(ds).flat_map(lambda x: x))