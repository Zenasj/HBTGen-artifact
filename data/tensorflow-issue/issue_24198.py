import tensorflow as tf

@tf_export("data.TextLineBlockDataset")
class TextLineBlockDataset(dataset_ops.Dataset):
    """A `Dataset` comprising lines from one or more text file blocks."""

    def __init__(self, filenames, begin_offsets, end_offsets, buffer_size=None):
        """Creates a `TextLineBlockDataset`.
        An element in zip(filenames, begin_offsets, end_offsets) denotes a text
        file block in file filename, beginning at begin_offset and ends at
        end_offset, in byte.
        `begin_offsets` and `end_offsets` will be smoothed to match text line
        boundaries under the hood.

        Args:
          filenames: A `tf.string` tensor containing one or more filenames.
          begin_offsets: A `tf.int64` 1-d tensor denoting begin offsets.
          end_offsets: A `tf.int64` 1-d tensor denoting end offsets.
          buffer_size: (Optional.) A `tf.int64` scalar denoting the number of bytes
            to buffer for each block, for each block. A value of 0 results in the 
            default values chosen on the compression type.
        """

if __name__ == '__main__':
    filenames = ["data/train.txt", "data/dev.txt", "data/test.txt"]
    n_files = len(filenames)
    # this is the 2nd of the 4 towers which consumes range 2/4 ~ 3/4 text lines of each file
    n_towers, tower_id = 4, 2
    n_splits = 8    # each range will be further split into 8 parts for better shuffle
    buffer_size = 256 * 1024

    block_params = []
    for file_id in range(n_files):
        file_size = os.path.getsize(filenames[file_id])
        split_size = file_size / (n_towers * n_splits)
        for split_id in range(n_splits):
            begin_offset = int(split_size * (tower_id * n_splits + split_id))
            end_offset = int(split_size * (tower_id * n_splits + split_id + 1))
            block_params.append((filenames[file_id], begin_offset, end_offset))
    dataset = TextLineBlockDataset(*zip(*block_params), buffer_size)
    dataset = dataset.shuffle(32000)  # shuffle in the same manner as `TextLineDataset`
    get_next = dataset.make_one_shot_iterator().get_next()

    sess = tf.Session()
    while True:
        try:
            print(sess.run(get_next).decode("utf-8"))
        except tf.errors.OutOfRangeError:
            break