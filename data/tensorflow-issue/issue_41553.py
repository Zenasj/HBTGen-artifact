import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    nb_checkpoints = 3

    inputs = layers.Input(shape=(8,))
    x = layers.Dense(2)(inputs)
    outputs = layers.Dense(2)(x)
    model = keras.Model(inputs=[inputs], outputs=[outputs])

    model_dir = '/tmp/save-weights'
    for i in range(nb_checkpoints):
        model.save_weights(os.path.join(model_dir, 'ckpt-%04d' % (i + 1)))

    state = tf.train.get_checkpoint_state(model_dir)
    print('')
    print(state.all_model_checkpoint_paths)

    checkpoint_fp = os.path.join(model_dir, 'checkpoint')
    print('\nContent of %s' % checkpoint_fp)
    with open(checkpoint_fp) as f:
        print(f.read())

    assert state.all_model_checkpoint_paths == nb_checkpoints, \
        'Expected %d checkpoints got %d' % (nb_checkpoints, len(state.all_model_checkpoint_paths))


if __name__ == '__main__':
    main()