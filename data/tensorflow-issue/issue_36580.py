import tensorflow as tf

# Instantiate the writer.
writer = tf.python_io.TFRecordWriter(outputImageFile)

# Every patch-worth of predictions we'll dump an example into the output
# file with a single feature that holds our predictions. Since our predictions
# are already in the order of the exported data, the patches we create here
# will also be in the right order.
patch = [[], [], [], []]
curPatch = 1
for prediction in predictions:
  patch[0].append(tf.argmax(prediction, 1))
  patch[1].append(prediction[0][0])
  patch[2].append(prediction[0][1])
  patch[3].append(prediction[0][2])
  # Once we've seen a patches-worth of class_ids...
  if (len(patch[0]) == PATCH_WIDTH * PATCH_HEIGHT):
    print('Done with patch ' + str(curPatch) + ' of ' + str(PATCHES) + '...')
    # Create an example
    example = tf.train.Example(
      features=tf.train.Features(
        feature={
          'prediction': tf.train.Feature(
              int64_list=tf.train.Int64List(
                  value=patch[0])),
          'bareProb': tf.train.Feature(
              float_list=tf.train.FloatList(
                  value=patch[1])),
          'vegProb': tf.train.Feature(
              float_list=tf.train.FloatList(
                  value=patch[2])),
          'waterProb': tf.train.Feature(
              float_list=tf.train.FloatList(
                  value=patch[3])),
        }
      )
    )
    # Write the example to the file and clear our patch array so it's ready for
    # another batch of class ids
    writer.write(example.SerializeToString())
    patch = [[], [], [], []]
    curPatch += 1

writer.close()