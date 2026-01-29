# tf.random.uniform((1, 192, 192, 3), dtype=tf.uint8)  # Inferred input shape and dtype from usage of input image resized to (width, height)

import math
import tensorflow as tf
import numpy as np

NUM_RESULTS = 1917
NUM_CLASSES = 91

X_SCALE = 10.0
Y_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

# Placeholder for box priors to be loaded externally
box_priors = []  # Will hold 4 lists: [ycenter, xcenter, h, w] each of length NUM_RESULTS


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # This model wraps a TensorFlow Lite interpreter internally
        # Since the original script calls TFLite interpreter, here we simulate
        # post-processing decoding and filtering in TensorFlow.
        # The underlying TFLite model is not implemented here as TF Keras layer,
        # because TFLite interpreter runs an external model.
        # We'll implement the decode and NMS logic only.

    def call(self, inputs):
        """
        `inputs` is expected to be a tensor of shape [1, H, W, 3] uint8 image
        This method will simulate output based on decoded boxes and scores.
        Since TFLite interpreter run is external, we simulate the post-processing pipeline:
         - dummy location predictions and class logits (simulated)
         - decode boxes using box_priors
         - compute pruned_predictions (scores > threshold)
         - apply NMS
         - return final predictions (scores, class_id, label, box coords)

        For demonstration, this dummy implementation returns fixed random predictions.
        A real-world usage expects TFLite interpreter outputs to be inputs to this logic.
        """

        batch_size = tf.shape(inputs)[0]
        assert batch_size == 1, "Only batch size 1 is supported as per original script"

        # Because this is a stub, simulate model outputs for locations and classes
        # locations shape: [NUM_RESULTS, 4]
        # classes shape: [NUM_RESULTS, NUM_CLASSES]
        # We create dummy tensors to allow code flow

        # Simulate locations with random values centered around box_priors values for demonstration
        box_priors_tf = tf.convert_to_tensor(box_priors, tf.float32)  # shape (4, NUM_RESULTS)
        # We'll reshape box_priors from list of lists to tensor shape [4, NUM_RESULTS]
        # box_priors are [ycenter, xcenter, h, w], each list length NUM_RESULTS
        
        # If box_priors do not have NumPy-compatible shape yet, return empty output
        if not box_priors or len(box_priors) < 4:
            # Return empty predictions
            return tf.constant([], shape=(0, 6), dtype=tf.float32)  # no predictions

        # Extract lists
        ycenter_prior = tf.squeeze(box_priors_tf[0])  # shape (NUM_RESULTS,)
        xcenter_prior = tf.squeeze(box_priors_tf[1])
        h_prior = tf.squeeze(box_priors_tf[2])
        w_prior = tf.squeeze(box_priors_tf[3])

        # Generate dummy "locations" around zero which later decode with prior boxes
        # Shape: [NUM_RESULTS,4] (ycenter_adj, xcenter_adj, h_adj, w_adj)
        # Using zeros which means decoded boxes == priors for simplicity
        locations = tf.zeros((NUM_RESULTS, 4), dtype=tf.float32)

        # locations tensor for decode_center_size_boxes: [NUM_RESULTS, 4]
        # where each location: [ycenter, xcenter, h, w] adjusted offsets

        # For demonstration we set locations to zeros and decode them to priors
        decoded_boxes = self.decode_center_size_boxes(locations, ycenter_prior, xcenter_prior, h_prior, w_prior)
        # shape: [NUM_RESULTS,4] (ymin, xmin, ymax, xmax), all normalized in [0,1]

        # Simulate class logits with random values for example (NUM_RESULTS, NUM_CLASSES)
        class_logits = tf.random.uniform((NUM_RESULTS, NUM_CLASSES), minval=-3.0, maxval=3.0, dtype=tf.float32)

        # Apply sigmoid to get scores
        scores = tf.math.sigmoid(class_logits)

        # Threshold for scores pruning (original set to 0.01)
        score_threshold = 0.01

        # Prepare pruned_predictions: list of (score, index, class_id, box) tuples
        # We'll flatten and filter within TensorFlow for efficiency

        # Broadcast boxes to classes:
        # scores shape (NUM_RESULTS, NUM_CLASSES), shape compatible with boxes (NUM_RESULTS,4)
        boxes_expanded = tf.reshape(decoded_boxes, [NUM_RESULTS, 1, 4])  # (N,1,4)
        boxes_tiled = tf.tile(boxes_expanded, [1, NUM_CLASSES, 1])     # (N,C,4)

        # Mask scores above threshold only for classes (skip class 0)
        class_indices = tf.range(NUM_CLASSES, dtype=tf.int32)
        class_indices = class_indices[1:]  # skipping class 0 background

        # Gather per class
        pruned_preds = []
        for c in class_indices.numpy():
            class_scores = scores[:, c]
            score_mask = class_scores > score_threshold
            filtered_scores = tf.boolean_mask(class_scores, score_mask)
            filtered_boxes = tf.boolean_mask(decoded_boxes, score_mask)
            indices = tf.where(score_mask)[:,0]

            # For easier handling, stack info by columns: [score, index, class_id, box4]
            class_ids = tf.fill(tf.shape(filtered_scores), c)

            # Stack results for NMS: boxes, scores only needed for NMS
            if tf.shape(filtered_scores)[0] > 0:
                pruned_preds.append(
                    tf.stack([
                        filtered_scores,
                        tf.cast(indices, tf.float32),
                        tf.cast(class_ids, tf.float32),
                        filtered_boxes[:,0],
                        filtered_boxes[:,1],
                        filtered_boxes[:,2],
                        filtered_boxes[:,3]
                    ], axis=1)
                )
        if len(pruned_preds) == 0:
            # No predictions found
            return tf.constant([], shape=(0,7), dtype=tf.float32)

        pruned_predictions = tf.concat(pruned_preds, axis=0)

        # Now apply NMS per class with iou_threshold=0.5 and max_boxes=10
        final_predictions = []
        iou_threshold = 0.5
        max_boxes = 10

        # We need to apply NMS per class
        unique_classes = tf.unique(pruned_predictions[:,2])[0]
        for c in unique_classes.numpy():
            class_mask = pruned_predictions[:,2] == c
            class_preds = tf.boolean_mask(pruned_predictions, class_mask)

            # Extract boxes and scores for TF NMS
            boxes_tf = class_preds[:,3:7]  # ymin,xmin,ymax,xmax
            scores_tf = class_preds[:,0]

            # tf.image.non_max_suppression expects boxes in [ymin,xmin,ymax,xmax]
            selected_indices = tf.image.non_max_suppression(
                boxes=boxes_tf,
                scores=scores_tf,
                max_output_size=max_boxes,
                iou_threshold=iou_threshold,
                name=None
            )

            selected = tf.gather(class_preds, selected_indices)
            final_predictions.append(selected)

        if len(final_predictions) == 0:
            return tf.constant([], shape=(0,7), dtype=tf.float32)

        final_predictions = tf.concat(final_predictions, axis=0)

        # Sort final predictions by score descending, limit to max_boxes overall
        max_final_boxes = max_boxes
        sorted_indices = tf.argsort(final_predictions[:,0], direction='DESCENDING')
        sorted_final = tf.gather(final_predictions, sorted_indices)[:max_final_boxes]

        # Return tensor with columns: score, index, class_id, ymin, xmin, ymax, xmax
        return sorted_final

    @staticmethod
    def decode_center_size_boxes(locations, ycenter_prior, xcenter_prior, h_prior, w_prior):
        """
        locations: [NUM_RESULTS,4] offsets predicted by the model: (ycenter_off, xcenter_off, h_off, w_off)
        The decode logic converts from center/size param to box coordinates: ymin,xmin,ymax,xmax.
        Using formulas from original code.

        The box_priors vectors are shape [NUM_RESULTS], for ycenter, xcenter, h, w respectively.

        Returns: tensor [NUM_RESULTS,4] => [ymin, xmin, ymax, xmax]
        """

        # Compute ycenter, xcenter, h, w decoded
        ycenter = locations[:,0] / Y_SCALE * h_prior + ycenter_prior
        xcenter = locations[:,1] / X_SCALE * w_prior + xcenter_prior

        h = tf.exp(locations[:,2] / H_SCALE) * h_prior
        w = tf.exp(locations[:,3] / W_SCALE) * w_prior

        ymin = ycenter - h / 2.0
        xmin = xcenter - w / 2.0
        ymax = ycenter + h / 2.0
        xmax = xcenter + w / 2.0

        decoded = tf.stack([ymin, xmin, ymax, xmax], axis=1)
        return decoded


def my_model_function():
    # Instantiate without weights (weights are inside TFLite model in original case)
    return MyModel()


def GetInput():
    # Returns a dummy random input tensor simulating a uint8 RGB image resized to 192x192 with batch 1
    # Shape inferred from original script where input image was resized to (width, height).
    # The original example uses mobilenet_v1_0.5_192.tflite (192x192 input)
    return tf.random.uniform(shape=(1, 192, 192, 3), minval=0, maxval=255, dtype=tf.uint8)

