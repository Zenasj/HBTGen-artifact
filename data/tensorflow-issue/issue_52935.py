# tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, max_output_size=5, score_threshold=0.75, iou_threshold=0.5):
        super(MyModel, self).__init__()
        # Load the BlazeFace model as a keras.Model
        # For this example, we simulate it by a placeholder model. 
        # In real use, replace with: tf.keras.models.load_model(path_to_blazeface)
        # Here we will build a dummy convolution model to simulate output shape:
        self.max_output_size = max_output_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        
        # The actual BlazeFace model should return a 3D tensor with shape (batch_size, N, C)
        # where N is number of predicted boxes, C the features per box.
        # Here, for demonstration, we create a dummy layer to simulate:
        # Normally you do: self.classifier = tf.keras.models.load_model(blazeface_path)
        self.classifier = keras.Sequential([
            layers.Conv2D(16, kernel_size=3, padding="same", activation="relu"),
            layers.Conv2D(17, kernel_size=1, padding="same")  # Output 17 channels per spatial location
        ])
    
    def call(self, inputs):
        # Normalize inputs from [0,255] uint8/floats to (-1,1) float32
        input_tensor = tf.cast(inputs, tf.float32)
        input_tensor = (input_tensor / 127.5) - 1.0
        
        # Run the base detector to get raw predictions of shape (B, H, W, C)
        temp_tensor = self.classifier(input_tensor)  # shape: (1, H, W, 17) for example
        
        # Reshape to (num_boxes, features)
        # Here, infer the number of boxes and features:
        shape = tf.shape(temp_tensor)
        batch_size = shape[0]
        H = shape[1]
        W = shape[2]
        C = shape[3]  # should be 17
        
        # Flatten spatial dims to get all boxes
        flatten = tf.reshape(temp_tensor, (batch_size, -1, C))  # (B, N, 17)
        
        # The last dimension feature map is 17, interpreted as:
        # final_boxes coordinates: 4 (center_yx and wh)
        # rest maybe landmarks, score, etc.
        
        # For this example, assume:
        # Box: first 4 features => center_yx (2), wh (2)
        # Scores: last feature (index 16)
        
        # Processing on batch dimension 1 for simplicity (batch=1)
        # We will write code assuming batch size = 1 for clarity
        
        temp = flatten[0]  # (N, 17)
        
        # Extract box params
        final_boxes = temp[:, 0:4]  # center_yx + wh
        
        # Split center and wh
        center_yx = final_boxes[:, 0:2]
        wh = final_boxes[:, 2:4]
        half_wh = wh / 2
        
        # Convert center_yx + wh to top-left bottom-right format (tlbr)
        tl = center_yx - half_wh
        tl = tf.clip_by_value(tl, 0.0, 1e8)  # large clamp as original code
        
        br = center_yx + half_wh
        
        # Switch from (y,x) to (x,y) as expected by tf.image.non_max_suppression
        # tf.image.non_max_suppression expects boxes in form [y1, x1, y2, x2], but here box_tlbr is constructed:
        # Here user swaps y,x for coordinates, so we replicate that logic
        
        # Stack for box tlbr = [x1,y1,x2,y2]
        xy1 = tf.stack([tl[:, 1], tl[:, 0]], axis=-1)
        xy2 = tf.stack([br[:, 1], br[:, 0]], axis=-1)
        box_tlbr = tf.concat([xy1, xy2], axis=1)  # shape (N,4)
        
        # Scores are last channel (index 16), reshape to vector
        raw_scores = temp[:, -1]
        scores = tf.reshape(raw_scores, [-1])
        
        # Perform non-max suppression
        selected_indices = tf.image.non_max_suppression(
            boxes=box_tlbr,
            scores=scores,
            max_output_size=self.max_output_size,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold
        )
        
        # Gather selected boxes and scores
        selected_rows = tf.gather(temp, selected_indices, axis=0)  # shape (M,17)
        
        # Extract box details without last score channel
        boxes = selected_rows[:, :-1]
        
        # If single box, keep dims consistent
        if len(boxes.shape) == 1:
            boxes = tf.expand_dims(boxes, axis=0)
        
        # Scale boxes by image size 128 (from the original code)
        orig_points = boxes * 128.0
        
        # Gather corresponding scores
        selected_scores = tf.gather(raw_scores, selected_indices, axis=0)
        selected_scores = tf.expand_dims(selected_scores, axis=1)
        
        # Concatenate boxes and scores to form final output (M, 16 + 1)
        final_result = tf.concat([orig_points, selected_scores], axis=1)
        
        # Add batch dimension for output consistency: (1, M, 17)
        final_result = tf.expand_dims(final_result, axis=0)
        
        return final_result


def my_model_function():
    # Returns an instance of MyModel initialized with default parameters matching the original example.
    return MyModel()


def GetInput():
    # Returns a random input tensor with shape (1,128,128,3) dtype float32 to feed MyModel.
    # The original model expects 128x128 RGB image batch of 1.
    return tf.random.uniform(shape=(1, 128, 128, 3), minval=0, maxval=255, dtype=tf.float32)

