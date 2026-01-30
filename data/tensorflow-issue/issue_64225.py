import tensorflow as tf

class DeterminantZeroLoss(Loss):
    def __init__(self, name="determinant_zero_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Calculate the determinant of the predicted matrix
        det = tf.linalg.det(y_pred)

        # Penalize when the determinant is close to zero
        loss = tf.abs(det)

        return loss