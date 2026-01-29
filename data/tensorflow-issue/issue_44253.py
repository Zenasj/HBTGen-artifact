# tf.random.uniform((B, 6), dtype=tf.float32) ← Input shape inferred as a batch of 6 joint angles per instance

import tensorflow as tf
from tensorflow import math as m
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Constants
        self.TF_PI = tf.constant(3.1415926535897932, dtype=tf.float32)
        self.TF_180 = tf.constant(180.0, dtype=tf.float32)
        self.TINY_VALUE = tf.constant(1e-6, dtype=tf.float32)

        # Initial DH parameters for a 6 DOF robotic arm (6x4 matrix):
        # Columns likely correspond to [a, alpha_deg, d, theta_offset]
        dh_init = tf.constant([
            [0.,    180.,  -650.,    0.],
            [270.,   90.,     0.,    0.],
            [800.,    0.,     0.,    0.],
            [140.,   90.,  -908.,    0.],
            [0.,    -96.,     0.,    0.],
            [0.,    -65.,   260.,    0.]
        ], dtype=tf.float32)

        # Set DH params as trainable variables to be optimized during training
        self.dh = tf.Variable(dh_init, trainable=True, dtype=tf.float32)

        # Buffers and intermediate vars (non-trainable)
        self.dh_processed = tf.Variable(tf.zeros_like(self.dh), trainable=False)
        self.trans_matrix = tf.Variable(tf.eye(4), trainable=False)
        
        # Buffers for pose computations
        self.out_Pose = tf.Variable(tf.zeros((6,), dtype=tf.float32), trainable=False)
        self.A_Deg = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.B_Deg = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.C_Deg = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.sA = tf.Variable(0., dtype=tf.float32, trainable=False)
        self.cA = tf.Variable(0., dtype=tf.float32, trainable=False)

    def radians(self, degrees):
        # Converts degrees to radians
        return degrees * (self.TF_PI / self.TF_180)

    def degrees(self, radians):
        # Converts radians to degrees
        return radians * (self.TF_180 / self.TF_PI)

    def joint_transform(self, input_values):
        """
        Compute individual joint transformation matrix from DH parameters.

        input_values: tensor of shape (4,) - [a, alpha_deg, d, theta_deg]
        returns: 4x4 transformation matrix as tensor
        """
        a = input_values[0]
        alpha = self.radians(input_values[1])
        d = input_values[2]
        theta = self.radians(input_values[3])

        cos_theta = m.cos(theta)
        sin_theta = m.sin(theta)
        cos_alpha = m.cos(alpha)
        sin_alpha = m.sin(alpha)

        trans_mat = tf.stack([
            [cos_theta,           -sin_theta,         0,              a],
            [sin_theta * cos_alpha, cos_theta * cos_alpha, -sin_alpha, -sin_alpha * d],
            [sin_theta * sin_alpha, cos_theta * sin_alpha,  cos_alpha,  cos_alpha * d],
            [0,                     0,                    0,          1]
        ], axis=0)

        self.trans_matrix.assign(trans_mat)
        return self.trans_matrix

    def to_pose(self, T):
        """
        Converts transformation matrix T (4x4) to pose vector [x, y, z, A, B, C]
        where A, B, C are Euler angles in degrees derived from rotation.

        Handles singularity (Gimbal Lock) cases as per original logic.
        """
        # Access elements with safe indexing
        T_ = T  # alias for clarity

        # Check for singularity
        cond = (m.abs(T_[1, 2]) <= self.TINY_VALUE) & (m.abs(T_[2, 2]) <= self.TINY_VALUE)

        def singular_case():
            # B = ±90 deg (Gimbal lock)
            self.C_Deg.assign(tf.constant(0., dtype=tf.float32))
            # atan2(y, x) with some denominator handling
            self.B_Deg.assign(self.degrees(
                m.atan2(T_[0, 2], m.divide_no_nan(T_[2, 2], m.cos(self.C_Deg)))
            ))
            self.A_Deg.assign(self.degrees(
                m.atan2(T_[1, 0], m.divide_no_nan(T_[1, 1], m.cos(self.C_Deg)))
            ))
        
        def normal_case():
            self.A_Deg.assign(self.degrees(m.atan2(-T_[0, 1], T_[0, 0])))
            self.sA.assign(m.sin(self.radians(self.A_Deg)))
            self.cA.assign(m.cos(self.radians(self.A_Deg)))
            self.B_Deg.assign(self.degrees(
                m.atan2(T_[0, 2], self.cA * T_[0, 0] - self.sA * T_[0, 1])
            ))
            self.C_Deg.assign(self.degrees(m.atan2(-T_[1, 2], T_[2, 2])))
        
        tf.cond(cond, singular_case, normal_case)

        # Translation part + computed Euler angles
        pose = tf.stack([
            T_[0, 3],
            T_[1, 3],
            T_[2, 3],
            self.A_Deg,
            self.B_Deg,
            self.C_Deg
        ])

        self.out_Pose.assign(pose)
        return self.out_Pose

    def forward_kinematics(self, theta):
        """
        Compute forward kinematics pose for a single set of joint angles.

        theta: tensor shape (6,) joint angle inputs (degrees)
        returns: tensor pose (6,) [x, y, z, A, B, C]
        """
        # Modified the theta offsets by the DH parameters
        actual_theta = self.dh[:, 3] + theta

        # Prepare the processed DH parameters with modified thetas
        self.dh_processed[:, 0].assign(self.dh[:, 0])  # a
        self.dh_processed[:, 1].assign(self.dh[:, 1])  # alpha_deg
        self.dh_processed[:, 2].assign(self.dh[:, 2])  # d
        self.dh_processed[:, 3].assign(actual_theta)   # theta_deg (modified)

        # Initialize transformation as identity
        b = tf.eye(4, dtype=tf.float32)
        # Compose transformations for each of the 6 joints
        for i in tf.range(6):
            joint_tf = self.joint_transform(self.dh_processed[i])
            b = tf.linalg.matmul(b, joint_tf)

        # Convert final transformation matrix to pose vector
        return self.to_pose(b)

    @tf.function
    def call(self, inputs):
        """
        Called during the forward pass.

        inputs: shape (batch_size, 6)
        returns: shape (batch_size, 6) with forward kinematic poses per input
        """
        # Use tf.map_fn to apply forward_kinematics over batch items
        # parallel_iterations=10 default is fine, dtype is float32 pose (6,)
        return tf.map_fn(self.forward_kinematics, elems=inputs, dtype=tf.float32)


def my_model_function():
    # Returns an instance of the MyModel Keras model with trainable DH parameters
    return MyModel()


def GetInput():
    # Generate a random batch of inputs:
    # Batch size B = 4 (arbitrary), 6 joint angles each in degrees [-180, 180]
    B = 4
    input_tensor = tf.random.uniform((B, 6), minval=-180, maxval=180, dtype=tf.float32)
    return input_tensor

