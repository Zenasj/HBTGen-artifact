from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
print(tf.__version__)
from tensorflow import math as m
from tensorflow.keras import layers

class Kinematics_Physics(layers.Layer):

    def __init__(self,):
        super(Kinematics_Physics, self).__init__()
        # define initial variables
        self.TF_PI = tf.constant(3.1415926535897932, dtype =tf.float32)
        self.TF_180 = tf.constant(180.0, dtype =tf.float32)
        self.TINY_VALUE = tf.constant(1e-6, dtype =tf.float32)
        self.out_Pose=tf.Variable([0,0,0,0,0,0],  dtype =tf.float32, trainable = False)
        self.A_Deg = tf.Variable(0,dtype = tf.float32, trainable = False)
        self.B_Deg = tf.Variable(0,dtype = tf.float32, trainable = False)
        self.C_Deg = tf.Variable(0,dtype = tf.float32, trainable = False)
        self.sA = tf.Variable(0,dtype = tf.float32, trainable = False)
        self.cA = tf.Variable(0,dtype = tf.float32, trainable = False)
        self.b = tf.Variable(tf.zeros((4,4),dtype = tf.float32),dtype = tf.float32, trainable = False)
        
        # initial dh parameters: 6 x 4 parameters for a robotic arm with 6 joints.
        dh = tf.constant([  [0 ,    180.,   -650.,  0.],
                            [270.,   90.,      0.,  0.],
                            [800.,    0.,      0.,  0.],
                            [140.,   90.,   -908.,  0.],
                            [0.,    -96.,      0.,  0.],
                            [0.,    -65.,   260.,   0.]] ,dtype = tf.float32)
        
        # convert to trainable weights?? for further optimization 
        self.dh = tf.Variable(initial_value=dh, trainable=True)

        # initialize a buffer to calculate the the modified angles
        self.dh_processed = tf.Variable(tf.zeros(self.dh.shape,dtype = tf.float32),dtype = tf.float32, trainable = False)
        # buffer for transformation matrix
        self.trans_matrix = tf.Variable(tf.zeros((4,4),dtype = tf.float32),  dtype =tf.float32, trainable = False)


    @tf.function
    def radians(self,a):
        return m.multiply(a,(tf.divide(self.TF_PI, self.TF_180 )))
    
    @tf.function
    def degrees(self,a):
        return m.multiply(a,(tf.divide(self.TF_180 , self.TF_PI)))
    
    @tf.function
    def joint_transform(self,input_values):
        '''input: format --> 1D array = [a, α, d, θ]'''
    
        a = input_values[0] # a
        alpha = self.radians(input_values[1]) #convert degrees to radians
        d = input_values[2] # d
        theta = self.radians(input_values[3]) #convert degrees to radians θ
        
        self.trans_matrix.assign(
                                [[      m.cos(theta),             -m.sin(theta),              0,                a],
                                [m.multiply(m.sin(theta),m.cos(alpha)),  m.multiply(m.cos(theta),m.cos(alpha)), -m.sin(alpha), m.multiply(-m.sin(alpha),d)],
                                [m.multiply(m.sin(theta),m.sin(alpha)),  m.multiply(m.cos(theta),m.sin(alpha)),  m.cos(alpha),  m.multiply(m.cos(alpha),d)],
                                [        0,                             0,                     0,            1        ]])
        return self.trans_matrix
    
    @tf.function
    def to_pose(self,T):
        # converts the transformation matrix to pose [x, y, z, alpha ]
        if(m.abs(T[1,2]) <= self.TINY_VALUE and m.abs(T[2,2]) <= self.TINY_VALUE): #% singular : B = +-90 (Gimbal Lock)
            self.C_Deg.assign(tf.constant(0, dtype =tf.float32))
            self.B_Deg.assign(self.degrees(m.atan2( T[0,2] , m.divide_no_nan(T[2,2], m.cos(self.C_Deg)) ))) #convert radians to degree
            self.A_Deg.assign(self.degrees(m.atan2( T[1,0] ,  m.divide_no_nan(T[1,1], m.cos(self.C_Deg)) ))) #convert radians to degree
        
        else:
            self.A_Deg.assign(self.degrees(m.atan2(-T[0,1] , T[0,0]))) #convert radians to degree
            self.sA.assign(m.sin(self.radians(self.A_Deg)))
            self.cA.assign(m.cos(self.radians(self.A_Deg)))
            self.B_Deg.assign(self.degrees(m.atan2(T[0,2] , m.multiply(self.cA,T[0,0]) - m.multiply(self.sA,T[0,1]))))#convert radians to degree
            self.C_Deg.assign(self.degrees(m.atan2(-T[1,2] , T[2,2]))) #convert radians to degree
        
        self.out_Pose.assign([ T[0,3], T[1,3], T[2,3],self.A_Deg, self.B_Deg , self.C_Deg])
        return self.out_Pose
    
    
    @tf.function
    def forward_kinematics(self, theta):
        
        # add the dh_theta to measured theta to get the final theta
        actual_theta = m.add(self.dh[:,3] ,theta)
        
        # create new dh_table with modified theta for forward kinematics calculation
        self.dh_processed[:,0].assign(self.dh[:,0])
        self.dh_processed[:,1].assign(self.dh[:,1])
        self.dh_processed[:,2].assign(self.dh[:,2])
        self.dh_processed[:,3].assign(actual_theta)
        b = tf.eye(4, dtype = tf.float32)
        for i in tf.range(6):
            b = (tf.linalg.matmul(b,self.joint_transform(self.dh_processed[i])))
        return self.to_pose(b)
      
    def __call__(self, inputs):
        return tf.map_fn(self.forward_kinematics, inputs, parallel_iterations=True, dtype=tf.float32)
    
    def get_config(self):
        config = super(Kinematics_Physics, self).get_config()
        return config
    

if __name__ == "__main__":
    # create a list of six joint angles
    ang = tf.Variable([[20,10,20,10,10,20],[15,10,20,22,20,32]], dtype = tf.float32, trainable=False)
    # Call the Kinematics_physics layer
    fk = Kinematics_Physics()
    print("trainable weights: ", len(fk.trainable_weights))
    print("non-trainable weights: ", len(fk.non_trainable_weights))
    print("trainable_weights: ", fk.trainable_weights)
    print("forward kinematic transform: ", fk(ang[:10]))
    # Everything works up to this.

    # # create a model using functional API and custom layer wrapped in Lambda layer
    in_= layers.Input(shape=(6,))
    for_kin = layers.Lambda(Kinematics_Physics())(in_)
    model = tf.keras.models.Model(inputs=in_, outputs=for_kin) 
    model.summary() 
    # Works but the weights are not trainable in this case
    
    
    # create a model using functional API
    in_= layers.Input(shape=(6,))
    for_kin = Kinematics_Physics()(in_)
    model = tf.keras.models.Model(inputs=in_, outputs=for_kin) 
    model.summary() 
    # Error: expected behavior => should create a model with 24 trainable parameters
    #        but the error happens in tf.map_fn of Kinematics_Physics().__call__ function