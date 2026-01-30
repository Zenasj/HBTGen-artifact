import math

# Build the approximation model
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
import tensorflow.compat.v1 as tf
from collections import namedtuple

sys.path.append(os.path.join('..','cnnModel'))
sys.path.append('..')
from cnnModel import cnnModel,rigidDeformer

def buildModel(config,pose,addNeutral=True):

    with open(config['data_params']['cache_file'],'rb') as file:
        data = pickle.load(file)
    cache = data
    parts = data['vCharts']
    faces = data['faces']
    neutral = data['neutral'][data['active']].astype('float32')
    uvs = data['uv']
    if 'parameter_mask' in data:
        mask = data['parameter_mask']
    else:
        mask = None

    # Create the model
    partCount = np.max(parts)+1
    data = {'pose':pose}
    usedVerts = []
    usedUVs = []
    for i in range(partCount):
        if np.sum(parts==i) > 0:
            data['image-'+str(i)] = tf.ones(1)
        else:
            data['image-'+str(i)] = None
        ref = faces.reshape(-1)
        idx = np.arange(len(neutral))[parts==i]
        if len(idx) == 0:
            continue
        usedFaces = [True if v in idx else False for v in ref]
        usedFaces = np.sum(np.asarray(usedFaces).reshape((-1,3)),-1) == 3
        faceIdx = np.arange(len(faces))[usedFaces]
        uv = uvs[idx]
        usedUVs.append(uv)
        usedVerts.append(idx)
    idx = np.concatenate(usedVerts)
    linear = np.zeros(neutral.shape,dtype='float32')
    if addNeutral:
        linear[idx] = neutral[idx]
    else:
        neutral = linear
    data['linear'] = linear
    dataset = namedtuple('Dataset','mask usedUVs usedVerts')(mask,usedUVs,usedVerts)
    model = cnnModel.buildModel(data,dataset,neutral,config)
    model['parts'] = parts
    model['cache'] = cache
    return model

class Approximator():

    def __init__(self,  sess):
        ####
        rigControls = tf.Variable(tf.zeros([1, 172], tf.float32))

        approximation_config = 'experiments/v00_refine_model_leaky.yaml'
        with open(approximation_config) as file:
            approximationConfig = yaml.load(file)
        with tf.variable_scope('refine'):
            refineMesh = buildModel(approximationConfig, rigControls, addNeutral=False)

        base_config = 'experiments/v00_base_model_leaky.yaml'
        with open(base_config) as file:
            baseConfig = yaml.load(file)
        mesh = buildModel(baseConfig, rigControls)

        have_refineMesh = True
        refineMesh['output'] = mesh['output'] + refineMesh['output']

        print(refineMesh['output'])
        print(type(refineMesh['output']))

        # Apply ridig deformer
        # Load info about the mesh
        with open(os.path.join(baseConfig['data_params']['cache_file']), 'rb') as file:
            data = pickle.load(file)
        parts = data['vCharts']
        neutral = data['neutral'][data['active']]
        print("data['neutral'].shape=", data['neutral'].shape)
        print('neutral.shape=', neutral.shape)
        faces = data['faces']
        mask = np.arange(len(parts))[parts > -1]

        if 'rigid_files' in baseConfig['data_params']:

            cur_rigidDeformer = rigidDeformer.RigidDeformer(neutral, [f for f in
                                                                      baseConfig['data_params']['rigid_files']], mask)
            final_base_mesh = cur_rigidDeformer.deformTF(mesh['output'][0])[np.newaxis]
            if have_refineMesh:
                final_refineMesh = cur_rigidDeformer.deformTF(refineMesh['output'][0])[np.newaxis]
        else:
            final_base_mesh = mesh['output']
            if have_refineMesh:
                final_refineMesh = refineMesh['output']

        # print('final_refineMesh.shape =', final_refineMesh.eval())
        print('type(final_refineMesh) =', type(final_refineMesh))

        ####
        vars = tf.trainable_variables()
        print(vars)
        vars_to_train = vars[0]
        vars_to_load = vars[1:]
        self.saver = tf.train.Saver(vars_to_load)

        checkpoint_dir = 'E:\\PycharmProjects\\FDFD-Metahuman-Y-up\\output\\v00_refine_model_leaky'
        self.checkpointFile = tf.train.latest_checkpoint(checkpoint_dir)
        self.saver.restore(sess, self.checkpointFile)

        self.rigControls=rigControls
        self.sess=sess
        self.final_refineMesh=final_refineMesh


    def fk(self,control):
        # vars = tf.trainable_variables()
        # print("after_vars =",vars)
        self.rigControls= control
        # print('self.rigControls=',self.rigControls)
        #
        # # init = tf.global_variables_initializer()
        # # self.sess.run(init)
        # vars = tf.trainable_variables()
        # print(vars)
        # vars_to_train = vars[0]
        # vars_to_load = vars[1:]
        # saver = tf.train.Saver(vars_to_load)
        #
        # checkpoint_dir = 'E:\\PycharmProjects\\FDFD-Metahuman-Y-up\\output\\v00_refine_model_leaky'
        # checkpointFile = tf.train.latest_checkpoint(checkpoint_dir)
        # saver.restore(sess, checkpointFile)

        final_refineMesh=self.sess.run([self.final_refineMesh])
        return final_refineMesh

    def ik(self,verts):
        pass








if __name__ == '__main__':
    # Load the model from file
    with tf.Session() as sess:
        approx=Approximator(sess)


        conbtrol0=tf.zeros([1, 172], tf.float32)
        verts0=approx.fk(conbtrol0)
        print('verts0=',verts0)

        conbtrol1 = tf.ones([1, 172], tf.float32)
        verts1 = approx.fk(conbtrol1)
        print('verts1=', verts1)

None

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow as tf
conbtrol0=tf.zeros([1, 172], tf.float32)
verts0= tf.math.multiply(conbtrol0, conbtrol0)
print('verts0=',verts0)

conbtrol1 = tf.ones([1, 172], tf.float32)
verts1 = tf.math.multiply(conbtrol1, conbtrol1)
print('verts1=', verts1)