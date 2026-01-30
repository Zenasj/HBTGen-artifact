import tensorflow as tf

with tf.train.MonitoredTrainingSession(...) as sess:
    ...
    saver.save(sess, 'model.ckpt')

saver.save(sess._sess._sess._sess._sess, 'model.ckpt')

def get_session(sess):
    session = sess
    while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
    return session

saver.save(get_session(sess), 'model.ckpt')

class SaveAtEnd(tf.train.SessionRunHook):
    '''a training hook for saving the final variables'''

    def __init__(self, filename, variables):
        '''hook constructor

        Args:
            filename: where the model will be saved
            variables: the variables that will be saved'''

        self.filename = filename
        self.variables = variables

    def begin(self):
        '''this will be run at session creation'''

        #pylint: disable=W0201
        self._saver = tf.train.Saver(self.variables, sharded=True)

    def end(self, session):
        '''this will be run at session closing'''

        self._saver.save(session, self.filename)