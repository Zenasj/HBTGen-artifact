import tensorflow as tf
from tensorflow import keras

class MultiLabelMacroSpecificity(tf.keras.metrics.Metric):
    
    def __init__(self, name='multi_label_macro_specificity', threshold=0.5, **kwargs):        
        super(MultiLabelMacroSpecificity, self).__init__(name=name, **kwargs)
        self.specificity = self.add_weight(name='mlm_spec', initializer='zeros')        
        self.threshold       = tf.constant(threshold)

        # replace this with tf confusion_matrix utils
        self.true_negatives  = self.add_weight(name='tn', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
    
    def update_state(self, y_true, y_pred):
        
        # Compare predictions and threshold.        
        pred_is_pos  = tf.greater(tf.cast(y_pred, tf.float32), self.threshold)            
        pred_is_neg  = tf.logical_not(tf.cast(pred_is_pos, tf.bool))
        # |-- in case of soft labeling        
        label_is_pos = tf.greater(tf.cast(y_true, tf.float32), self.threshold)                
        label_is_neg = tf.logical_not(tf.cast(label_is_pos, tf.bool))
        
        self.true_negatives.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(pred_is_neg, label_is_neg), tf.float32)))
        self.false_positives.assign_add(
            tf.reduce_sum(tf.cast(tf.logical_and(pred_is_pos, label_is_neg), tf.float32))
        )
        
        tn = self.true_negatives
        fp = self.false_positives
        specificity = tf.div_no_nan(tn, tf.add(tn, fp))
        self.specificity.assign(specificity)
        return specificity
    
    def result(self):
        return self.specificity