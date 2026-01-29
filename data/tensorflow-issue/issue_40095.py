# tf.random.uniform((None, len(feature_columns)), dtype=tf.float32) ← input shape is dynamic batch and feature dict inputs, exact shape can't be fixed due to feature_columns use

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstruct the feature_columns as in the issue, embedding categorical + bucketized numerical
        # Since we can't rely on the original dataset here,
        # we embed a minimal mockup of feature columns for demonstration.
        # In reality, you would pass actual feature_columns in init or construct from config.
        
        # For demonstration, imagine input features include:
        countries = ['afghanistan', 'aland islands', 'albania', 'algeria', 'american samoa', 
                     'andorra', 'angola', 'anguilla', 'antarctica', 'antigua and barbuda', 
                     'argentina', 'armenia', 'aruba', 'australia', 'austria', 'azerbaijan', 
                     'bahamas (the)', 'bahrain', 'bangladesh', 'barbados', 'belarus', 
                     'belgium', 'belize', 'benin', 'bermuda']
        
        # Setup categorical feature column with embeddings
        country_categorical = tf.feature_column.categorical_column_with_vocabulary_list(
            'country', countries)
        country_embedding = tf.feature_column.embedding_column(country_categorical, dimension=8)
        
        # Setup some numeric columns for demonstration
        # We'll define one numeric column, bucketized like boundaries=[25,50,75,90,95,99]
        age_numeric = tf.feature_column.numeric_column('age')
        age_bucketized = tf.feature_column.bucketized_column(age_numeric, boundaries=[25,50,75,90,95,99])
        
        # Collect feature columns to mimic the original model
        self.feature_columns = [country_embedding, age_bucketized]
        
        # DenseFeatures layer consumes feature columns
        self.feature_layer = tf.keras.layers.DenseFeatures(self.feature_columns)
        
        # Sequential dense layers following the original model
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        # inputs expected as dict with keys: 'country' (string), 'age' (float or int)
        x = self.feature_layer(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return the MyModel instance
    return MyModel()

def GetInput():
    # We must return a dict of tensors consistent with feature_columns:
    # 'country' as a string tensor, and 'age' as float tensor
    # Assume batch_size = 4 for example
    batch_size = 4
    
    countries = ['afghanistan', 'aland islands', 'albania', 'algeria', 'american samoa', 
                 'andorra', 'angola', 'anguilla', 'antarctica', 'antigua and barbuda', 
                 'argentina', 'armenia', 'aruba', 'australia', 'austria', 'azerbaijan', 
                 'bahamas (the)', 'bahrain', 'bangladesh', 'barbados', 'belarus', 
                 'belgium', 'belize', 'benin', 'bermuda']
    
    import numpy as np
    # Randomly pick batch_size countries (as strings)
    sampled_countries = np.random.choice(countries, batch_size)
    
    # Create age inputs as float values in [20, 100] range to cover bucket boundaries
    ages = np.random.uniform(20, 100, size=(batch_size,))
    
    # Convert numpy arrays to tensors with appropriate dtypes
    country_tensor = tf.convert_to_tensor(sampled_countries, dtype=tf.string)
    age_tensor = tf.convert_to_tensor(ages, dtype=tf.float32)
    
    # Return as dict matching expected input signature for feature_columns
    return {'country': country_tensor, 'age': age_tensor}

