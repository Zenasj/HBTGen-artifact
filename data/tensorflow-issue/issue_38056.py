from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow.keras as keras
import numpy as np

predictors_layer = keras.layers.Input(
    shape = (2, ),
    name = "predictors"
)

N_layer = keras.layers.Input(
  shape = (1, ),
  name = "N"
)

logNchooseK_layer = keras.layers.Input(
  shape = (1, ),
  name = "logNchooseK"
)

output_layer = keras.layers.Dense(
  units = 1,
  activation = "linear",
  name = "output"
)(predictors_layer)

model = keras.models.Model(
    inputs = (predictors_layer, N_layer, logNchooseK_layer),
    outputs = output_layer
)

def binomial_loss(y_true, y_pred):
    predicted_prob = keras.backend.exp(y_pred) / (1 + keras.backend.exp(y_pred))

    neg_loglik = -(logNchooseK_layer + y_true * keras.backend.log(predicted_prob) + (N_layer - y_true) * keras.backend.log(1 - predicted_prob))

    return neg_loglik

model.compile(
    loss = binomial_loss,
    optimizer = "adam"#,
    #experimental_run_tf_function = False
)


# train model ----------------------------------------------------------------------------------------------

# the columns are: predictor1, predictor2, N: number of trials, true success probability as function of predictors 1+2, k: realized number of successes, log(N choose k)
sample_data = np.matrix(
    [
        [0.586	,0.78	,88	,0.706	,65	,48.214931],
        [0.709	,1.46	,52	,0.809	,41	,24.824317],
        [-0.109	,-0.644	,1	,0.369	,1	,0],
        [-0.453	,-1.55	,2	,0.199	,0	,0],
        [0.606	,-1.6	,15	,0.29	,4	,7.218910],
        [-1.82	,1.81	,31	,0.609	,15	,19.521092],
        [0.63	,-0.482	,83	,0.488	,44	,54.944102],
        [-0.276	,0.62	,51	,0.581	,29	,32.681372],
        [-0.284	,0.612	,81	,0.579	,45	,53.223945],
        [-0.919	,-0.162	,7	,0.359	,2	,3.044522],
        [-0.116	,0.812	,93	,0.634	,61	,57.420880],
        [1.82	,2.2	,81	,0.928	,75	,19.597920],
        [0.371	,2.05	,8	,0.848	,7	,2.079442],
        [0.52	,1.63	,61	,0.815	,54	,19.893774],
        [-0.751	,0.254	,72	,0.454	,34	,47.429363],
        [0.817	,0.491	,52	,0.685	,33	,31.966485],
        [-0.886	,-0.324	,73	,0.335	,17	,37.410978],
        [-0.332	,-1.66	,75	,0.196	,16	,36.684713],
        [1.12	,1.77	,10	,0.868	,8	,3.806662],
        [0.299	,0.0258	,40	,0.542	,20	,25.649407]
    ]
)

model.fit(
  x = {
    'predictors': sample_data[:,(0,1)],
    'N': sample_data[:,2],
    'logNchooseK': sample_data[:,5]
  },
  y = {
    'output': sample_data[:,4]
  },
  validation_split = 0.2,
  epochs = 1000
)

# evaluate model

predicted_logodds = model.predict(
    x = {
        'predictors': sample_data[:,(0,1)],
        'N': sample_data[:,2],
        'logNchooseK': sample_data[:,5]
    }
)

predicted_probs = np.exp(predicted_logodds) / (1 + np.exp(predicted_logodds))

print(np.transpose(np.reshape([sample_data[:, 3], predicted_probs], [2, 20])))