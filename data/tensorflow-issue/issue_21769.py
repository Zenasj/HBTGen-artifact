import numpy as np
import random
from tensorflow.keras import layers
from tensorflow.keras import models

EMBEDDED = True
if EMBEDDED:
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.models import Model
else:
    from keras.layers import Input, Dense
    from keras.models import Model

inp = Input((32,))
x = Dense(1)(inp)
model = Model(inp, x)
model.name = 'mymodel'

def crossover_rnn(model_1, model_2):
    """
    Executes crossover for the RNN in the GA for 2 models, modifying the first model
    :param model_1:
    :param model_2:
    :return:
    """
    # new_model = copy.copy(model_1)
    new_model = model_1

    # Probabilty of models depending on their RMSE test score
    # Lower RMSE score has higher prob
    test_score_total = model_1.test_score + model_2.test_score
    model_1_prob = 1 - (model_1.test_score / test_score_total)
    model_2_prob = 1 - model_1_prob
    # Probabilities of each item for each model (all items have same probabilities)
    model_1_prob_item = model_1_prob / (len(model_1.layers) - 2)
    model_2_prob_item = model_2_prob / (len(model_2.layers) - 2)

    # Number of layers of new generation depend on probability of each model
    num_layers_new_gen = int(model_1_prob * (len(model_1.layers) - 1) + model_2_prob * (len(model_2.layers) - 1))

    # Create list of int with positions of the layers of both models.
    cross_layers_pos = []
    # Create list of weights
    attention_weights = []
    # Add positions of layers for model 1. Input and ouput layer are not added.
    for i in range(2, len(model_1.layers)):
        mod_item = type('', (), {})()
        mod_item.pos = i
        mod_item.model = 1
        cross_layers_pos.append(mod_item)
        attention_weights.append(model_1_prob_item)

    # Add positions of layers for model 2. Input and ouput layer are not added.
    for i in range(2, len(model_2.layers)):
        mod_item = type('', (), {})()
        mod_item.pos = i
        mod_item.model = 2
        cross_layers_pos.append(mod_item)
        attention_weights.append(model_2_prob_item)

    collect_gc()

    # If new num of layers are larger than the num crossover layers, keep num of crossover layers
    if num_layers_new_gen > len(cross_layers_pos):
        num_layers_new_gen = len(cross_layers_pos)

    # Randomly choose num_layers_new_gen layers of the new list
    cross_layers_pos = list(np.random.choice(cross_layers_pos, size=num_layers_new_gen, replace=False,
                                             p=attention_weights))

    # Add both group of hidden layers to new group of layers using previously chosen layer positions of models
    cross_layers = []
    for i in range(len(cross_layers_pos)):
        mod_item = cross_layers_pos[i]
        if mod_item.model == 1:
            cross_layers.append(model_1.layers[mod_item.pos])
        else:
            cross_layers.append(model_2.layers[mod_item.pos])

    collect_gc()

    # Add input layer randomly from parent 1 or parent 2
    bit_random = random.randint(0, 1)
    if bit_random == 0:
        cross_layers.insert(0, model_1.layers[0])
    else:
        cross_layers.insert(0, model_2.layers[0])

    bit_random = random.randint(0, 1)
    if bit_random == 0:
        cross_layers.append(model_1.layers[len(model_1.layers) - 1])
    else:
        cross_layers.append(model_2.layers[len(model_2.layers) - 1])

    # Set new layers
    new_model.layers = cross_layers

    return new_model