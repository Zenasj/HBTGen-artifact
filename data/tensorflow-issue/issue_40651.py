def create_base_model(p, units=20):
    # Specs
    input_layer = k.layers.Input((p,))

    penultimate_layer = k.layers.Dense(units, activation='elu')(input_layer)
    penultimate_layer = k.layers.Dense(p)(penultimate_layer)

    training_output = k.layers.Dense(1)(penultimate_layer)

    # Models
    boost_model = k.Model(inputs=input_layer, outputs=penultimate_layer)
    training_model = k.Model(inputs=input_layer, outputs=training_output)

    # Compile and export
    training_model.compile(optimizer='sgd', loss='mse')
    return training_model, boost_model


def create_staged_model(p: int, model: k.Model, units=20):
    # Freeze prior model
    for layer_i in model.layers:
        layer_i.trainable = False

    # Specs
    input_layer = k.layers.Input((p,))

    penultimate_layer = k.layers.concatenate([model.output, input_layer], axis=-1)
    penultimate_layer = k.layers.Dense(units, activation='elu')(penultimate_layer)
    penultimate_layer = k.layers.Dense(p)(penultimate_layer)

    training_output = k.layers.Dense(1)(penultimate_layer)

    # Models
    boost_model = k.Model(inputs=[model.input, input_layer], outputs=penultimate_layer)
    training_model = k.Model(inputs=[model.input, input_layer], outputs=training_output)

    # Compile and export
    training_model.compile(optimizer='sgd', loss='mse')
    return training_model, boost_model


fit_kwargs = dict(
    epochs=50, 
    validation_split=0.1, 
    callbacks=[
        k.callbacks.ReduceLROnPlateau(factor=.5, patience=5, min_lr=1e-6),
        k.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]
)


f0, _ = create_base_model(p)
f0.fit(x, y, **fit_kwargs)  # Works fine

f1, _ = create_staged_model(p, model0)
f1.fit([x, x], y, **fit_kwargs)  # Breaks on fit