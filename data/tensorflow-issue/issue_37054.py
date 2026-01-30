import tensorflow as tf

def build_model(params_path = 'test/params', enc_lstm_units = 128, unroll = True, use_gru=False, optimizer='adam', display_summary=True):
    """
    Build keras model

    Parameters:

    params_path (str): Path for saving/loading the params.

    enc_lstm_units (int): Positive integer, dimensionality of the output space.

    unroll (bool): Boolean (default False). If True, the network will be unrolled, else a symbolic loop will be used. Unrolling can speed-up a RNN, although it tends to be more memory-intensive. Unrolling is only suitable for short sequences.

    use_gru (bool): GRU will be used instead of LSTM

    optimizer (str): optimizer to be used

    display_summary (bool): Set to true for verbose information.


    Returns:

    model (keras model): built model object.
    
    params (dict): Generated params (encoding, decoding dicts ..).

    """
    # generateing the encoding, decoding dicts
    params = build_params(params_path = params_path)

    input_encoding = params['input_encoding']
    input_decoding = params['input_decoding']
    input_dict_size = params['input_dict_size']
    output_encoding = params['output_encoding']
    output_decoding = params['output_decoding']
    output_dict_size = params['output_dict_size']
    max_input_length = params['max_input_length']
    max_output_length = params['max_output_length']


    if display_summary:
        print('Input encoding', input_encoding)
        print('Input decoding', input_decoding)
        print('Output encoding', output_encoding)
        print('Output decoding', output_decoding)


    # We need to define the max input lengths and max output lengths before training the model.
    # We pad the inputs and outputs to these max lengths
    encoder_input = Input(shape=(max_input_length,))
    decoder_input = Input(shape=(max_output_length,))

    # Need to make the number of hidden units configurable
    encoder = Embedding(input_dict_size, enc_lstm_units, input_length=max_input_length, mask_zero=True)(encoder_input)
    # using concat merge mode since in my experiments it g ave the best results same with unroll
    if not use_gru:
        encoder = Bidirectional(LSTM(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)
        encoder_outs, forward_h, forward_c, backward_h, backward_c = encoder
        encoder_h = concatenate([forward_h, backward_h])
        encoder_c = concatenate([forward_c, backward_c])
    
    else:
        encoder = Bidirectional(GRU(enc_lstm_units, return_sequences=True, return_state=True, unroll=unroll), merge_mode='concat')(encoder)        
        encoder_outs, forward_h, backward_h= encoder
        encoder_h = concatenate([forward_h, backward_h])
    

    # using 2* enc_lstm_units because we are using concat merge mode
    # cannot use bidirectionals lstm for decoding (obviously!)
    
    decoder = Embedding(output_dict_size, 2 * enc_lstm_units, input_length=max_output_length, mask_zero=True)(decoder_input)

    if not use_gru:
        decoder = LSTM(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=[encoder_h, encoder_c])
    else:
        decoder = GRU(2 * enc_lstm_units, return_sequences=True, unroll=unroll)(decoder, initial_state=encoder_h)


    # luong attention
    attention = dot([decoder, encoder_outs], axes=[2, 2])
    attention = Activation('softmax', name='attention')(attention)

    context = dot([attention, encoder_outs], axes=[2,1])

    decoder_combined_context = concatenate([context, decoder])

    output = TimeDistributed(Dense(enc_lstm_units, activation="tanh"))(decoder_combined_context)
    output = TimeDistributed(Dense(output_dict_size, activation="softmax"))(output)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
    
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver) 
    with strategy.scope():
      model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
      model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if display_summary:
        model.summary()
    
    return model, params