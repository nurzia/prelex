from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers import Dense, Dropout, Bidirectional,\
    Input, Lambda, Embedding, LSTM, Flatten, TimeDistributed,\
    Activation, BatchNormalization, GRU


def build_model(bptt=50, input_dim=128,
                recurrent_dim=100, lr=0.001,
                dropout=0.0, num_layers=1):
    """
    function building a model with n BLSTMs with m parameters and n BLSTMs with m/2 parameters and a final TDD that gives the output
    """
    input_ = Input(shape=(bptt, input_dim), dtype='float32', name='input_')

    bn = BatchNormalization(axis=1, input_shape=(bptt, input_dim))(input_)

    for i in range(num_layers):
        if i == 0:
            curr_input = bn
        else:
            curr_input = curr_enc_out
        
        curr_enc_out = Bidirectional(GRU(units=recurrent_dim,
                                          return_sequences=True,
                                          activation='tanh',
                                          name='enc_'+str(i + 1)),
                                     merge_mode='sum')(curr_input)

    output_ = TimeDistributed(Dense(2, activation='softmax'), name='output_')(curr_enc_out)

    model = Model(inputs=input_, outputs=output_)
    optim = Adam(lr=lr)
    model.compile(optimizer=optim,
                  loss={'output_': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model


def build_lm(bptt=50, input_dim=128, recurrent_dim=100,
             lr=0.001, num_layers=1):
    # statefulness?!
    
    # forward lm:
    forward_input = Input(shape=(bptt, input_dim), dtype='float32', name='forward_in')
    bn = BatchNormalization(axis=1, input_shape=(bptt, input_dim))(forward_input)

    for i in range(num_layers):
        if i == 0:
            curr_input = bn
        else:
            curr_input = curr_enc_out
        
        curr_enc_out = Bidirectional(LSTM(units=recurrent_dim,
                                          return_sequences=True,
                                          activation='tanh'),
                                     merge_mode='sum')(curr_input)

    forward_output = TimeDistributed(Dense(input_dim, activation='linear'),
                              name='forward_out')(curr_enc_out)

    # backward lm:
    backward_input = Input(shape=(bptt, input_dim), dtype='float32', name='backward_in')
    bn = BatchNormalization(axis=1, input_shape=(bptt, input_dim))(backward_input)

    for i in range(num_layers):
        if i == 0:
            curr_input = bn
        else:
            curr_input = curr_enc_out
        
        curr_enc_out = Bidirectional(LSTM(units=recurrent_dim,
                                          return_sequences=True,
                                          activation='tanh'),
                                     merge_mode='sum')(curr_input)

    backward_output = TimeDistributed(Dense(input_dim, activation='linear'),
                              name='backward_out')(curr_enc_out)


    model = Model(inputs=[forward_input, backward_input],
                  outputs=[forward_output, backward_output])
    optim = Adam(lr=lr)
    model.compile(optimizer=optim,
                  loss={'forward_out': 'mse', 'backward_out': 'mse'})

    return model