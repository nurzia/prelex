from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers import Dense, Dropout, Bidirectional,\
    Input, Lambda, Embedding, LSTM, Flatten, TimeDistributed,\
    Activation, BatchNormalization, GRU


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
        
        curr_enc_out = LSTM(units=recurrent_dim,
                            return_sequences=True,
                            activation='tanh')(curr_input)

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
        
        curr_enc_out = LSTM(units=recurrent_dim,
                            return_sequences=True,
                            activation='tanh')(curr_input)

    backward_output = TimeDistributed(Dense(input_dim, activation='linear'),
                              name='backward_out')(curr_enc_out)


    model = Model(inputs=[forward_input, backward_input],
                  outputs=[forward_output, backward_output])
    optim = Adam(lr=lr)
    model.compile(optimizer=optim,
                  loss={'forward_out': 'mse', 'backward_out': 'mse'})

    return model