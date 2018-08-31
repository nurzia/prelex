from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, SGD
from keras import backend as K
from keras.layers import Dense, Dropout, Bidirectional,\
    Input, Lambda, Embedding, LSTM, Flatten, TimeDistributed,\
    Activation, BatchNormalization

def build_model(bptt=50, input_dim=128,
                recurrent_dim=100, lr=0.001,
                dropout=0.0, num_layers=1):
    """
    function building a model with n BLSTMs with m parameters and n BLSTMs with m/2 parameters and a final TDD that gives the output
    """
    input_ = Input(shape=(bptt, input_dim), dtype='float32', name='input_')

    bn = BatchNormalization(axis=1, input_shape=(bptt, input_))(input_)

    for i in range(num_layers):
        if i == 0:
            curr_input = bn
        else:
            curr_input = curr_enc_out
        
        curr_enc_out = Bidirectional(LSTM(units=recurrent_dim,
                                          return_sequences=True,
                                          activation='tanh',
                                          name='enc_lstm_'+str(i + 1)),
                                     merge_mode='sum')(curr_input)

    output_ = TimeDistributed(Dense(2, activation='softmax'), name='output_')(curr_enc_out)

    model = Model(inputs=input_, outputs=output_)
    optim = Adam(lr=lr)
    model.compile(optimizer=optim,
                  loss={'output_': 'categorical_crossentropy'},
                  metrics=['accuracy'])
    return model