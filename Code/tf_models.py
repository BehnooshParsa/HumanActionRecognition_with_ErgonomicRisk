import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda, concatenate,multiply, add
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial
clipped_relu = partial(relu, max_value=5)

def max_filter(x):
    # Max over the best filter score (like ICRA paper)
    max_values = K.max(x, 2, keepdims=True)
    max_flag = tf.greater_equal(x, max_values)
    out = x * tf.cast(max_flag, tf.float32)
    return out

def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True)+1e-5
    out = x / max_values
    return out

def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)  
    return multiply([tanh_out, sigm_out])

#  -------------------------------------------------------------
def temporal_convs_linear(n_nodes, conv_len, n_classes, n_feat, max_len, 
                        causal=False, loss='categorical_crossentropy', 
                        optimizer='adam', return_param_str=False):
    """ Used in paper: 
    Segmental Spatiotemporal CNNs for Fine-grained Action Segmentation
    Lea et al. ECCV 2016

    Note: Spatial dropout was not used in the original paper. 
    It tends to improve performance a little.  
    """

    inputs = Input(shape=(max_len,n_feat))
    if causal: model = ZeroPadding1D((conv_len//2,0))(model)
    model = Convolution1D(n_nodes, conv_len, input_dim=n_feat, input_length=max_len, border_mode='same', activation='relu')(inputs)
    if causal: model = Cropping1D((0,conv_len//2))(model)

    model = SpatialDropout1D(0.3)(model)

    model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)
    
    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal")

    if return_param_str:
        param_str = "tConv_C{}".format(conv_len)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model


def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len, 
            loss='categorical_crossentropy', causal=False, 
            optimizer="rmsprop", activation='norm_relu',
            return_param_str=False):
    n_layers = len(n_nodes)
    inputs = Input(shape=(max_len,n_feat))
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = Convolution1D(n_nodes[i], conv_len, border_mode='same')(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)
        
        if activation=='norm_relu': 
            model = Activation('relu')(model)            
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)            
        
        model = MaxPooling1D(2)(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = Convolution1D(n_nodes[-i-1], conv_len, border_mode='same')(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation=='norm_relu': 
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)
    
    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model

def ED_TCN_atrous(n_nodes, conv_len, n_classes, n_feat, max_len, 
                loss='categorical_crossentropy', causal=False, 
                optimizer="rmsprop", activation='norm_relu',
                return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(None,n_feat))
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = AtrousConvolution1D(n_nodes[i], conv_len, atrous_rate=i+1, border_mode='same')(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)
        
        if activation=='norm_relu': 
            model = Activation('relu')(model)            
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)            

    # ---- Decoder ----
    for i in range(n_layers):
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = AtrousConvolution1D(n_nodes[-i-1], conv_len, atrous_rate=n_layers-i, border_mode='same')(model)      
        if causal: model = Cropping1D((0,conv_len//2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation=='norm_relu': 
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax" ))(model)

    model = Model(input=inputs, output=model)

    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-TCNa_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model



def TimeDelayNeuralNetwork(n_nodes, conv_len, n_classes, n_feat, max_len, 
                loss='categorical_crossentropy', causal=False, 
                optimizer="rmsprop", activation='sigmoid',
                return_param_str=False):
    # Time-delay neural network
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len,n_feat))
    model = inputs
    inputs_mask = Input(shape=(max_len,1))
    model_masks = [inputs_mask]

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len//2,0))(model)
        model = AtrousConvolution1D(n_nodes[i], conv_len, atrous_rate=i+1, border_mode='same')(model)
        # model = SpatialDropout1D(0.3)(model)
        if causal: model = Cropping1D((0,conv_len//2))(model)
        
        if activation=='norm_relu': 
            model = Activation('relu')(model)            
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation=='wavenet': 
            model = WaveNet_activation(model) 
        else:
            model = Activation(activation)(model)            

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "TDN_C{}".format(conv_len)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model



def Dilated_TCN(num_feat, num_classes, nb_filters, dilation_depth, nb_stacks, max_len, 
            activation="wavenet", tail_conv=1, use_skip_connections=True, causal=False, 
            optimizer='adam', return_param_str=False):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """

    def residual_block(x, s, i, activation):
        original_x = x

        if causal:
            x = ZeroPadding1D(((2**i)//2,0))(x)
            conv = AtrousConvolution1D(nb_filters, 2, atrous_rate=2**i, border_mode='same',
                                        name='dilated_conv_%d_tanh_s%d' % (2**i, s))(x)
            conv = Cropping1D((0,(2**i)//2))(conv)
        else:
            conv = AtrousConvolution1D(nb_filters, 3, atrous_rate=2**i, border_mode='same',
                                    name='dilated_conv_%d_tanh_s%d' % (2**i, s))(x)                                        

        conv = SpatialDropout1D(0.3)(conv)
        # x = WaveNet_activation(conv)

        if activation=='norm_relu': 
            x = Activation('relu')(conv)
            x = Lambda(channel_normalization)(x)
        elif activation=='wavenet': 
            x = WaveNet_activation(conv) 
        else:
            x = Activation(activation)(conv)        

        #res_x  = Convolution1D(nb_filters, 1, border_mode='same')(x)
        #skip_x = Convolution1D(nb_filters, 1, border_mode='same')(x)
        x  = Convolution1D(nb_filters, 1, border_mode='same')(x)

        res_x = add([original_x, x])

        #return res_x, skip_x
        return res_x, x

    input_layer = Input(shape=(max_len, num_feat))

    skip_connections = []

    x = input_layer
    if causal:
        x = ZeroPadding1D((1,0))(x)
        x = Convolution1D(nb_filters, 2, border_mode='same', name='initial_conv')(x)
        x = Cropping1D((0,1))(x)
    else:
        x = Convolution1D(nb_filters, 3, border_mode='same', name='initial_conv')(x)    

    for s in range(nb_stacks):
        for i in range(0, dilation_depth+1):
            x, skip_out = residual_block(x, s, i, activation)
            skip_connections.append(skip_out)

    if use_skip_connections:
        x = add(skip_connections)
    x = Activation('relu')(x)
    x = Convolution1D(nb_filters, tail_conv, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution1D(num_classes, tail_conv, border_mode='same')(x)
    x = Activation('softmax', name='output_softmax')(x)

    model = Model(input_layer, x)
    model.compile(optimizer, loss='categorical_crossentropy', sample_weight_mode='temporal', metrics=['accuracy'])

    if return_param_str:
        param_str = "D-TCN_C{}_B{}_L{}".format(2, nb_stacks, dilation_depth)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model

def BidirLSTM(n_nodes, n_classes, n_feat, max_len=None, 
                causal=True, loss='categorical_crossentropy', optimizer="adam",
                return_param_str=False):
    
    inputs = Input(shape=(None,n_feat))
    model = LSTM(n_nodes, return_sequences=True)(inputs)

    # Birdirectional LSTM
    if not causal:
        model_backwards = LSTM(n_nodes, return_sequences=True, go_backwards=True)(inputs)
        # model = merge([model, model_backwards],mode="concat")
        model = concatenate([model, model_backwards])

    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)
    
    model = Model(input=inputs, output=model)
    model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal", metrics=['accuracy'])
    
    if return_param_str:
        param_str = "LSTM_N{}".format(n_nodes)
        if causal:
            param_str += "_causal"
    
        return model, param_str
    else:
        return model

def repeat_vector(args):
        sequence_layer = args[0]
        return RepeatVector(K.shape(sequence_layer)[1])


def ED_LSTM(n_nodes, n_classes, n_feat, max_len, 
                causal=True, loss='categorical_crossentropy', optimizer="adam",
                return_param_str=True):
	
	# # for layer in model.layers:
	# # 	print(layer.output_shape)
	# print("Here1")
	# model.add(LSTM(n_feat, return_sequences=True))
	# print("Here2")	

	#alternate
	inputs = Input(shape=(max_len, n_feat))
	model = Sequential()
	model.add(LSTM(n_nodes,input_shape = (max_len,n_feat), return_sequences = False))
	model.add(RepeatVector(max_len))
	model.add(LSTM(n_feat, return_sequences=True))
	model.add(Dense(n_classes, activation="softmax"))
	

	# inputs = Input(shape=(max_len, n_feat))
	# encoded = LSTM(n_nodes)(inputs)
	# encoded1 = LSTM(512)(encoded)
	# encoded2 = LSTM(256)(encoded1)
	# #print(K.int_shape(encoded))
	# #print(K.int_shape(inputs)[1])
	# decoded = RepeatVector(max_len)(encoded2)
	# #decoded = Lambda(repeat_vector, output_shape=(None, n_feat)) ([encoded, inputs])
	# print("Here1")
	# decoded1 = LSTM(1024, return_sequences=True)(decoded)
	# decoded2 = LSTM(4096, return_sequences=True)(decoded1)
	# print("Here2")
	# midoutput = Model(inputs,decoded2)
	# model = Sequential()
	# model.add(midoutput)
	# model.add(TimeDistributed(Dense(n_classes, activation="softmax")))


	#alternate code
	# num_encoder_tokens = 40
	# num_decoder_tokens = 93
	# latent_dim = 256
	# # Define an input sequence and process it.
	# encoder_inputs = Input(shape=(None, num_encoder_tokens))
	# encoder = LSTM(latent_dim, return_state=True)
	# encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# # We discard `encoder_outputs` and only keep the states.
	# encoder_states = [state_h, state_c]
	# # Set up the decoder, using `encoder_states` as initial state.
	# decoder_inputs = Input(shape=(None, num_decoder_tokens))
	# # We set up our decoder to return full output sequences,
	# # and to return internal states as well. We don't use the
	# # return states in the training model, but we will use them in inference.
	# decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
	# decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	# decoder_dense = Dense(num_decoder_tokens, activation='softmax')
	# decoder_outputs = decoder_dense(decoder_outputs)
	# # Define the model that will turn
	# # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	# model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# # plot the model
	# plot_model(model, to_file='model.png', show_shapes=True)

	model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal", metrics=['accuracy'])
	print("Here2")
	if return_param_str:
		param_str = "ED_LSTM_N{}".format(n_nodes)
		if causal:
			param_str += "_causal"

		return model, param_str
	else:
		return model


def Deep_LSTM(n_nodes, n_classes, n_feat, max_len, 
                causal=True, loss='categorical_crossentropy', optimizer="adam",
                return_param_str=True):

	model = Sequential()
	model.add(LSTM(n_nodes, return_sequences=True,input_shape=(None,n_feat)))
	model.add(LSTM(n_nodes, return_sequences=True,input_shape=(None,n_feat)))
	model.add(TimeDistributed(Dense(n_classes, activation="softmax")))
	model.compile(optimizer=optimizer, loss=loss, sample_weight_mode="temporal", metrics=['accuracy'])
	
	if return_param_str:
		param_str = "Deep_LSTM_N{}".format(n_nodes)
		if causal:
			param_str += "_causal"

		return model, param_str
	else:
		return model
