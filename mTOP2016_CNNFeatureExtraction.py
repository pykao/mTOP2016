# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:44:28 2016

@author: poyu
"""
import os
# SET THE GPU DEVICE
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import theano
import sys
import lasagne
import lasagne.layers.dnn
from theano.tensor import TensorType


def get_CNNparameters():
    number_filter = 512
    filter_size = 8
    whiten = 1
    zca_t1 = 0.01
    zca_famd = 0.1
    pooling = [3, 3, 3]
    if whiten == 1:
        return number_filter, filter_size, [whiten, zca_t1, zca_famd], pooling
    if whiten == 0:
        return number_filter, filter_size, [whiten, None, None], pooling
# Loading the dataset
# DO NOT need to change anything
# Output:
# X_t1_train: labeled (#15)
# X_fa_train: labeled (#15)
# X_md_train: labeled (#15)
# y_train: label for the training dataset (#15)
# X_t1_test: unlabeled (#12)
# X_fa_test: unlabeled (#12)
# X_md_test: unlabeled (#12)

def load_dataset():
    print "Loading dataset..."
    X_t1_train = np.load('./numpy_data/X_T1_scaled_train.npy')
    X_fa_train = np.load('./numpy_data/X_FA_scaled_train.npy')
    X_md_train = np.load('./numpy_data/X_MD_scaled_train.npy')
    X_t1_test = np.load('./numpy_data/X_T1_scaled_test.npy')
    X_fa_test = np.load('./numpy_data/X_FA_scaled_test.npy')
    X_md_test = np.load('./numpy_data/X_MD_scaled_test.npy')
    y_train = np.load('./numpy_data/y_train.npy')
    return X_t1_train.astype(np.float32), X_fa_train.astype(np.float32), X_md_train.astype(np.float32), y_train.astype(np.float32), X_t1_test.astype(np.float32), X_fa_test.astype(np.float32), X_md_test.astype(np.float32)


# Loading the dictionary
# DO NOT need to change anything
# Input:
#   num_filter: number of filter
#   filter_size: size of filter
#   whiten: Need to whiten(1) the data or not unwhiten(0)
#   zca_whiten: whiten parameter for MR Ti images
#   zca_famd: whiten parameter for DT FA and DT MD images
# Output:
# X_t1_dictionary: original shape of MR T1 dictionary
# X_fa_dictionary: original shape of DT FA dictionary
# X_md_dictionary: original shape of DT MD dictionary

def load_dictionary(num_filter, filter_size, whiten, zca_t1, zca_famd):

    #working_dir = './dictionary_0.01_0.01_1024_10'

    if whiten == 0:
        t1_name = [os.path.join(root, name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'X_T1_dictionary' in name and str(num_filter) in name and 'size.'+str(filter_size) in name and 'unwhiten' in name and name.endswith('.npy')]

        fa_name = [os.path.join(root, name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'X_FA_dictionary' in name and str(num_filter) in name and 'size.'+str(filter_size) in name and 'unwhiten' in name and name.endswith('.npy')]

        md_name = [os.path.join(root, name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'X_MD_dictionary' in name and str(num_filter) in name and 'size.'+str(filter_size) in name and 'unwhiten' in name and name.endswith('.npy')]
    if whiten ==1:
        t1_name = [os.path.join(root, name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'X_T1_dictionary' in name and str(num_filter) in name and 'size.'+str(filter_size) in name and 'zcat1.'+str(zca_t1) in name and 'zcafamd.'+str(zca_famd) in name and name.endswith('.npy')]
        fa_name = [os.path.join(root, name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'X_FA_dictionary' in name and str(num_filter) in name and 'size.'+str(filter_size) in name and 'zcat1.'+str(zca_t1) in name and 'zcafamd.'+str(zca_famd) in name and name.endswith('.npy')]
        md_name = [os.path.join(root, name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'X_MD_dictionary' in name and str(num_filter) in name and 'size.'+str(filter_size) in name and 'zcat1.'+str(zca_t1) in name and 'zcafamd.'+str(zca_famd) in name and name.endswith('.npy')]
    #print type(t1_name), type(fa_name), type(md_name)

    #a =  os.path.join(working_dir, t1_name[0])
    print "MR T1 dictionary path:", t1_name[0]
    print "DT FA dictionary path:", fa_name[0]
    print "DT MD dictionary path:", md_name[0]
    X_t1_dictionary = np.load(t1_name[0])
    X_fa_dictionary = np.load(fa_name[0])
    X_md_dictionary = np.load(md_name[0])
    return X_t1_dictionary.astype(np.float32) , X_fa_dictionary.astype(np.float32), X_md_dictionary.astype(np.float32)
# Create the feature bank for cnn
def feature_bank(dictionary):
    filter_dim = int(round(dictionary.shape[0]**(1./3.)))
    filter_bank = np.ones(( dictionary.shape[1], 1, filter_dim, filter_dim, filter_dim), dtype = np.float32)
    for i in range(dictionary.shape[1]):
        tmp = dictionary[:,i]
        reshape_tmp = np.reshape(tmp, (filter_dim, filter_dim, filter_dim))
        filter_bank[i,0,:,:] = reshape_tmp
    return filter_bank.astype(np.float32)
# Convert data to theano form
def dataTheano(inputs):
    #print "Converting data to Theano form..."
    outputs = np.ones((inputs.shape[0],1,inputs.shape[1],inputs.shape[2],inputs.shape[3]))
    for i in range(inputs.shape[0]):
        outputs[i,0,:,:,:] = inputs[i,:,:,:]
    return outputs.astype(np.float32)

# Build CNN to extract feature
# NEED to change the num_filter, filter_size, whiten, zca_t1, zca_famd,
# NEED to change filter_size in 3D convolutional layers
# NEED to change pool_size in pooling layers
def build_cnn(input_var_t1 = None, input_var_fa = None, input_var_md = None):

    print "Loading Pretrained Parameters..."

    num_filter, filter_size_d, whiten, pooling_size = get_CNNparameters()


    print "Number of filter :", num_filter

    print "Filter size: ", filter_size_d

    print "Pooling size:", pooling_size

    if whiten[0] == 0:
        print 'Unwhiten...'
    if whiten[0] == 1:
        print 'Whiten data with ZCA ', str(whiten[1]), 'for MR T1, and ZCA ', str(whiten[2]), 'for DT FA and DT MD'

    X_t1_dictionary, X_fa_dictionary, X_md_dictionary = load_dictionary(num_filter, filter_size_d, whiten[0], whiten[1], whiten[2])

    print "Dictonaries type:", X_t1_dictionary.dtype, X_fa_dictionary.dtype, X_md_dictionary.dtype

    t1_pretrained = feature_bank(X_t1_dictionary)

    fa_pretrained = feature_bank(X_fa_dictionary)

    md_pretrained = feature_bank(X_md_dictionary)
    print t1_pretrained.shape, fa_pretrained.shape, md_pretrained.shape
    l_t1_in = lasagne.layers.InputLayer(shape = (None, 1,  156, 198, 144), input_var = input_var_t1)

    l_fa_in = lasagne.layers.InputLayer(shape = (None, 1, 78, 99, 72), input_var = input_var_fa)

    l_md_in = lasagne.layers.InputLayer(shape = (None, 1, 78, 99, 72), input_var = input_var_md)

    l_con1_t1 = lasagne.layers.dnn.Conv3DDNNLayer(l_t1_in, num_filters = t1_pretrained.shape[0], filter_size = (filter_size_d, filter_size_d, filter_size_d), stride = (2, 2, 2), W = t1_pretrained)

    l_pooling_t1 = lasagne.layers.dnn.Pool3DDNNLayer(l_con1_t1, pool_size = (pooling_size[0],pooling_size[1],pooling_size[2]))

    l_con1_fa = lasagne.layers.dnn.Conv3DDNNLayer(l_fa_in, num_filters = fa_pretrained.shape[0], filter_size = (filter_size_d/2,filter_size_d/2,filter_size_d/2), stride = (1, 1, 1), W = fa_pretrained)

    l_pooling_fa = lasagne.layers.dnn.Pool3DDNNLayer(l_con1_fa, pool_size = (pooling_size[0],pooling_size[1],pooling_size[2]))

    l_con1_md = lasagne.layers.dnn.Conv3DDNNLayer(l_md_in, num_filters = md_pretrained.shape[0], filter_size = (filter_size_d/2,filter_size_d/2,filter_size_d/2), stride = (1, 1, 1), W = md_pretrained)

    l_pooling_md = lasagne.layers.dnn.Pool3DDNNLayer(l_con1_md, pool_size = (pooling_size[0],pooling_size[1],pooling_size[2]))

    l_concat = lasagne.layers.ConcatLayer([l_pooling_t1, l_pooling_fa, l_pooling_md], axis=1)

    #l_den2 = lasagne.layers.DenseLayer(l_concat, num_units = 3, nonlinearity=lasagne.nonlinearities.softmax)

    #l_den1 = lasagne.layers.DenseLayer(l_concat, num_units = 1024, cnonlinearity=lasagne.nonlinearities.rectify)

    #l_den2 =  lasagne.layers.DenseLayer(l_den1, num_units = 3, nonlinearity=lasagne.nonlinearities.softmax)


    #param_count = lasagne.layers.count_params(l_concat)

    #print(param_count)

    #print(lasagne.layers.get_output_shape(l_con1_fa))

    #print t1_pretrained.max(), t1_pretrained.min()
    #print l_con1_t1.W.get_value().max(), l_con1_t1.W.get_value().min()

    #print fa_pretrained.max(), fa_pretrained.min()
    #print l_con1_fa.W.get_value().max(), l_con1_fa.W.get_value().min()

    #print md_pretrained.max(), md_pretrained.min()
    #print l_con1_md.W.get_value().max(), l_con1_md.W.get_value().min()
    #print all_params

    if (t1_pretrained == l_con1_t1.W.get_value()).all() and (fa_pretrained == l_con1_fa.W.get_value()).all() and (md_pretrained == l_con1_md.W.get_value()).all():
        print "Loaded all weights..."
    return l_concat


def main():

    # NEED to change the following parameters as the same as in build_cnn section

    # subject number starts from 0 to 26
    subject = int(sys.argv[1])

    # parameter for features
    num_filter, filter_size, whiten, pooling_size = get_CNNparameters()
    #working_dir = 'dictionary_0.01_0.01_1024_10'

    if whiten[0] == 1:
        working_filepath =[os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in dirs if 'number.'+str(num_filter) in name and 'size.'+str(filter_size) in name and 'zcat1.'+str(whiten[1]) in name and 'zcafamd.'+str(whiten[2]) in name]
        working_dir = working_filepath[0]

    if whiten[0] == 0:
        working_filepath = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in dirs if 'number.'+str(num_filter) in name and 'size.'+str(filter_size) in name and 'unwhiten' in name]
        working_dir = working_filepath[0]
    print "Working Directory: ", working_dir

    X_t1_train, X_fa_train, X_md_train, y_train, X_t1_test, X_fa_test, X_md_test = load_dataset()

    print "Training data type:" ,X_t1_train.dtype, X_fa_train.dtype, X_md_train.dtype, y_train.dtype, X_t1_test.dtype

    print "Converting data to Theano form..."

    X_t1_train_new = dataTheano(X_t1_train)

    #X_t1_train_new.astype(np.float32)

    X_fa_train_new = dataTheano(X_fa_train)

    #X_fa_train_new.astype(np.float32)

    X_md_train_new = dataTheano(X_md_train)

    #X_md_train_new.astype(np.float32)

    X_t1_test_new = dataTheano(X_t1_test)

    X_t1_new = np.concatenate((X_t1_train_new, X_t1_test_new), axis = 0)

    X_fa_test_new = dataTheano(X_fa_test)

    X_fa_new = np.concatenate((X_fa_train_new, X_fa_test_new), axis = 0)

    X_md_test_new = dataTheano(X_md_test)

    X_md_new = np.concatenate((X_md_train_new, X_md_test_new), axis = 0)

    #print X_t1_train_new.dtype, X_t1_new.dtype

    #print X_t1_train_new.shape, X_fa_train_new.shape, X_md_train_new.shape
    #print X_t1_test_new.shape, X_fa_test_new.shape, X_md_test_new.shape
    #print X_t1_new.shape, X_fa_new.shape, X_md_new.shape
    #print X_t1_new.dtype, X_fa_new.dtype, X_md_new.dtype


    # print X_t1_train_new.shape, X_fa_train_new.shape, X_md_train_new.shape
    #print "T1 shape: %s" % (X_t1_train_new.shape)
    #print "FA shape: %s" % (X_fa_train_new.shape)
    #print "MD shape: %s" % (X_md_train_new.shape)


    #X_t1_dictionary, X_fa_dictionary, X_md_dictionary = load_dictionary()
    #X_t1_featurebank = feature_bank(X_t1_dictionary)
    #X_fa_featurebank = feature_bank(X_fa_dictionary)
    #X_md_featurebank = feature_bank(X_md_dictionary)
    #print X_t1_featurebank.shape, X_fa_featurebank.shape, X_md_featurebank.shape
    #print X_t1_featurebank.dtype, X_fa_featurebank.dtype, X_md_featurebank.dtype

    dtensor5 = TensorType('float32', (False, False, False, False, False))

    # Prepare Theano variables for inputs and targets
    input_var_t1 = dtensor5('input_var_t1')
    input_var_fa = dtensor5('input_var_fa')
    input_var_md = dtensor5('input_var_md')
    # target_var = T.ivector('targets')

    print("Building model and compiling functions...")

    network = build_cnn(input_var_t1, input_var_fa, input_var_md)

    print "Number of parameters of network: %d" %lasagne.layers.count_params(network)

    print "Output shape of network: ", lasagne.layers.get_output_shape(network)

    # print lasagne.layers.get_all_layers(network)

    #all_params = lasagne.layers.get_all_params(network)

    #print all_params


    print("Get the output of the model...")

    # net1 = lasagne.layers.get_all_layers(network)

    output = lasagne.layers.get_output(network, {input_var_t1: input_var_t1,input_var_fa: input_var_fa, input_var_md: input_var_md},  deterministic = True)

    get_feature_function = theano.function([input_var_t1, input_var_fa, input_var_md], output)

    print "Subject : %d" %(subject+1)

    feature = get_feature_function(X_t1_new[subject:subject+1,:,:,:,:],X_fa_new[subject:subject+1,:,:,:,:],X_md_new[subject:subject+1,:,:,:,:])
    filename = 'subject'+str(subject+1).zfill(2)+'_pooling.'+str(pooling_size[0])+'.'+str(pooling_size[1])+'.'+str(pooling_size[2])+'_feature_representation.npy'


    filename1 = os.path.join(working_dir,filename)

    print filename1

        #print filename1

        # feature = get_feature_function(X_t1_new[:,:,:,:,:], X_fa_new[:,:,:,:,:], X_md_new[:,:,:,:,:])

    np.save(filename1, feature)

    print "Output Shape: ", feature.shape
    print "Maximun: %4f" % feature.max()
    print "Minumum: %.4f" % feature.min()

if __name__ == "__main__":
    main()
