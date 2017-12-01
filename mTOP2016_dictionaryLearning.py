# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 18:07:31 2016

@author: poyu
"""

import os
import numpy as np
import nibabel as nib
from scipy import linalg
from LFRKM import GainShapeKMeans

# This section gives you the boundary of non-zero region
def countNonzeroBoundary(model):

    # Get the filepaths of Brain MR T1 Image
    if model =='T1':
        brain_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "MR_T1" in name and name.endswith("nii")]
        brain_filepaths.sort()
        x_dim = 182
        y_dim = 218
        z_dim = 182
    elif model =='FA':
        brain_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "DT_FA" in name and name.endswith("nii")]
        brain_filepaths.sort()
        x_dim = 91
        y_dim = 109
        z_dim = 91
    elif model == 'MD':
        brain_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "DT_MD" in name and name.endswith("nii")]
        brain_filepaths.sort()
        x_dim = 91
        y_dim = 109
        z_dim = 91
    # Count non-zero boundary
    # x : 14 - 164
    # y : 14 - 200
    # z : 9 - 151
    boundaryMap = np.zeros((x_dim,y_dim,z_dim), dtype = np.int)

    for i in range(27):
        img_MRI = nib.load(brain_filepaths[i])
        MRI_data = img_MRI.get_data()
        boundaryMap = boundaryMap + MRI_data

    x_lower_count = 0
    x_upper_count = x_dim-1
    while (np.sum(boundaryMap[x_lower_count,:,:])==0):
        x_lower_count = x_lower_count + 1

    while (np.sum(boundaryMap[x_upper_count,:,:])==0):
        x_upper_count = x_upper_count - 1

    y_lower_count = 0
    y_upper_count = y_dim-1

    while (np.sum(boundaryMap[:,y_lower_count,:])==0):
        y_lower_count = y_lower_count + 1

    while (np.sum(boundaryMap[:,y_upper_count,:])==0):
        y_upper_count = y_upper_count - 1

    z_lower_count = 0
    z_upper_count = z_dim-1

    while (np.sum(boundaryMap[:,:,z_lower_count])==0):
        z_lower_count = z_lower_count + 1

    while (np.sum(boundaryMap[:,:,z_upper_count])==0):
        z_upper_count = z_upper_count - 1
    return [x_lower_count, x_upper_count, y_lower_count, y_upper_count, z_lower_count, z_upper_count]


def createBrainData(model, window_size, step_size, boundary, threshold = 0.25):

    if model == 'T1':
        # Get the filepaths of Brain MR T1 Image
        brain_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "MR_T1" in name and name.endswith("nii")]
        brain_filepaths.sort()
    elif model =='FA':
        brain_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "DT_FA" in name and name.endswith("nii")]
        brain_filepaths.sort()
    elif model == 'MD':
        brain_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "DT_MD" in name and name.endswith("nii")]
        brain_filepaths.sort()

    # The patches size
    w = window_size
    num_threshold = threshold * w**3
    # step_size
    s = step_size

    count_patches = len(range(boundary[0],boundary[1],s))*len(range(boundary[2],boundary[3],s))*len(range(boundary[4],boundary[5],s))*27

    sub = np.zeros([w**3, count_patches ], dtype = np.float32)
    subpatch = np.zeros([w, w, w], dtype = np.float32)

    count = 0
    for num_sub in range(27):
        # Load the MR T1 Image
        img_MRI = nib.load(brain_filepaths[num_sub])
        MRI_data = img_MRI.get_data()
        for i in range(boundary[0],boundary[1],s):
            for j in range(boundary[2],boundary[3],s):
                for k in range(boundary[4],boundary[5],s):
                    if np.count_nonzero(MRI_data[i:i+w, j:j+w, k:k+w]) >= num_threshold:
                        subpatch = MRI_data[i:i+w, j:j+w, k:k+w]
                        #reshaped_sub = np.reshape(subpatch, (1,w**3))
                        reshaped_sub = np.reshape(subpatch, (1,-1))
                       # if reshaped_sub.shape == sub[:,count]:
                        sub[:, count] = reshaped_sub
                        count = count + 1
    X = sub[:,:count]
    print X.shape
    return X.shape, X

def normalizedDataFeatureLearning(inputs):
    #inputs = np.copy(inputs) # this allows you to have original dtat
    (x_dim,y_dim) =inputs.shape
    for i in range(y_dim):
        mean = np.mean(inputs[:,i])
        std = np.std(inputs[:,i])
        if std == 0:
            print('Warning: Standard deviation is zero!!!')
        inputs[:,i] = (inputs[:,i]-mean)/std
    #return inputs # this allows you to have original dtat

def ZCA_whiten(inputs, eps = 0.01): # inputs: NXM (https://gist.github.com/duschendestroyer/5170087)
    #Correlation matrix MxM
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1]
    #Singular Value Decomposition
    #U,S,V = np.linalg.svd(sigma)
    U,S,V = linalg.svd(sigma)
    #Whitening constant, it prevents division by zero
    epsilon = eps
    #ZCA Whitening matrix
    tmp = np.dot(U, np.diag(1.0/np.sqrt(S+epsilon)))
    ZCAMatrix = np.dot(tmp, U.T)

    return np.dot(ZCAMatrix,inputs)   # Data whitening (http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening)
'''
def createInitialDictionary(nfeature, nsample):
    random_dictionary = np.random.standard_normal(size = (nfeature,nsample))
    lengths = np.sqrt((random_dictionary**2).sum(axis=0))
    initial_dictionary = random_dictionary/lengths
    return initial_dictionary
# Data
def transformation(X, dictionary):
    X = X.T
    dictionary = dictionary.T
    code = np.dot(X, dictionary)
    mask = np.zeros(code.shape)
    mask[xrange(X.shape[0]),abs(code).argmax(axis=1)] = 1
    code *= mask
    code = code.T
    return code
'''
# ==========================Creating the folder================================
def createDictionaryFolder(whiten, number_filter, window_size, zca_t1, zca_famd):
    print "Creating the folder to save the dictionary..."
    if whiten == 1:
        foldername = "number."+str(number_filter)+".size."+str(window_size)+".zcat1."+str(zca_t1)+".zcafamd."+str(zca_famd)
        folder_dir = os.path.join(os.getcwd(), foldername)
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
    if whiten == 0:
        foldername = "number."+str(number_filter)+".size."+str(window_size)+".unwhiten"
        folder_dir = os.path.join(os.getcwd(), foldername)
        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
    return folder_dir

# ===================================MR T1=====================================
# ========================= Create dictionary for T1 ==========================
def createT1Dictionary(whiten, number_filter, window_size, zca_t1, zca_famd,  step_size, folder_dir, threshold=0.25):
    print "Creating dictionary for T1..."
    boundary_t1 = countNonzeroBoundary('T1')
    _, X_t1 = createBrainData('T1', window_size, step_size, boundary_t1, threshold = 0.25)
    normalizedDataFeatureLearning(X_t1)
    if whiten == 1:
        X_t1_normalized_zca = ZCA_whiten(X_t1, zca_t1)
    if whiten == 0:
        X_t1_normalized_zca = X_t1

    T1_Dictionary = GainShapeKMeans(number_filter)

    T1_Dictionary.fit(X_t1_normalized_zca.T)

    if whiten == 1:
        X_T1_filename = "X_T1_dictionary."+"number."+str(number_filter)+".size."+str(window_size)+".zcat1."+str(zca_t1)+".zcafamd."+str(zca_famd)
        X_T1_dir = os.path.join(folder_dir, X_T1_filename)
        np.save(X_T1_dir+'.npy', T1_Dictionary.dictionary)
        #np.savetxt(X_T1_dir+'.txt', T1_Dictionary.dictionary)

    if whiten == 0:
        X_T1_filename = "X_T1_dictionary."+"number."+str(number_filter)+".size."+str(window_size)+".unwhiten"
        X_T1_dir = os.path.join(folder_dir, X_T1_filename)
        np.save(X_T1_dir+'.npy', T1_Dictionary.dictionary)
        #np.savetxt(X_T1_dir+'.txt', T1_Dictionary.dictionary)

# =======================================DT FA=================================
# ============================ Create dictionary for FA =======================
def createFADictionary(whiten, number_filter, window_size, zca_t1, zca_famd, step_size, folder_dir, threshold=0.25):
    print "Creating dictionary for FA..."
    boundary_fa = countNonzeroBoundary('FA')
    _, X_fa = createBrainData('FA', window_size/2, step_size/2, boundary_fa, threshold = 0.25)
    normalizedDataFeatureLearning(X_fa)
    if whiten == 1:
        X_fa_normalized_zca = ZCA_whiten(X_fa, zca_famd)
    if whiten == 0:
        X_fa_normalized_zca = X_fa

    FA_Dictionary = GainShapeKMeans(number_filter)

    FA_Dictionary.fit(X_fa_normalized_zca.T)

    if whiten == 1:
        X_FA_filename = "X_FA_dictionary."+"number."+str(number_filter)+".size."+str(window_size)+".zcat1."+str(zca_t1)+".zcafamd."+str(zca_famd)
        X_FA_dir = os.path.join(folder_dir, X_FA_filename)
        np.save(X_FA_dir+'.npy', FA_Dictionary.dictionary)
        #np.savetxt(X_FA_dir+'.txt', FA_Dictionary.dictionary)

    if whiten == 0:
        X_FA_filename = "X_FA_dictionary."+"number."+str(number_filter)+".size."+str(window_size)+".unwhiten"
        X_FA_dir = os.path.join(folder_dir, X_FA_filename)
        np.save(X_FA_dir+'.npy', FA_Dictionary.dictionary)
        #np.savetxt(X_FA_dir+'.txt', FA_Dictionary.dictionary)

# ====================================DT MD====================================
# ========================== Create dictionary for MD =========================
def createMDDictionary(whiten, number_filter, window_size, zca_t1, zca_famd, step_size, folder_dir, threshold=0.25):
    print "Creating dictioary for MD..."

    boundary_md = countNonzeroBoundary('MD')

    _, X_md = createBrainData('MD', window_size/2, step_size/2, boundary_md, threshold = 0.25)

    normalizedDataFeatureLearning(X_md)

    if whiten == 1:
        X_md_normalized_zca = ZCA_whiten(X_md, zca_famd)
    if whiten == 0:
        X_md_normalized_zca = X_md

    MD_Dictionary = GainShapeKMeans(number_filter)

    MD_Dictionary.fit(X_md_normalized_zca.T)

    if whiten == 1:
        X_MD_filename = "X_MD_dictionary."+"number."+str(number_filter)+".size."+str(window_size)+".zcat1."+str(zca_t1)+".zcafamd."+str(zca_famd)
        X_MD_dir = os.path.join(folder_dir, X_MD_filename)
        np.save(X_MD_dir+'.npy', MD_Dictionary.dictionary)
        #np.savetxt(X_MD_dir+'.txt', MD_Dictionary.dictionary)

    if whiten == 0:
        X_MD_filename = "X_MD_dictionary."+"number."+str(number_filter)+".size."+str(window_size)+".unwhiten"
        X_MD_dir = os.path.join(folder_dir, X_MD_filename)
        np.save(X_MD_dir+'.npy', MD_Dictionary.dictionary)
        #np.savetxt(X_MD_dir+'.txt', MD_Dictionary.dictionary)

def main():
    # Chagne the variable here
    # If whiten == 1, whiten the image, if whiten == 0, unwhiten the image
    whiten = 1

    # Feature size, fa and md has half of the size
    window_size = 8

    # Step_size for cropping image
    step_size = 4

    # threshold if it has less than 25% of non-zero voxel, this patch will be threw away
    threshold = 0.25

    # number of feature we want to learn


    number_filter = 3000

    # ZCA whiten parameter
    zca_t1 = 0.01

    zca_famd = 0.1

    print "Running K-means dictionary learning"
    if whiten == 0:
        print "unwhiten dataset..."
    if whiten == 1:
        print "whiten dataset with %.3f for T1 and %.3f for FA&MD..." % (zca_t1, zca_famd)

    print "feature size: %d..." % window_size

    print "Number of features: %d..." % number_filter


    folder_dir = createDictionaryFolder(whiten, number_filter, window_size, zca_t1, zca_famd)

    createT1Dictionary(whiten, number_filter, window_size, zca_t1, zca_famd, step_size, folder_dir, threshold)

    createFADictionary(whiten, number_filter, window_size, zca_t1, zca_famd, step_size, folder_dir, threshold)

    createMDDictionary(whiten, number_filter, window_size, zca_t1, zca_famd, step_size, folder_dir, threshold)


if __name__ == "__main__":
    main()






