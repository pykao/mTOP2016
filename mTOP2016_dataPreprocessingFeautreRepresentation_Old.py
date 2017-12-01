# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:44:28 2016

@author: poyu
"""
import os
import nibabel as nib
import numpy as np

#def load_dataset():

def normalized_3d(inputs):
    inputs=np.copy(inputs)
    x_dim, y_dim, z_dim = inputs.shape
    number_non_zero = np.float32(np.count_nonzero(inputs))
    sum_inputs = np.float32(inputs.sum())
    mean_inputs = sum_inputs/number_non_zero
    input_nonzero = inputs[np.nonzero(inputs)]
    std_input = np.std(input_nonzero)
    for i in range(x_dim):
        for j in range(y_dim):
            for k in range(z_dim):
                if inputs[i,j,k] != 0 :
                    inputs[i,j,k] = (inputs[i,j,k]-mean_inputs)/std_input
    return inputs

boundary_t1 = [11, 167, 9, 207, 9, 153]

t1_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "MR_T1" in name and name.endswith("nii")]

t1_filepaths.sort()

boundary_fa = [6, 84, 5, 104, 8, 80]

fa_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "DT_FA" in name and name.endswith("nii")]

fa_filepaths.sort()

boundary_md = [6, 84, 5, 104, 8, 80]

md_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "Brain" in name and "DT_MD" in name and name.endswith("nii")]

md_filepaths.sort()

# =============================Train=========================================================

train_subject = np.array([3, 4, 5, 7, 8, 10, 11, 13, 16, 17, 20, 22, 25, 26, 27])-1

train_label = np.array([1, 2, 0, 2, 2, 1, 0, 2, 1, 0, 1, 0, 2, 1, 0])

train_t1 = np.zeros((train_subject.shape[0],156,198,144), dtype = np.float32)
train_t1_scaled = train_t1

train_fa = np.zeros((train_subject.shape[0],78,99,72), dtype = np.float32)
train_fa_scaled = train_fa

train_md = np.zeros((train_subject.shape[0],78,99,72), dtype = np.float32)
train_md_scaled = train_md

count_train = 0
for sub in train_subject:

    img_t1 = nib.load(t1_filepaths[sub])
    t1_data = img_t1.get_data()
    train_t1[count_train,:] = t1_data[boundary_t1[0]:boundary_t1[1],boundary_t1[2]:boundary_t1[3],boundary_t1[4]:boundary_t1[5]]
    train_t1_scaled[count_train,:] = normalized_3d(train_t1[count_train,:])

    img_fa = nib.load(fa_filepaths[sub])
    fa_data = img_fa.get_data()
    train_fa[count_train,:] = fa_data[boundary_fa[0]:boundary_fa[1],boundary_fa[2]:boundary_fa[3],boundary_fa[4]:boundary_fa[5]]
    train_fa_scaled[count_train,:] = normalized_3d(train_fa[count_train,:])

    img_md = nib.load(md_filepaths[sub])
    md_data = img_md.get_data()
    train_md[count_train,:] = md_data[boundary_md[0]:boundary_md[1],boundary_md[2]:boundary_md[3],boundary_md[4]:boundary_md[5]]
    train_md_scaled[count_train,:] = normalized_3d(train_md[count_train,:])

    count_train = count_train+1

test_subject = np.array([1, 2, 6, 9, 12, 14, 15, 18, 19, 21, 23, 24])-1

test_t1 = np.zeros((test_subject.shape[0],156,198,144), dtype = np.float32)
test_t1_scaled = test_t1

test_fa = np.zeros((test_subject.shape[0],78,99,72), dtype = np.float32)
test_fa_scaled = test_fa

test_md = np.zeros((test_subject.shape[0],78,99,72), dtype = np.float32)
test_md_scaled = test_md

count_test = 0
for sub in test_subject:

    img_t1 = nib.load(t1_filepaths[sub])
    t1_data = img_t1.get_data()
    test_t1[count_test,:] = t1_data[boundary_t1[0]:boundary_t1[1],boundary_t1[2]:boundary_t1[3],boundary_t1[4]:boundary_t1[5]]
    test_t1_scaled[count_test,:] = normalized_3d(test_t1[count_test,:] )
    img_fa = nib.load(fa_filepaths[sub])
    fa_data = img_fa.get_data()
    test_fa[count_test,:] = fa_data[boundary_fa[0]:boundary_fa[1],boundary_fa[2]:boundary_fa[3],boundary_fa[4]:boundary_fa[5]]
    test_fa_scaled[count_test,:] = normalized_3d(test_fa[count_test,:])
    img_md = nib.load(md_filepaths[sub])
    md_data = img_md.get_data()
    test_md[count_test,:] = md_data[boundary_md[0]:boundary_md[1],boundary_md[2]:boundary_md[3],boundary_md[4]:boundary_md[5]]
    test_md_scaled[count_test,:] = normalized_3d(test_md[count_test,:])
    count_test = count_test+1

print train_t1_scaled.shape, train_fa_scaled.shape, train_md_scaled.shape
print test_t1_scaled.shape, test_fa_scaled.shape, test_md_scaled.shape
#np.save('X_T1_train.npy', train_t1)
np.save('X_T1_scaled_train.npy', train_t1_scaled)
#np.save('X_FA_train.npy', train_fa)
np.save('X_FA_scaled_train.npy', train_fa_scaled)
#np.save('X_MD_train.npy', train_md)
np.save('X_MD_scaled_train.npy', train_md_scaled)
#np.save('y_train.npy', train_label)

#np.save('X_T1_test.npy', test_t1)
np.save('X_T1_scaled_test.npy', test_t1_scaled)
#np.save('X_FA_test.npy', test_fa)
np.save('X_FA_scaled_test.npy', test_fa_scaled)
#np.save('X_MD_test.npy', test_md)
np.save('X_MD_scaled_test.npy', test_md_scaled)
