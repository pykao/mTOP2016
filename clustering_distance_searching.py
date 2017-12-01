import os
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn import metrics
from sklearn.cluster import KMeans
np.set_printoptions(precision = 4)
def getParameters():
    num_filter = 512
    filter_size = 8
    whiten = 1
    zca_t1 = 0.01
    zca_famd = 0.1
    pooling_size = [3,3,3]
    return num_filter, filter_size, whiten, zca_t1, zca_famd, pooling_size

def distance_searching_label(D):
    subject = range(27)

    label0 = [None]*9
    
    label1 = [None]*9
    
    label2 = [None]*9
    
    #=====27 largest distance between two points =====# 
    
    largest_distance = np.unravel_index(D.argmax(),D.shape)
    
    #print largest_distance[0], largest_distance[1]
    
    
    #==== first initial point to label0====#
    label0[0] = largest_distance[0]
    
    
    remain26 = [x for x in subject if x !=  largest_distance[0]] 
    #print remain26
    
    
    #====26 first initial point to label1====#
    label1[0] = largest_distance[1]
    
    #print label0
    #print label1
    
    
    remain25 = [x for x in remain26 if x != largest_distance[1]]
    #print remain25
    
    #label0[1] = [x for x in remain25 if D[label0[0]]
    
    #for i in remain25:
    #    print label0[0], i, D[label0[0]][i]    
    
    # =====25 Assign one point to label 0 ===== #
    dist25 = [D[label0[0]][x] for x in remain25]
    
    index25 = dist25.index(min(dist25))
    
    #print remain25
    #print remain25[index25]
    
    label0[1] = remain25[index25]
    
    #
    
    #print label0
    
    remain24 = [x for x in remain25 if x != label0[1]]
    
    # =====24 Assign one point to label 1 ===== #
    
    #print remain24
    
    #print len(remain24)
    
    #for i in remain24:
    #    print label1[0], i, D[label1[0]][i]
    
    dist24 = [D[label1[0]][x] for x in remain24]
    
    #print dist24
    
    
    index24 = dist24.index(min(dist24))
    
    #print min(dist24)
    
    label1[1] = remain24[index24]
    
    #print label0
    #print label1
    
    remain23 = [x for x in remain24 if x != label1[1]]
    
    # =====23 Assign one point to label 0 ===== #
    
    #print remain23
    #for i in remain23:
        #print label0[0], i, D[label0[0]][i]
    #    print label0[1], i, D[label0[1]][i]
    
    dist23_0 = [D[label0[0]][x] for x in remain23]
    dist23_1 = [D[label0[1]][x] for x in remain23]
    
    if min(dist23_0) <= min(dist23_1):
        dist23 = dist23_0
    if min(dist23_1) <= min(dist23_0):
        dist23 = dist23_1
    
    #print min(dist23)
    
    index23 = dist23.index(min(dist23))
    
    #print index23
    
    label0[2] = remain23[index23]
    
    #print label0
    #print label1
    remain22 = [x for x in remain23 if x != label0[2]]
    #print remain22
    
    # =====22 Assign one point to label 1 ===== #
    
    #for i in remain22:
        #print label1[0], i, D[label1[0]][i]
    #    print label1[1], i, D[label1[1]][i]
    
    dist22_0 = [D[label1[0]][x] for x in remain22]
    dist22_1 = [D[label1[1]][x] for x in remain22]
    #print min(dist22_0), min(dist22_1)
    if min(dist22_0) <= min(dist22_1):
        dist22 = dist22_0
    if min(dist22_1) <= min(dist22_1):
        dist22 = dist22_1
    
    index22 = dist22.index(min(dist22))
    
    label1[2] = remain22[index22]
    
    #print label1
    remain21 = [x for x in remain22 if x != label1[2]]
    #print remain21, len(remain21)
    
    # =====21 Assign one point to label0 ===== #
    
    #for i in remain21:
        #print label0[0], i, D[label0[0]][i]
        #print label0[1], i, D[label0[1]][i]
    #    print label0[2], i, D[label0[2]][i]
    dist21_0 = [D[label0[0]][x] for x in remain21]
    dist21_1 = [D[label0[1]][x] for x in remain21]
    dist21_2 = [D[label0[2]][x] for x in remain21]
    
    #print min(dist21_0), min(dist21_1), min(dist21_2)
    
    if min(dist21_0) <= min(dist21_1) and min(dist21_0) <= min(dist21_2):
        dist21 = dist21_0
    if min(dist21_1) <= min(dist21_0) and min(dist21_1) <= min(dist21_2):
        dist21 = dist21_1
    if min(dist21_2) <= min(dist21_0) and min(dist21_2) <= min(dist21_1):
        dist21 = dist21_2
    
    index21 = dist21.index(min(dist21))
    
    label0[3] =  remain21[index21]
    
    #print label0
    
    remain20 = [x for x in remain21 if x != label0[3]]
    #print remain20, len(remain20)
    
    
    # =====20 Assign one point to label1 ===== #
    
    #print label1
    
    dist20_0 = [D[label1[0]][x] for x in remain20]
    dist20_1 = [D[label1[1]][x] for x in remain20]
    dist20_2 = [D[label1[2]][x] for x in remain20]
    
    #print min(dist20_0), min(dist20_1), min(dist20_2)
    
    min20 = [min(dist20_0), min(dist20_1), min(dist20_2)]
    
    min20_index =  min20.index(min(min20))
    
    dist20 = [D[label1[min20_index]][x] for x in remain20]
    
    index20 = dist20.index(min(dist20))
    
    label1[3] = remain20[index20]
    
    
    #for i in remain20:
    #    print label1[min20_index], i , D[label1[min20_index]][i]
    #print "label 0:", label0
    #print "label 1:", label1
    
    remain19 = [x for x in remain20 if x != label1[3]]
    
    #print remain19, len(remain19)
    
    # =====19 Assign one point to label0 ===== #
    
    dist19_0 = [D[label0[0]][x] for x in remain19]
    dist19_1 = [D[label0[1]][x] for x in remain19]
    dist19_2 = [D[label0[2]][x] for x in remain19]
    dist19_3 = [D[label0[3]][x] for x in remain19]
    
    min19 = [min(dist19_0), min(dist19_1), min(dist19_2), min(dist19_3)]
    
    min19_index =  min19.index(min(min19))
    
    dist19 = [D[label0[min19_index]][x] for x in remain19]
    
    index19 = dist19.index(min(dist19))
    
    label0[4] = remain19[index19]
    
    remain18 = [x for x in remain19 if x != label0[4]]
    
    #print label0
    #print label1
    #print remain18, len(remain18)
    
    # =====18 Assign one point to label1 ===== #
    
    dist18_0 = [D[label1[0]][x] for x in remain18]
    dist18_1 = [D[label1[1]][x] for x in remain18]
    dist18_2 = [D[label1[2]][x] for x in remain18]
    dist18_3 = [D[label1[3]][x] for x in remain18]
    
    min18 = [min(dist18_0), min(dist18_1), min(dist18_2), min(dist18_3)]
    
    min18_index =  min18.index(min(min18))
    
    dist18 = [D[label1[min18_index]][x] for x in remain18]
    
    index18 = dist18.index(min(dist18))
    
    label1[4] = remain18[index18]
    
    remain17 = [x for x in remain18 if x != label1[4]]
    
    #print label0
    #print label1
    #print remain17, len(remain17)
    
    # =====17 Assign one point to label0 ===== #
    
    dist17_0 = [D[label0[0]][x] for x in remain17]
    dist17_1 = [D[label0[1]][x] for x in remain17]
    dist17_2 = [D[label0[2]][x] for x in remain17]
    dist17_3 = [D[label0[3]][x] for x in remain17]
    dist17_4 = [D[label0[4]][x] for x in remain17]
    
    min17 = [min(dist17_0), min(dist17_1), min(dist17_2), min(dist17_3),min(dist17_4)]
    
    min17_index =  min17.index(min(min17))
    
    dist17 = [D[label0[min17_index]][x] for x in remain17]
    
    index17 = dist17.index(min(dist17))
    
    label0[5] = remain17[index17]
    
    remain16 = [x for x in remain17 if x != label0[5]]
    
    #print label0
    #print label1
    #print remain16, len(remain16)
    
    
    # =====16 Assign one point to label1 ===== #
    
    dist16_0 = [D[label1[0]][x] for x in remain16]
    dist16_1 = [D[label1[1]][x] for x in remain16]
    dist16_2 = [D[label1[2]][x] for x in remain16]
    dist16_3 = [D[label1[3]][x] for x in remain16]
    dist16_4 = [D[label1[4]][x] for x in remain16]
    
    min16 = [min(dist16_0), min(dist16_1), min(dist16_2), min(dist16_3),min(dist16_4)]
    
    min16_index =  min16.index(min(min16))
    
    dist16 = [D[label1[min16_index]][x] for x in remain16]
    
    index16 = dist16.index(min(dist16))
    
    label1[5] = remain16[index16]
    
    remain15 = [x for x in remain16 if x != label1[5]]
    
    #print label0
    #print label1
    #print remain15, len(remain15)
    
    # =====15 Assign one point to label0 ===== #
    
    dist15_0 = [D[label0[0]][x] for x in remain15]
    dist15_1 = [D[label0[1]][x] for x in remain15]
    dist15_2 = [D[label0[2]][x] for x in remain15]
    dist15_3 = [D[label0[3]][x] for x in remain15]
    dist15_4 = [D[label0[4]][x] for x in remain15]
    dist15_5 = [D[label0[5]][x] for x in remain15]
    
    min15 = [min(dist15_0), min(dist15_1), min(dist15_2), min(dist15_3),min(dist15_4),min(dist15_5)]
    
    min15_index =  min15.index(min(min15))
    
    dist15 = [D[label0[min15_index]][x] for x in remain15]
    
    index15 = dist15.index(min(dist15))
    
    label0[6] = remain15[index15]
    
    remain14 = [x for x in remain15 if x != label0[6]]
    
    #print label0
    #print label1
    #print remain14, len(remain14)
    
    # =====14 Assign one point to label1 ===== #
    
    dist14_0 = [D[label1[0]][x] for x in remain14]
    dist14_1 = [D[label1[1]][x] for x in remain14]
    dist14_2 = [D[label1[2]][x] for x in remain14]
    dist14_3 = [D[label1[3]][x] for x in remain14]
    dist14_4 = [D[label1[4]][x] for x in remain14]
    dist14_5 = [D[label1[5]][x] for x in remain14]
    
    min14 = [min(dist14_0), min(dist14_1), min(dist14_2), min(dist14_3),min(dist14_4),min(dist14_5)]
    
    min14_index =  min14.index(min(min14))
    
    dist14 = [D[label1[min14_index]][x] for x in remain14]
    
    index14 = dist14.index(min(dist14))
    
    label1[6] = remain14[index14]
    
    remain13 = [x for x in remain14 if x != label1[6]]
    
    #print label0
    #print label1
    #print remain13, len(remain13)
    
    # =====13 Assign one point to label0 ===== #
    
    dist13_0 = [D[label0[0]][x] for x in remain13]
    dist13_1 = [D[label0[1]][x] for x in remain13]
    dist13_2 = [D[label0[2]][x] for x in remain13]
    dist13_3 = [D[label0[3]][x] for x in remain13]
    dist13_4 = [D[label0[4]][x] for x in remain13]
    dist13_5 = [D[label0[5]][x] for x in remain13]
    dist13_6 = [D[label0[6]][x] for x in remain13]
    
    min13 = [min(dist13_0), min(dist13_1), min(dist13_2), min(dist13_3),min(dist13_4),min(dist13_5), min(dist13_6)]
    
    min13_index =  min13.index(min(min13))
    
    dist13 = [D[label0[min13_index]][x] for x in remain13]
    
    index13 = dist13.index(min(dist13))
    
    label0[7] = remain13[index13]
    
    remain12 = [x for x in remain13 if x != label0[7]]
    
    #print label0
    #print label1
    #print remain12, len(remain12)
    
    # =====12 Assign one point to label1 ===== #
    
    dist12_0 = [D[label1[0]][x] for x in remain12]
    dist12_1 = [D[label1[1]][x] for x in remain12]
    dist12_2 = [D[label1[2]][x] for x in remain12]
    dist12_3 = [D[label1[3]][x] for x in remain12]
    dist12_4 = [D[label1[4]][x] for x in remain12]
    dist12_5 = [D[label1[5]][x] for x in remain12]
    dist12_6 = [D[label1[6]][x] for x in remain12]
    
    min12 = [min(dist12_0), min(dist12_1), min(dist12_2), min(dist12_3),min(dist12_4),min(dist12_5), min(dist12_6)]
    
    min12_index =  min12.index(min(min12))
    
    dist12 = [D[label1[min12_index]][x] for x in remain12]
    
    index12 = dist12.index(min(dist12))
    
    label1[7] = remain12[index12]
    
    remain11 = [x for x in remain12 if x != label1[7]]
    
    #print label0
    #print label1
    #print remain11, len(remain11)
    
    # =====11 Assign one point to label0 ===== #
    
    dist11_0 = [D[label0[0]][x] for x in remain11]
    dist11_1 = [D[label0[1]][x] for x in remain11]
    dist11_2 = [D[label0[2]][x] for x in remain11]
    dist11_3 = [D[label0[3]][x] for x in remain11]
    dist11_4 = [D[label0[4]][x] for x in remain11]
    dist11_5 = [D[label0[5]][x] for x in remain11]
    dist11_6 = [D[label0[6]][x] for x in remain11]
    dist11_7 = [D[label0[7]][x] for x in remain11]
    
    min11 = [min(dist11_0), min(dist11_1), min(dist11_2), min(dist11_3),min(dist11_4),min(dist11_5), min(dist11_6), min(dist11_7)]
    
    min11_index =  min11.index(min(min11))
    
    dist11 = [D[label0[min11_index]][x] for x in remain11]
    
    index11 = dist11.index(min(dist11))
    
    label0[8] = remain11[index11]
    
    remain10 = [x for x in remain11 if x != label0[8]]
    
    #print label0
    #print label1
    #print remain10, len(remain10)
    
    # =====10 Assign one point to label1 ===== #
    
    dist10_0 = [D[label1[0]][x] for x in remain10]
    dist10_1 = [D[label1[1]][x] for x in remain10]
    dist10_2 = [D[label1[2]][x] for x in remain10]
    dist10_3 = [D[label1[3]][x] for x in remain10]
    dist10_4 = [D[label1[4]][x] for x in remain10]
    dist10_5 = [D[label1[5]][x] for x in remain10]
    dist10_6 = [D[label1[6]][x] for x in remain10]
    dist10_7 = [D[label1[7]][x] for x in remain10]
    
    min10 = [min(dist10_0), min(dist10_1), min(dist10_2), min(dist10_3),min(dist10_4),min(dist10_5), min(dist10_6), min(dist10_7)]
    
    min10_index =  min10.index(min(min10))
    
    dist10 = [D[label1[min10_index]][x] for x in remain10]
    
    index10 = dist10.index(min(dist10))
    
    label1[8] = remain10[index10]
    
    label2 = [x for x in remain10 if x != label1[8]]
    
    
    
    label = [None]*27
    
    for i in label0:
        label[i] = 0
    for i in label1:
        label[i] = 1 
    for i in label2:
        label[i] = 2
    
    
    
    dist9_0 = D[label0[0]][label2[0]]+D[label1[0]][label2[0]]
    dist9_1 = D[label0[0]][label2[1]]+D[label1[0]][label2[1]]
    dist9_2 = D[label0[0]][label2[2]]+D[label1[0]][label2[2]]
    dist9_3 = D[label0[0]][label2[3]]+D[label1[0]][label2[3]]
    dist9_4 = D[label0[0]][label2[4]]+D[label1[0]][label2[4]]
    dist9_5 = D[label0[0]][label2[5]]+D[label1[0]][label2[5]]
    dist9_6 = D[label0[0]][label2[6]]+D[label1[0]][label2[6]]
    dist9_7 = D[label0[0]][label2[7]]+D[label1[0]][label2[7]]
    dist9_8 = D[label0[0]][label2[8]]+D[label1[0]][label2[8]]
    
    li = [[label2[0], dist9_0], [label2[1], dist9_1], [label2[2], dist9_2], [label2[3], dist9_3], [label2[4], dist9_4], [label2[5], dist9_5], [label2[6], dist9_6], [label2[7], dist9_7], [label2[8], dist9_8]]
    
    list_label2 =  sorted(li,key=lambda l:l[1], reverse=True)
    
    label2 = [list_label2[0][0], list_label2[1][0], list_label2[2][0], list_label2[3][0], list_label2[4][0], list_label2[5][0], list_label2[6][0], list_label2[7][0], list_label2[8][0]]
    
    print label0
    print label1
    print label2
    print label
    
    y_label_path = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'y_train.npy' in name]
    
    y_label = np.load(y_label_path[0])
    
    print "Adjusted Rand Index: %.4f" %(metrics.adjusted_rand_score(y_label, label[:15]))
    print "Homogeneity Score: %.4f" %(metrics.homogeneity_score(y_label, label[:15]))
    return label, label0[0], label1[0], label2[0]

#from sklearn import svm
#from sklearn import cross_validation

num_filter, filter_size, whiten, zca_t1, zca_famd, pooling_size = getParameters()

print "Num_filter: %d" % num_filter
print "Filter_size: %d" % filter_size
if whiten ==0:
    print 'Unwhiten'
if whiten ==1:
    print 'Whiten with ZCA MR T1 %.3f and ZCA DT FA&MD %.3f' % (zca_t1, zca_famd)

#print "Adjusted Rand Index: %.4f" %(metrics.adjusted_rand_score(label, y_pred[:15]))
#print "Homogeneity Score: %.4f" %(metrics.homogeneity_score(label, y_pred[:15]))
num_subject = 27
if whiten == 1:
    feature_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "feature_representation" in name and 'pooling.'+str(pooling_size[0])+'.'+str(pooling_size[1])+'.'+str(pooling_size[2]) in name and 'number.'+str(num_filter) in root and 'zcat1.'+str(zca_t1) in root and 'zcafamd.'+str(zca_famd) in root]
if whiten == 0:
    feature_filepaths = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if "feature_representation" in name and 'pooling.'+str(pooling_size[0])+'.'+str(pooling_size[1])+'.'+str(pooling_size[2]) in name and 'unwhiten' in root]
feature_filepaths.sort()

print feature_filepaths[0]

feature_vectors_size = np.load(feature_filepaths[0]).shape
a =  (num_subject, feature_vectors_size[1], feature_vectors_size[2], feature_vectors_size[3], feature_vectors_size[4])
#print feature_vectors_size#, a[1]*a[2]*a[3]*a[4]
#print np.load(feature_filepaths[0])[1,:,:,:,:].max()
feature_vectors = np.zeros(a , dtype = np.float32)
#print feature_vectors.shape

for i in range(num_subject):
    feature_vectors[i, :, :, :, :] = np.load(feature_filepaths[i])[0,:,:,:,:]


reshaped_feature_vectors = np.reshape(feature_vectors, (num_subject, -1), order = 'C')

#print reshaped_feature_vectors.shape

for i in range(num_subject):
    if reshaped_feature_vectors[i,:].max() != feature_vectors[i,:,:,:,:].max():
        print "Reshape has some problems!"
        # print reshaped_feature_vectors[i,:].max(), feature_vectors[i,:,:,:,:].max()

print reshaped_feature_vectors.shape

initial_seed = np.zeros((3,reshaped_feature_vectors.shape[1]),dtype = np.float32)

Y = pdist(reshaped_feature_vectors, 'euclidean')

Z = squareform(Y)

pred_label, cluster0, cluster1, cluster2 = distance_searching_label(Z)

print cluster0, cluster1, cluster2

initial_seed[0,:] = reshaped_feature_vectors[cluster0,:]

initial_seed[1,:] = reshaped_feature_vectors[cluster1,:]

initial_seed[2,:] = reshaped_feature_vectors[cluster2,:]


distance = KMeans(n_clusters = 3, init = initial_seed).fit_transform(reshaped_feature_vectors)

print distance
