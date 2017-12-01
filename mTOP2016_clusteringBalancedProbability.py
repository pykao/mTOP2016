import os
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import svm
#from sklearn import cross_validation
from sklearn import metrics
from itertools import combinations
np.set_printoptions(precision = 4)
def getParameters():
    num_filter = 512
    filter_size = 8
    whiten = 1
    zca_t1 = 0.01
    zca_famd = 0.1
    pooling_size = [3,3,3]
    return num_filter, filter_size, whiten, zca_t1, zca_famd, pooling_size

def balancelabel(distance):
    num = distance.shape[0]
    cluster = distance.shape[1]
    D = distance
    min_D = np.sum(np.amax(D, axis= 1))
    label0 = [None]*(num/cluster)
    label1 = [None]*(num/cluster)
    label2 = [None]*(num/cluster)

    if num == 15:
        for class1 in combinations(range(num),num/cluster):
            remaining1 = [i for i in xrange(num) if i not in class1]
            #print remaining1
            for class2 in combinations(remaining1, num/cluster):
                class3 = [i for i in xrange(num) if i not in class1 and i not in class2]
                tmp = D[class1[0]][0]+D[class1[1]][0]+D[class1[2]][0]+D[class1[3]][0]+D[class1[4]][0]+D[class2[0]][1]+D[class2[1]][1]+D[class2[2]][1]+D[class2[3]][1]+D[class2[4]][1]+D[class3[0]][2]+D[class3[1]][2]+D[class3[2]][2]+D[class3[3]][2]+D[class3[4]][2]
                if tmp <= min_D:
                    min_D = tmp

                    #print min_D
                    label0 = list(class1)
                    label1 = list(class2)
                    label2 = class3
                    #print label0, label1, label2
    if num == 12:
        for class1 in combinations(range(num),num/cluster):
            remaining1 = [i for i in xrange(num) if i not in class1]
            #print remaining1
            for class2 in combinations(remaining1, num/cluster):
                class3 = [i for i in xrange(num) if i not in class1 and i not in class2]
                tmp = D[class1[0]][0]+D[class1[1]][0]+D[class1[2]][0]+D[class1[3]][0]+D[class2[0]][1]+D[class2[1]][1]+D[class2[2]][1]+D[class2[3]][1]+D[class3[0]][2]+D[class3[1]][2]+D[class3[2]][2]+D[class3[3]][2]
                if tmp <= min_D:
                    min_D = tmp

                    #print min_D
                    label0 = list(class1)
                    label1 = list(class2)
                    label2 = class3
                    #print label0, label1, label2
    label = [None]*num

    for i in label0:
        label[i] = 0
    for i in label1:
        label[i] = 1
    for i in label2:
        label[i] = 2
    #print label0, label1, label2
    #print label
    return label

def probLabel(distance, label):
    D = distance

    prob = np.zeros((1,27), dtype = np.float32)
    for i in range(27):
        if label[i] == 0:
            prob[0,i] = (1/D[i][0])/((1/(D[i][0]))+(1/D[i][1])+(1/D[i][2]))
        if label[i] == 1:
            prob[0, i] = (1/D[i][1])/((1/(D[i][0]))+(1/D[i][1])+(1/D[i][2]))
        if label[i] == 2:
            prob[0, i] = (1/D[i][2])/(((1/D[i][0]))+(1/D[i][1])+(1/D[i][2]))
        #print i, prob[0,i]
    return prob


def main():
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

    label_path = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'y_train.npy' in name]

    print label_path
    label = np.load(label_path[0])

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
    print "Groundtruth:", list(label)

    #y_pred = KMeans(n_clusters = 3,n_init = 10, max_iter = 300,precompute_distances = 'auto', n_jobs = -1 ).fit_predict(reshaped_feature_vectors)

    #y_pred = KMeans(n_clusters = 3).fit_predict(reshaped_feature_vectors)
    distance = KMeans(n_clusters = 3).fit_transform(reshaped_feature_vectors)
    #print distance.shape
    label_15 = balancelabel(distance[:15,:])
    label_12 = balancelabel(distance[15:,:])
    #print len(label_15+label_12)
    prob = probLabel(distance, label_15+label_12)
    print "Adjusted Rand Index: %.4f" %(metrics.adjusted_rand_score(label, label_15))
    print "Homogeneity Score: %.4f" %(metrics.homogeneity_score(label, label_15))
    print "Prediction :", label_15+label_12
    print "Probility: "
    print prob
if __name__ == '__main__':
    main()
