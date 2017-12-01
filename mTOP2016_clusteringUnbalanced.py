import os
import numpy as np
from sklearn.cluster import KMeans
#from sklearn import svm
#from sklearn import cross_validation
from sklearn import metrics

def main():
    whiten =1
    zca_t1 = 0.01
    zca_famd = 0.1
    num_filter = 128
    filter_size = 8
    pooling_size = [3,3,3]
    
    print "Num_filter: %d" % num_filter
    print "Filter_size: %d" % filter_size
    print "Pooling_size: ", pooling_size
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
    
    
    label_path = [os.path.join(root,name) for root, dirs, files in os.walk(os.getcwd()) for name in files if 'y_train.npy' in name]
    
    #print label_path
    label = np.load(label_path[0])
    
    feature_vectors_size = np.load(feature_filepaths[0]).shape
    a =  (num_subject, feature_vectors_size[1], feature_vectors_size[2], feature_vectors_size[3], feature_vectors_size[4])
    print feature_vectors_size#, a[1]*a[2]*a[3]*a[4]
    #print np.load(feature_filepaths[0])[1,:,:,:,:].max()
    feature_vectors = np.zeros(a , dtype = np.float32)
    #print feature_vectors.shape
    
    for i in range(num_subject):
        feature_vectors[i, :, :, :, :] = np.load(feature_filepaths[i])[0,:,:,:,:]
    
    
    reshaped_feature_vectors = np.reshape(feature_vectors, (num_subject, -1), order = 'C')
    
    print reshaped_feature_vectors.shape
    
    for i in range(num_subject):
        if reshaped_feature_vectors[i,:].max() != feature_vectors[i,:,:,:,:].max():
            print "Reshape has some problems!"
            # print reshaped_feature_vectors[i,:].max(), feature_vectors[i,:,:,:,:].max()
    
    print "Groundtruth:", label
    
    #y_pred = KMeans(n_clusters = 3,n_init = 10, max_iter = 300,precompute_distances = 'auto', n_jobs = -1 ).fit_predict(reshaped_feature_vectors)
    
    y_pred = KMeans(n_clusters = 3).fit_predict(reshaped_feature_vectors)
    
    print "Adjusted Rand Index: %.4f" %(metrics.adjusted_rand_score(label, y_pred[:15]))
    print "Homogeneity Score: %.4f" %(metrics.homogeneity_score(label, y_pred[:15]))
    print "Groundtruth:", label
    print "Prediction: ", y_pred
if __name__ == "__main__":
    main()