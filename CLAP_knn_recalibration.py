import numpy as np
import time
import argparse



##### Settings #####


directory = './'

parser = argparse.ArgumentParser()
parser.add_argument('--survey', help='select a survey (1:SDSS, 2:CFHTLS, 3:KiDS)', type=int)
parser.add_argument('--recali_op', help='options for recalibration (0: no recalibration; 1: only using the training sample; 2: using both the training and the validation samples)', type=int)
parser.add_argument('--ne', help='# experiment', type=int)

args = parser.parse_args()
survey = args.survey
recali_op = args.recali_op
ne = args.ne

if recali_op == 0:
    recali_label = 'NoRecali_'
elif recali_op == 1:
    recali_label = 'RecaliTrain_'
elif recali_op == 2:
    recali_label = 'RecaliTrain+Val_'
    
# grid of k values
klist = list((1 + np.arange(40)) * 5) + list((1 + np.arange(40)) * 10 + 200) + list((1 + np.arange(20)) * 20 + 600) + list((1 + np.arange(20)) * 50 + 1000)
n_k = len(klist)
kmax = 2000

bins_pit_distri = 100  # use 100 bins to express a PIT distribution



##### Load data #####


if survey == 1:  # SDSS
    # n_train = 393219
    # n_test = 103305
    # n_val = 20000
    z_min = 0.0
    z_max = 0.4
    bins = 180
    datalabel = 'knn+recal_SDSS_ne' + str(ne)
    
    # output file from supervised contrastive learning that contains latent vectors
    fscl = np.load(directory + 'output_InceptionNetTreyer_ADAM_SCL_batch64_ite150000_n16e512_CoeffRecon100_SDSS_scratch_z04_bin180_cv1ne' + str(ne) + '_.npz')
    
    # user-defined catalog that contains spectroscopic redshifts (zspec)
    catalog = np.load(directory + 'zphoto_catalog_SDSS.npz')
    
    
elif survey == 2:  # CFHTLS
    # n_train = 100000
    # n_test = 20000
    # n_val = 14759
    z_min = 0.0
    z_max = 4.0
    bins = 1000
    datalabel = 'knn+recal_CFHTLS_ne' + str(ne)
    
    # output file from supervised contrastive learning that contains latent vectors
    fscl = np.load(directory + 'output_InceptionNetTreyer_ADAM_SCL_batch64_ite120000_n16e512_CoeffRecon1_CFHTLS_scratch_z4_bin1000_cv1ne' + str(ne) + '_.npz')
    
    # user-defined catalog that contains spectroscopic redshifts (zspec)
    catalog = np.load(directory + 'zphoto_catalog_CFHTLS.npz')
    

elif survey == 3:  # KiDS
    # n_train = 100000
    # n_test = 20000
    # n_val = 14147
    z_min = 0.0
    z_max = 3.0
    bins = 800
    datalabel = 'knn+recal_KiDS_ne' + str(ne)
    
    # output file from supervised contrastive learning that contains latent vectors
    fscl = np.load(directory + 'output_InceptionNetTreyer_ADAM_SCL_batch64_ite120000_n16e512_CoeffRecon100_KiDS_scratch_z3_bin800_cv1ne' + str(ne) + '_.npz')
    
    # user-defined catalog that contains spectroscopic redshifts (zspec)
    catalog = np.load(directory + 'zphoto_catalog_KiDS+VIKING.npz')
    

id_latent_train = fscl['id_train']
id_latent_test = fscl['id_test']
id_latent_val = fscl['id_validation']
id_latent_test_ext = np.concatenate([id_latent_test, id_latent_val], 0)

id_catalog_train = catalog['id_train']
id_catalog_test = catalog['id_test']
id_catalog_val = catalog['id_validation']

if survey == 2:  # only use high-quality spectroscopic redshifts for CFHTLS
    id_latent_train = id_latent_train[:100000]
    id_catalog_train = id_catalog_train[:100000]


n_train = len(id_catalog_train)
n_test = len(id_catalog_test)
n_val = len(id_catalog_val)
n_all = n_train + n_test + n_val
n_test_ext = n_test + n_val


latent = fscl['latent']
latent_train = latent[id_latent_train]      
latent_test = latent[id_latent_test]
latent_val = latent[id_latent_val]
    
zspec = catalog['zspec']
zspec_train = zspec[id_catalog_train]
zspec_val = zspec[id_catalog_val]

print (datalabel)
print ('Training,Test,Validation:', n_train, n_test, n_val)
print (latent_train.shape, latent_test.shape, latent_val.shape)



##### Adaptive KNN #####


get_id_knn = 1
# get the IDs of k nearest neighbors (from the training sample) for all instances in the test, validation, training samples
try:
    f = np.load(directory + datalabel + '_idknn_.npy')
    get_id_knn = 0
    print ("not run 'get_id_knn'")
except:
    print ("run 'get_id_knn'")


compute_pit = 1
# compute the PIT values using the obtained k nearest neighbors (running over the k list) for each instance in the training sample
try:
    f = np.load(directory + datalabel + '_pitknn_train_.npy')
    compute_pit = 0
    print ("not run 'compute_pit'")
except:
    print ("run 'compute_pit'")


compute_pitdist = 1
# compute the PIT distributions using the obtained PIT values from the training sample (running over the k list) for each instance in the test and validation samples
try:
    f = np.load(directory + datalabel + '_pitdist_.npy')
    compute_pitdist = 0
    print ("not run 'compute_pitdist'")
except:
    print ("run 'compute_pitdist'")
        

compute_metrics = 1
# compute the distance metrics using the obtained PIT distributions and the flat distribution for each instance in the test and validation samples
try:
    f = np.load(directory + datalabel + '_pitdist_metrics_.npz')
    compute_metrics = 0
    print ("not run 'compute_metrics'")
except:
    print ("run 'compute_metrics'")



##### Recalibration #####


get_nearest_val = 1
# get the nearest neighbor from the validation sample for each instance in the training sample
# to be used for recalibration that involves the validation sample ("recali_op == 2")
try:
    f = np.load(directory + datalabel + '_nearest_val_for_train_.npy')
    get_nearest_val = 0
    print ("not run 'get_nearest_val'")
except:
    print ("run 'get_nearest_val'")
    

compute_pit_with_nval = 1
# same as "compute_pit" but using the spec-z of the nearest neighbor from the validation sample for each instance in the training sample
# to be used for recalibration that involves the validation sample ("recali_op == 2")
try:
    f = np.load(directory + datalabel + '_pitknn_train_with_nval_.npy')
    compute_pit_with_nval = 0
    print ("not run 'compute_pit_with_nval'")
except:
    print ("run 'compute_pit_with_nval'")
    

generate_raw_zprob = 1
# generate raw photo-z probability density estimates using all the results obtained above for the test and validation samples ("recali_op" controls whether to apply recalibration)
try:
    f = np.load(directory + datalabel + '_zprob_raw_' + recali_label + '.npy')
    generate_raw_zprob = 0
    print ("not run 'generate_raw_zprob'", recali_label)
except:
    print ("run 'generate_raw_zprob'", recali_label)

    

##### Start computing #####


if get_id_knn == 1:
    print ('get_id_knn:')
    id_knn = np.zeros((n_all, kmax))
        
    start = time.time()
    for i in range(n_all):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        distsq_i = np.mean((latent[i:i+1] - latent_train) ** 2, 1)       
        id_knn_i = np.argsort(distsq_i)[:kmax]
        id_knn[i] = id_knn_i       
    id_knn = np.cast['int32'](id_knn)
    np.save(directory + datalabel + '_idknn_', id_knn)


if compute_pit == 1:
    print ('compute_pit:')
    id_knn_train = np.load(directory + datalabel + '_idknn_.npy')[id_latent_train]
    pit_knn = np.zeros((n_train, n_k))
    
    start = time.time()
    for i in range(n_train):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        id_knn_i = id_knn_train[i]
        zi = zspec_train[i]
        for j in range(n_k):
            k = klist[j]
            zspec_i_k = zspec_train[id_knn_i[:k]]
            pit_knn[i, j] = (len(zspec_i_k[zspec_i_k < zi]) + 0.5 * len(zspec_i_k[zspec_i_k == zi])) / k
    np.save(directory + datalabel + '_pitknn_train_', pit_knn) 
    

if compute_pitdist == 1:
    print ('compute_pitdist:')
    id_knn_test_ext = np.load(directory + datalabel + '_idknn_.npy')[id_latent_test_ext]
    pit_knn_train = np.load(directory + datalabel + '_pitknn_train_.npy')
    pit_distri = np.zeros((n_test_ext, n_k, bins_pit_distri))
        
    start = time.time()
    for i in range(n_test_ext):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        pit_knn_i = pit_knn_train[id_knn_test_ext[i]]
        if i == 0: 
            print (pit_knn_i.shape)  # (kmax, n_k)
        for j in range(n_k):
            k = klist[j]
            pit_distri_ij = np.histogram(pit_knn_i[:k, j], bins_pit_distri, (0, 1))[0] / k
            pit_distri[i, j] = pit_distri_ij
    np.save(directory + datalabel + '_pitdist_', pit_distri)


if compute_metrics == 1:
    print ('compute_metrics:')
    pit_distri = np.load(directory + datalabel + '_pitdist_.npy')
    flat_distri = np.ones(bins_pit_distri) / bins_pit_distri
    cdf_flat_distri = np.cumsum(flat_distri)
    wasser = np.zeros((n_test_ext, n_k))  # 1-Wasserstein distance
    tv = np.zeros((n_test_ext, n_k))  # total variation
    ce = np.zeros((n_test_ext, n_k))  # cross-entropy
    mse = np.zeros((n_test_ext, n_k))  # mean square error
        
    start = time.time()
    for i in range(n_test_ext):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        for j in range(n_k):
            k = klist[j]
            pit_distri_ij = pit_distri[i, j]     
            wasser[i, j] = np.sum(abs(np.cumsum(pit_distri_ij) - cdf_flat_distri)) / bins_pit_distri
            tv[i, j] = 0.5 * np.sum(abs(pit_distri_ij - flat_distri))
            ce[i, j] = -1 * np.sum(flat_distri * np.log(pit_distri_ij + 10**(-20)) + (1 - flat_distri) * np.log(1 - pit_distri_ij + 10**(-20)))
            mse[i, j] = np.sum((pit_distri_ij - flat_distri) ** 2)   
    np.savez(directory + datalabel + '_pitdist_metrics_', wasser=wasser, tv=tv, ce=ce, mse=mse)


if get_nearest_val == 1:
    print ('get_nearest_val:')
    id_nval = np.zeros(n_train)
    
    start = time.time()
    for i in range(n_train):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        distsq_i = np.mean((latent_train[i:i+1] - latent_val) ** 2, 1)
        id_nval[i] = np.argmin(distsq_i)
    id_nval = np.cast['int32'](id_nval)
    np.save(directory + datalabel + '_nearest_val_for_train_', id_nval)
    
    
if compute_pit_with_nval == 1:
    print ('compute_pit_with_nval:')
    id_nval = np.load(directory + datalabel + '_nearest_val_for_train_.npy')
    id_knn_train = np.load(directory + datalabel + '_idknn_.npy')[id_latent_train]
    pit_knn = np.zeros((n_train, n_k))
    
    start = time.time()
    for i in range(n_train):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        id_knn_i = id_knn_train[i]
        zi = zspec_val[id_nval[i]]
        for j in range(n_k):
            k = klist[j]
            zspec_i_k = zspec_train[id_knn_i[:k]]
            pit_knn[i, j] = (len(zspec_i_k[zspec_i_k < zi]) + 0.5 * len(zspec_i_k[zspec_i_k == zi])) / k
    np.save(directory + datalabel + '_pitknn_train_with_nval_', pit_knn)


if generate_raw_zprob == 1:
    print ('generate_raw_zprob:')
    id_knn_test_ext = np.load(directory + datalabel + '_idknn_.npy')[id_latent_test_ext]
    fpitdist = np.load(directory + datalabel + '_pitdist_metrics_.npz')
    distmetric = fpitdist['wasser']  # use 1-Wasserstein distance as the distance metric
    pit_knn_train = np.load(directory + datalabel + '_pitknn_train_.npy')
    if recali_op == 2:
        pit_knn_train2 = np.load(directory + datalabel + '_pitknn_train_with_nval_.npy')
    zprob = np.zeros((n_test_ext, bins))
    pit_grid = (np.arange(bins_pit_distri) + 0.5) / bins_pit_distri
    
    start = time.time()
    for i in range(n_test_ext):
        if i % 2000 == 0: print(i, str((time.time() - start) / 60) + ' minutes')
        j = np.argmin(distmetric[i])
        k = klist[j]
        zspec_i_k = zspec_train[id_knn_test_ext[i, :k]]
        zprob_i = np.histogram(zspec_i_k, bins, (z_min, z_max))[0]
        zprob_i = zprob_i / np.sum(zprob_i)  # initial probability density estimate
        if recali_op > 0: # apply recalibration
            zprob_i_cdf = np.cumsum(zprob_i)
            pit_knn_ki = pit_knn_train[id_knn_test_ext[i, :k]][:, j]
            pit_distri_ki = np.histogram(pit_knn_ki, bins_pit_distri, (0, 1))[0] / k
            if recali_op == 2:
                pit_knn_ki2 = pit_knn_train2[id_knn_test_ext[i, :k]][:, j]
                pit_distri_ki2 = np.histogram(pit_knn_ki2, bins_pit_distri, (0, 1))[0] / k
                pit_distri_ki = 0.5 * (pit_distri_ki + pit_distri_ki2)     
            para = np.polyfit(pit_grid, pit_distri_ki, 2)  # 2nd-order polynomial fit
            zprob_i = zprob_i * np.polyval(para, zprob_i_cdf)  # (recalibrated) raw probability density estimate
        zprob[i] = zprob_i
    zprob = zprob / np.sum(zprob, 1, keepdims=True)
    np.save(directory + datalabel + '_zprob_raw_' + recali_label, zprob)
