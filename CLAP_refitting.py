import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time
import threading
import argparse



##### Settings #####


directory = './'

parser = argparse.ArgumentParser()
parser.add_argument('--survey', help='select a survey (1:SDSS, 2:CFHTLS, 3:KiDS)', type=int)
parser.add_argument('--recali_op', help='options for recalibration (0: no recalibration; 1: only using the training sample; 2: using both the training and the validation samples)', type=int)
parser.add_argument('--ne', help='# expriment', type=int)

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
    
    
cross_val = 1
repetition_per_ite = 1
use_cpu = False
num_threads = 4 + 1

learning_rate = 0.0001
lr_reduce_factor = 5.0
ite_val_print = 5000  # print validation results per # iterations during training    

iterations = 40000  # total number of iterations for training
ite_point_save = [20000, 30000, 40000]  # save model at # iterations during training

batch_train = 128



##### Load data #####


if survey == 1:  # SDSS
    # n_train = 393219
    # n_test = 103305
    # n_val = 20000
    z_min = 0.0
    z_max = 0.4
    bins = 180
    datalabel = 'knn+recal_SDSS_ne' + str(ne)
    InfoAdd = 'z04_bin' + str(bins)
    Survey = 'SDSS_'
    
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
    InfoAdd = 'z4_bin' + str(bins)
    Survey = 'CFHTLS_'
    
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
    InfoAdd = 'z3_bin' + str(bins)
    Survey = 'KiDS_'
    
    # output file from supervised contrastive learning that contains latent vectors
    fscl = np.load(directory + 'output_InceptionNetTreyer_ADAM_SCL_batch64_ite120000_n16e512_CoeffRecon100_KiDS_scratch_z3_bin800_cv1ne' + str(ne) + '_.npz')
    
    # user-defined catalog that contains spectroscopic redshifts (zspec)
    catalog = np.load(directory + 'zphoto_catalog_KiDS+VIKING.npz')
    

wbin = (z_max - z_min) / bins
zlist = (0.5 + np.arange(bins)) * wbin + z_min
InfoAdd = InfoAdd + '_cv' + str(cross_val) + 'ne' + str(ne)

# raw probability density estimates from adaptive KNN & recalibration
zprob_raw = np.load(directory + datalabel + '_zprob_raw_' + recali_label + '.npy')

# normalize and reset null probability densities
filt = np.sum(zprob_raw, 1) == 0
zprob_raw = zprob_raw / np.sum(zprob_raw, 1, keepdims=True)
zprob_raw[filt] = 1.0 / bins

id_latent_test = fscl['id_test']
id_latent_val = fscl['id_validation']
id_latent_test_ext = np.concatenate([id_latent_test, id_latent_val], 0)

id_catalog_test = catalog['id_test']
id_catalog_val = catalog['id_validation']

n_test = len(id_catalog_test)
n_val = len(id_catalog_val)
n_test_ext = n_test + n_val
           
latent = fscl['latent']
latent_test = latent[id_latent_test]
latent_val = latent[id_latent_val]

zspec = catalog['zspec']
zspec_test = zspec[id_catalog_test]
zspec_val = zspec[id_catalog_val]

zprob_raw_test = zprob_raw[:n_test]
zprob_raw_val = zprob_raw[n_test:]

print ('Test,Validation:', n_test, n_val)
print (latent_test.shape, latent_val.shape, zprob_raw_test.shape, zprob_raw_val.shape)    
    

# use the test sample as the "training" sample (input: latent_test; label: zprob_raw_test)
if survey == 1:  # for SDSS, use one half of the sample for training and the other half for inference
    n_train = int(n_test / 2)
    latent_train = latent_test[:n_train]
    zprob_raw_train = zprob_raw_test[:n_train]
elif survey == 2 or survey == 3:  # for CFHTLS and KiDS, use all the sample for training
    n_train = n_test
    latent_train = latent_test
    zprob_raw_train = zprob_raw_test



###### Model load / save paths #####


Algorithm = 'refitting_ADAM_' + recali_label

if repetition_per_ite != 1: Algorithm = Algorithm + 'rep' + str(repetition_per_ite) + '_'
Algorithm = Algorithm + 'batch' + str(batch_train) + '_'
Algorithm = Algorithm + 'ite' + str(iterations) + '_'

Algorithm = Algorithm + 'wasser+ce_'

Pretrain = 'scratch_'

fx = Algorithm + Survey + Pretrain + InfoAdd + '_'
    
model_savepath = directory + 'model_' + fx + '/'
output_savepath = directory + 'output_' + fx

fi = open(directory + 'f_' + fx + '.txt', 'a')
fi.write(fx + '\n\n')
fi.close()

print ('#####') 
print (fx)
print ('#####')



##### Network & Cost Function #####


def prelu(x):
    with tf.name_scope('PRELU'):
        _alpha = tf.get_variable('prelu', shape=x.get_shape()[-1], dtype = x.dtype, initializer=tf.constant_initializer(0.0))
    return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)

   
def fully_connected(input, num_outputs, name, act='relu', reuse=False):           
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        num_input_units = input.get_shape()[-1].value
        weights_shape = [num_input_units, num_outputs]
    #    weights = tf.get_variable('weights', shape=weights_shape, initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.get_variable('weights', shape=weights_shape, initializer=tf.glorot_uniform_initializer())
        biases = tf.get_variable('biases', shape=[num_outputs], initializer=tf.constant_initializer(0.1))
        
        outputs = tf.matmul(input, weights)
        outputs = tf.nn.bias_add(outputs, biases)

        if act == 'prelu': outputs = prelu(outputs)
        elif act == 'relu': outputs = tf.nn.relu(outputs)
        elif act == 'tanh': outputs = tf.nn.tanh(outputs)
        elif act == 'sigmoid': outputs = tf.sigmoid(outputs)
        elif act == 'leakyrelu': outputs = tf.nn.leaky_relu(outputs)
        elif act == None: pass
        return outputs
                
    
lr = tf.placeholder(tf.float32, shape=[], name='lr')
x = tf.placeholder(tf.float32, shape=[None, latent.shape[1]], name='x')
y = tf.placeholder(tf.float32, shape=[None, bins], name='y')

fc1 = fully_connected(input=x, num_outputs=1024, name='fc1', act='relu')
zlogits = fully_connected(input=fc1, num_outputs=bins, name='fc2', act=None)
p = tf.nn.softmax(zlogits)

cost1 = tf.reduce_mean(abs(tf.cumsum(y, 1) - tf.cumsum(p, 1)) * (z_max - z_min), 1)
cost2 = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=zlogits)
cost = 100 * tf.reduce_mean(cost1 * cost2)

    

##### Session, optimizer #####


if use_cpu: session_conf = tf.ConfigProto(device_count={'GPU':0})#log_device_placement=True
else:
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_conf)

tvars = tf.trainable_variables()
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
optimizer = optimizer.minimize(cost, var_list=tvars)      
session.run(tf.global_variables_initializer())
        


##### Functions for computing redshift estimates and statistics #####    
    

def get_z_stats(zphoto_q, zspec_q):
    deltaz = (zphoto_q - zspec_q) / (1 + zspec_q)
    residual = np.mean(deltaz)
    sigma_mad = 1.4826 * np.median(abs(deltaz - np.median(deltaz)))
    
    if survey == 1: eta_th = 0.05  # SDSS
    elif survey == 2 or survey == 3: eta_th = 0.15  # CFHTLS, KiDS
    
    eta = len(deltaz[abs(deltaz) > eta_th]) / float(len(deltaz))
    return residual, sigma_mad, eta


def get_z_point_estimates(zlist, zprob_q, N_sample):
    zphoto_mean = np.sum(zprob_q * np.expand_dims(zlist, 0), 1)

    zphoto_mode = np.zeros(N_sample)
    for i in range(N_sample):
        zphoto_mode[i] = zlist[np.argmax(zprob_q[i])]

    zphoto_median = np.zeros(N_sample)
    for i in range(N_sample):
        zphoto_median[i] = zlist[np.argmin(abs(np.cumsum(zprob_q[i]) - 0.5))]
    return zphoto_mean, zphoto_mode, zphoto_median      


def get_cost_z_stats(data_q, session, x, y, pred, output_z_prob_stats=False):
    x_q, zspec_q, y_q = data_q
    N_sample = len(zspec_q)
    batch = 1024
    zprob_q = np.zeros((N_sample, bins))
    cost_q = 0
    
    for i in range(0, N_sample, batch):
        index_i = np.arange(i, min(i + batch, N_sample))
        x_batch = x_q[index_i]
        y_batch = y_q[index_i]
        feed_dict = {x:x_batch, y:y_batch}

        zprob_q_i, cost_q_i = session.run(pred, feed_dict = feed_dict)
        zprob_q[index_i] = zprob_q_i
        cost_q = cost_q + cost_q_i * len(index_i)
    cost_q = cost_q / N_sample
    
    zphoto_mean, zphoto_mode, zphoto_median = get_z_point_estimates(zlist, zprob_q, N_sample)
    residual, sigma_mad, eta = get_z_stats(zphoto_mean, zspec_q)                      
    if output_z_prob_stats:
        return cost_q, zprob_q, zphoto_mean, zphoto_mode, zphoto_median, residual, sigma_mad, eta
    else:
        return cost_q, residual, sigma_mad, eta


##### Training #####


def Train(i, th, fi):
    global x_, y_
    global running
    
    if th == 0:
        feed_dict = {x:x_, y:y_, lr:learning_rate}
            
        if i == 0 or (i + 1) % ite_val_print == 0:
            ss = 'iteration:' + str(i+1) + ' lr:' + str(learning_rate) + ' mini-batch:' + str(batch_train) + ' time:' + str((time.time() - start) / 60) + ' minutes' + '\n'
            print (ss)
            fi = open(directory + 'f_' + fx + '.txt', 'a')
            fi.write(ss)

            cost_train = session.run(cost, feed_dict = feed_dict)
            print ('cost_training:', cost_train)
            fi.write('cost_training:' + str(cost_train) + '\n')

            outputs_val = get_cost_z_stats([latent_val, zspec_val, zprob_raw_val], session, x, y, [p, cost])
            print ('outputs_val:', outputs_val)
            fi.write('outputs_val:' + str(outputs_val) + '\n\n')
            fi.close()
            
        for t in range(repetition_per_ite):
            session.run(optimizer, feed_dict = feed_dict)
        running = 0
        
    else:
        def read_data(j):
            index_j = np.arange((j-1)*int(batch_train/(num_threads-1)), j*int(batch_train/(num_threads-1)))
            subbatch = len(index_j)
            obj_sample = np.random.choice(n_train, subbatch)

            while True:
                if running == 0: break
            
            x_[index_j] = latent_train[obj_sample]
            y_[index_j] = zprob_raw_train[obj_sample]
                                    
        for j in range(1, num_threads):
            if th == j:                
                read_data(j)


obj_sample = np.random.choice(n_train, batch_train)
x_ = latent_train[obj_sample]
y_ = zprob_raw_train[obj_sample]
        
start = time.time()
print ('Start training...')
       
for i in range(iterations):
    running = 1
        
    threads = []
    for th in range(num_threads):
        t = threading.Thread(target = Train, args = (i, th, fi))
        threads.append(t)
    for th in range(num_threads):
        threads[th].start()
    for th in range(num_threads):
        threads[th].join()    
        
    if (i + 1) in ite_point_save:
        saver = tf.train.Saver()
        saver.save(session, model_savepath, i)
            


##### Saving #####


cost_q, zprob_q, zphoto_mean, zphoto_mode, zphoto_median, residual, sigma_mad, eta = get_cost_z_stats([latent_test, zspec_test, zprob_raw_test], session, x, y, [p, cost], output_z_prob_stats=True)
np.savez(output_savepath, zphoto_mean=zphoto_mean, zphoto_mode=zphoto_mode, zphoto_median=zphoto_median, cost=cost_q, zprob=zprob_q)

print ('test statistics:', residual, sigma_mad, eta)
print (fx)
