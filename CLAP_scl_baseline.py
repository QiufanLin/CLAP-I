import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time
from CLAP_network import *
from CLAP_data import *
import threading
import argparse



##### Settings #####


directory = './'

parser = argparse.ArgumentParser()
parser.add_argument('--ne', help='# experiment', type=int)
parser.add_argument('--phase', help='0: conducting training; 1: conducting inference', type=int)
parser.add_argument('--survey', help='select a survey (1:SDSS, 2:CFHTLS, 3:KiDS)', type=int)
#parser.add_argument('--bins', help='# of redshift bins in the density output', type=int)
parser.add_argument('--net', help='select a network', type=int)
parser.add_argument('--size_latent_main', help='size of the main latent vector that encodes redshift information', type=int)
parser.add_argument('--size_latent_ext', help='size of the extended latent vector that encodes other information', type=int, default=0)
parser.add_argument('--coeff_recon', help='coefficient of the image reconstruction loss term', type=int, default=1)
parser.add_argument('--batch_train', help='size of a mini-batch for training', type=int)
parser.add_argument('--texp', help='experiment type (0: end-to-end baseline model, e.g., vanilla CNN; 1: supervised contrastive learning)', type=int)
parser.add_argument('--num_gmm', help='# of Gaussian mixture components in the density output (0: using softmax instead)', type=int, default=0)
parser.add_argument('--rate_dropout', help='dropout rate used in the dropout layers (0: not using dropout)', type=float, default=0)

args = parser.parse_args()
ne = args.ne
phase = args.phase
survey = args.survey
#bins = args.bins
net = args.net
size_latent_main = args.size_latent_main
size_latent_ext = args.size_latent_ext
coeff_recon = args.coeff_recon
batch_train = args.batch_train
texp = args.texp
num_gmm = args.num_gmm  #5
rate_dropout = args.rate_dropout  #0.5

nsample_dropout = 100


cross_val = 1
repetition_per_ite = 1
use_cpu = False
num_threads = 4 + 1

learning_rate_ini = 0.0001
learning_rate_step = 60000  # reduce the learning rate at # iterations during training
lr_reduce_factor = 5.0
ite_val_print = 5000  # print validation results per # iterations during training    

if survey == 1:  # SDSS
    iterations = 150000  # total number of iterations for training
    ite_point_save = [i for i in range(120000, 150000+10000, 10000)]  # save model at # iterations during training
    img_size = 64
    bands = 5   # number of optical bands
    Survey = 'SDSS_'
    
elif survey == 2:  # CFHTLS
    iterations = 120000
    ite_point_save = [i for i in range(90000, 120000+10000, 10000)]
    img_size = 64
    bands = 5
    Survey = 'CFHTLS_'
    
elif survey == 3:  # KiDS
    iterations = 120000
    ite_point_save = [i for i in range(90000, 120000+10000, 10000)]
    img_size = 64
    bands = 4
    Survey = 'KiDS_'



###### Redshift range, bin width #####
    

if survey == 1:  # SDSS
    z_min = 0.0
    z_max = 0.4
    bins = 180  # number of redshift bins in the density output
    InfoAdd = 'z04_bin' + str(bins)

elif survey == 2:  # CFHTLS
    z_min = 0.0
    z_max = 4.0
    bins = 1000
    InfoAdd = 'z4_bin' + str(bins)

elif survey == 3:  # KiDS
    z_min = 0.0
    z_max = 3.0
    bins = 800
    InfoAdd = 'z3_bin' + str(bins)
    
wbin = (z_max - z_min) / bins
InfoAdd = InfoAdd + '_cv' + str(cross_val) + 'ne' + str(ne)    

    

###### Model load / save paths #####


if net == 0: Net = 'InceptionNetPasquet_'
elif net == 1: Net = 'InceptionNetTreyer_'
    
Algorithm = 'ADAM_'
if texp == 0:                  
    Algorithm = Algorithm + 'end2end_'
elif texp == 1:                  
    Algorithm = Algorithm + 'SCL_'

if repetition_per_ite != 1: Algorithm = Algorithm + 'rep' + str(repetition_per_ite) + '_'
Algorithm = Algorithm + 'batch' + str(batch_train) + '_'
Algorithm = Algorithm + 'ite' + str(iterations) + '_'
Algorithm = Algorithm + 'n' + str(size_latent_main) + 'e' + str(size_latent_ext) + '_'

if texp == 1:                  
    Algorithm = Algorithm + 'CoeffRecon' + str(coeff_recon) + '_'

if num_gmm > 0:
    Algorithm = Algorithm + 'GMM' + str(num_gmm) + '_'

if rate_dropout > 0:
    Algorithm = Algorithm + 'dropout' + str(rate_dropout).replace('.', '') + 'n' + str(nsample_dropout) + '_'
    
Pretrain = 'scratch_'  
    
fx = Net + Algorithm + Survey + Pretrain + InfoAdd + '_'

model_savepath = directory + 'model_' + fx + '/'
output_savepath = directory + 'output_' + fx

if phase == 0:
    fi = open(directory + 'f_' + fx + '.txt', 'a')
    fi.write(fx + '\n\n')
    fi.close()

elif phase == 1:
    model_load = directory + 'model_' + fx + '/'
    iterations = 0

print ('#####') 
print (fx)
print ('#####')
    
    

##### Load data #####


getdata = GetData(directory=directory, texp=texp, size_latent_main=size_latent_main, img_size=img_size, bands=bands,
                  bins=bins, z_min=z_min, wbin=wbin, survey=survey, nsample_dropout=nsample_dropout, phase=phase)

n_train = getdata.n_train
n_test = getdata.n_test
n_val = getdata.n_val
id_test = np.arange(n_test)
id_validation = np.arange(n_val) + n_test
id_train = np.arange(n_train) + n_test + n_val

if phase == 0:
    x_val, zspec_val, y_val, inputadd_val = getdata.get_batch_data(id_=[])



##### Network & Cost Function #####

   
model = Model(texp=texp, survey=survey, img_size=img_size, bands=bands, bins=bins, size_latent_main=size_latent_main, size_latent_ext=size_latent_ext,
              net=net, num_gmm=num_gmm, rate_dropout=rate_dropout, nsample_dropout=nsample_dropout, name='model_', reuse=False)

if texp == 0:
    p_set, cost_zp_set, latent_set = model.get_outputs()
    cost = cost_zp_set[0]
    
elif texp == 1:
    cost_zp, cost_recon, cost_contra, cost_zcontra, p_set, cost_zp_set, latent_set = model.get_outputs()
    cost_recon = coeff_recon * cost_recon
    cost = cost_zp + cost_recon + cost_contra + cost_zcontra
        
lr = model.lr

x = model.x
inputadd = model.inputadd
y = model.y
    
x2 = model.x2
inputadd2 = model.inputadd2
y2 = model.y2

x_morph = model.x_morph
x2_morph = model.x2_morph



##### Session, saver / optimizer #####


if use_cpu: session_conf = tf.ConfigProto(device_count={'GPU':0})#log_device_placement=True
else:
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True
session = tf.InteractiveSession(config=session_conf)

if phase == 0:
    tvars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)         
    optimizer = optimizer.minimize(cost, var_list=tvars)
    session.run(tf.global_variables_initializer())
    
elif phase == 1:
    tvars = tf.trainable_variables()
    if texp == 0:
        tvars = [var for var in tvars if ('decoder' not in var.name)]
    saver = tf.train.Saver(var_list=tvars)
    saver.restore(session, tf.train.latest_checkpoint(model_load))



##### Training #####


batch_train_ini = batch_train
if texp == 1:
    batch_train = int(batch_train/2)
# For supervised contrastive learning, two different sets of instances will be loaded as contrastive pairs at a time in a mini-batch, thus "batch_train" halves.

    
def Train(i, th):
    global x_, y_, inputadd_, x2_, y2_, inputadd2_, x_morph_, x2_morph_
    global running
    
    if th == 0:
        feed_dict = {x:x_, y:y_, inputadd:inputadd_, lr:learning_rate}
        if texp == 1:
            feed_dict.update({x2:x2_, y2:y2_, inputadd2:inputadd2_, x_morph:x_morph_, x2_morph:x2_morph_})
            
        if i == 0 or (i + 1) % ite_val_print == 0:
            ss = 'iteration:' + str(i+1) + ' lr:' + str(learning_rate) + ' mini-batch:' + str(batch_train_ini) + ' time:' + str((time.time() - start) / 60) + ' minutes' + '\n'
            print (ss)
            fi = open(directory + 'f_' + fx + '.txt', 'a')
            fi.write(ss)

            if texp == 0:
                cost_train = session.run(cost, feed_dict = feed_dict)
                print ('cost_training (zp_single):', cost_train)
                fi.write('cost_training (zp_single):' + str(cost_train) + '\n')
            elif texp == 1:
                cost_train1 = session.run(cost_zp_set, feed_dict = feed_dict)
                cost_train2 = session.run([cost_zp, cost_recon, cost_contra, cost_zcontra], feed_dict = feed_dict)
                print ('cost_training (zp_single):', cost_train1)
                print ('cost_training (zp, recon, contra, zcontra):', cost_train2)
                fi.write('cost_training (zp_single):' + str(cost_train1) + '\n')
                fi.write('cost_training (zp, recon, contra, zcontra):' + str(cost_train2) + '\n')

            outputs_val = getdata.get_cost_z_stats([x_val, zspec_val, y_val, inputadd_val], session, x, y, inputadd, x2, y2, inputadd2, p_set, cost_zp_set, latent_set)
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
            x_list, y_list, inputadd_list = getdata.get_next_subbatch(subbatch)

            while True:
                if running == 0: break
            
            x_[index_j] = x_list[0][0]
            y_[index_j] = y_list[0]
            inputadd_[index_j] = inputadd_list[0]
            if texp == 1:
                x_morph_[index_j] = x_list[0][1]
                x2_[index_j] = x_list[1][0]
                y2_[index_j] = y_list[1]
                inputadd2_[index_j] = inputadd_list[1]
                x2_morph_[index_j] = x_list[1][1]
                              
        for j in range(1, num_threads):
            if th == j:                
                read_data(j)
                   
                
if phase == 0:
    x_list, y_list, inputadd_list = getdata.get_next_subbatch(batch_train)
    
    x_ = x_list[0][0]
    y_ = y_list[0]
    inputadd_ = inputadd_list[0]
    if texp == 1:
        x_morph_ = x_list[0][1]
        x2_ = x_list[1][0]
        y2_ = y_list[1]
        inputadd2_ = inputadd_list[1]
        x2_morph_ = x_list[1][1]
        
    start = time.time()
    print ('Start training...')
       
    for i in range(iterations):
        if i == 0: 
            learning_rate = learning_rate_ini
        if i == learning_rate_step - 1: 
            learning_rate = learning_rate / lr_reduce_factor
        running = 1
        
        threads = []
        for th in range(num_threads):
            t = threading.Thread(target = Train, args = (i, th))
            threads.append(t)
        for th in range(num_threads):
            threads[th].start()
        for th in range(num_threads):
            threads[th].join()
        
        if (i + 1) in ite_point_save:
            saver = tf.train.Saver()
            saver.save(session, model_savepath, i)
            
            

##### Saving #####


if phase == 1:  # not use dropout
    if rate_dropout == 0: 
        cost_zp_q, latent_q, zprob_q, zphoto_mean, zphoto_mode, zphoto_median = getdata.get_cost_z_stats([], session, x, y, inputadd, x2, y2, inputadd2, p_set, cost_zp_set, latent_set)
    else:  # use dropout
        cost_zp_q, latent_q, zprob_q, zphoto_mean, zphoto_mode, zphoto_median = getdata.get_cost_z_stats_dropout(session, x, y, inputadd, x2, y2, inputadd2, p_set, cost_zp_set, latent_set)
   
    np.savez(output_savepath, n_train=n_train, n_test=n_test, n_val=n_val, id_train=id_train, id_test=id_test, id_validation=id_validation,
             zphoto_mean=zphoto_mean, zphoto_mode=zphoto_mode, zphoto_median=zphoto_median, cost_zp=cost_zp_q, latent=latent_q, zprob=zprob_q)
print (fx)
