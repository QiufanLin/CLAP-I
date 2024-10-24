import numpy as np
import argparse



##### Settings #####


directory = './'

parser = argparse.ArgumentParser()
parser.add_argument('--survey', help='select a survey (1:SDSS, 2:CFHTLS, 3:KiDS)', type=int)
parser.add_argument('--recali_op', help='options for recalibration (0: no recalibration; 1: only using the training sample; 2: using both the training and the validation samples)', type=int)

args = parser.parse_args()
survey = args.survey
recali_op = args.recali_op

if recali_op == 0:
    recali_label = 'NoRecali_'
elif recali_op == 1:
    recali_label = 'RecaliTrain_'
elif recali_op == 2:
    recali_label = 'RecaliTrain+Val_'
    
nelist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # user-specified (default: 10 realizations in an emsemble)
n_nelist = len(nelist)



##### Load data #####


if survey == 1:  # SDSS
    # n_train = 393219
    # n_test = 103305
    # n_val = 20000
    z_min = 0.0
    z_max = 0.4
    bins = 180
    InfoAdd = 'z04_bin' + str(bins)
    Survey = 'SDSS_'
    
    # output file that contains probability density estimates from supervised contrastive learning
    fscl_pre = directory + 'output_InceptionNetTreyer_ADAM_SCL_batch64_ite150000_n16e512_CoeffRecon100_SDSS_scratch_z04_bin180_cv1ne'
    
    # user-defined catalog that contains spectroscopic redshifts (zspec)
    catalog = np.load(directory + 'zphoto_catalog_SDSS.npz') 
    
    
elif survey == 2:  # CFHTLS
    # n_train = 100000
    # n_test = 20000
    # n_val = 14759
    z_min = 0.0
    z_max = 4.0
    bins = 1000
    InfoAdd = 'z4_bin' + str(bins)
    Survey = 'CFHTLS_'
    
    # output file that contains probability density estimates from supervised contrastive learning
    fscl_pre = directory + 'output_InceptionNetTreyer_ADAM_SCL_batch64_ite120000_n16e512_CoeffRecon1_CFHTLS_scratch_z4_bin1000_cv1ne'
    
    # user-defined catalog that contains spectroscopic redshifts (zspec)
    catalog = np.load(directory + 'zphoto_catalog_CFHTLS.npz')
    
    
elif survey == 3:  # KiDS
    # n_train = 100000
    # n_test = 20000
    # n_val = 14147
    z_min = 0.0
    z_max = 3.0
    bins = 800
    InfoAdd = 'z3_bin' + str(bins)
    Survey = 'KiDS_'
    
    # output file that contains probability density estimates from supervised contrastive learning
    fscl_pre = directory + 'output_InceptionNetTreyer_ADAM_SCL_batch64_ite120000_n16e512_CoeffRecon100_KiDS_scratch_z3_bin800_cv1ne'
    
    # user-defined catalog that contains spectroscopic redshifts (zspec)
    catalog = np.load(directory + 'zphoto_catalog_KiDS+VIKING.npz')
    

wbin = (z_max - z_min) / bins
zlist = (0.5 + np.arange(bins)) * wbin + z_min
InfoAdd = InfoAdd + '_cv1ne'

# output file that contains probability density estimates from refitting
frefit_pre = directory + 'output_refitting_ADAM_' + recali_label + 'batch128_ite40000_wasser+ce_' + Survey + 'scratch_' + InfoAdd

id_catalog_test = catalog['id_test']
zspec_test = catalog['zspec'][id_catalog_test]
n_test = len(id_catalog_test)

y_test = np.zeros((n_test, bins))
for i in range(n_test):
    z_index = max(0, min(bins - 1, int((zspec_test[i] - z_min) / wbin)))
    y_test[i, z_index] = 1.0
y_test_cdf = np.cumsum(y_test, 1)

print ('nelist:', nelist)
print ('Test:', n_test)



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


def get_moments(zlist, zprob_q):
    zphoto_mean = np.sum(zprob_q * np.expand_dims(zlist, 0), 1, keepdims=True)
    variance = np.sum(zprob_q * (np.expand_dims(zlist, 0) - zphoto_mean) ** 2, 1)
    skewness = np.sum(zprob_q * (np.expand_dims(zlist, 0) - zphoto_mean) ** 3, 1) / variance ** 1.5
    kurtosis = np.sum(zprob_q * (np.expand_dims(zlist, 0) - zphoto_mean) ** 4, 1) / variance ** 2
    return np.sqrt(variance), skewness, kurtosis
    

def get_pit_indiv(zprob_q, y_q_cdf, y_q):
    return np.sum(zprob_q * (1 - y_q_cdf + 0.5 * y_q), 1)


def get_wasser_indiv(wbin, zprob_q, y_q_cdf):
    return np.sum(abs(np.cumsum(zprob_q, 1) - y_q_cdf), 1) * wbin


def get_crps_indiv(wbin, zprob_q, y_q_cdf):
    return np.sum((np.cumsum(zprob_q, 1) - y_q_cdf) ** 2, 1) * wbin


def get_ce_indiv(zprob_q, y_q):
    return -1 * np.sum(y_q * np.log(zprob_q + 10**(-20)), 1)


def get_entropy_indiv(zprob_q):
    return -1 * np.sum(zprob_q * np.log(zprob_q + 10**(-20)), 1)


def get_mean_harmonic(p_set):
    n = len(p_set)
    s = np.sum(np.array([1 / p_set[i] for i in range(n)]), 0)
    ph = n / s
    ph[np.isnan(ph)] = 0
    ph = ph / np.sum(ph, 1, keepdims=True)
    return ph



##### Start computing and collecting results ##### 
    
  
from scipy.ndimage import gaussian_filter as gf

# use the Gaussian filter to smooth the output probability density estimates from refitting
alpha_gaussfilt = 0.05
print ('Gaussian filter:', alpha_gaussfilt)

resne_collect = {}
# results from refitting
resne_collect['zphoto_prob_density_refit'] = np.zeros((n_nelist+2, n_test, bins))  # photo-z probability density
resne_collect['zphoto_mean_refit'] = np.zeros((n_nelist+2, n_test))  # the probability-weighted mean redshift
resne_collect['zphoto_mode_refit'] = np.zeros((n_nelist+2, n_test))  # the redshift at the peak probability
resne_collect['zphoto_median_refit'] = np.zeros((n_nelist+2, n_test))  # the median redshift at which the cumulative probability is 0.5
resne_collect['pit_refit'] = np.zeros((n_nelist+2, n_test))  # probability integral transform
resne_collect['wasser_refit'] = np.zeros((n_nelist+2, n_test))  # 1-Wasserstein distance
resne_collect['crps_refit'] = np.zeros((n_nelist+2, n_test))  # continuous ranked probability score
resne_collect['ce_refit'] = np.zeros((n_nelist+2, n_test))  # cross-entropy with the one-hot label
resne_collect['entropy_refit'] = np.zeros((n_nelist+2, n_test))  # entropy
resne_collect['std_refit'] = np.zeros((n_nelist+2, n_test))  # standard deviation
resne_collect['skewness_refit'] = np.zeros((n_nelist+2, n_test))  # skewness
resne_collect['kurtosis_refit'] = np.zeros((n_nelist+2, n_test))  # kurtosis
zprob_set_refit = np.zeros((n_nelist, n_test, bins))  # tentative placeholder for individual realizations

# results from supervised contrastive learning
resne_collect['zphoto_prob_density_scl'] = np.zeros((n_nelist+2, n_test, bins))
resne_collect['zphoto_mean_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['zphoto_mode_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['zphoto_median_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['pit_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['wasser_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['crps_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['ce_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['entropy_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['std_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['skewness_scl'] = np.zeros((n_nelist+2, n_test))
resne_collect['kurtosis_scl'] = np.zeros((n_nelist+2, n_test))
zprob_set_scl = np.zeros((n_nelist, n_test, bins))

    
for j in range(n_nelist+2):
    # j < n_nelist: individual realizations (default: 10 realizations in an emsemble)
    # j == n_nelist: combined using arithmetic mean
    # j == n_nelist+1: combined using harmonic mean
    if j < n_nelist:
        ne = nelist[j]
        print (j, ne)
    
        # output file that contains probability density estimates from refitting
        frefit = np.load(frefit_pre + str(ne) + '_.npz')
        zprob_test_refit = frefit['zprob']
        
        # smooth "zprob_test_refit"
        std_ini, _, _, = get_moments(zlist, zprob_test_refit)
        sigma_j = alpha_gaussfilt * std_ini / wbin  # sigma = 0.05 * std of the unsmoothed estimate        
        for i in range(n_test):
            zprob_test_refit[i] = gf(zprob_test_refit[i], sigma=sigma_j[i], mode='constant')
        zprob_test_refit = zprob_test_refit / np.sum(zprob_test_refit, 1, keepdims=True)
        
        zprob_set_refit[j] = zprob_test_refit
        resne_collect['zphoto_prob_density_refit'][j] = zprob_test_refit

        # output file that contains probability density estimates from supervised contrastive learning
        fscl = np.load(fscl_pre + str(ne) + '_.npz')
        id_latent_test = fscl['id_test']
        zprob_test_scl = fscl['zprob'][id_latent_test]
        
        zprob_set_scl[j] = zprob_test_scl
        resne_collect['zphoto_prob_density_scl'][j] = zprob_test_scl
        
        print (zprob_test_refit.shape, zprob_test_scl.shape)
    
    elif j == n_nelist:
        print (j, 'arithmetic mean')
        zprob_test_refit = np.mean(zprob_set_refit, 0)
        zprob_test_scl = np.mean(zprob_set_scl, 0)
            
    else:
        print (j, 'harmonic mean')
        zprob_test_refit = get_mean_harmonic(zprob_set_refit)
        zprob_test_scl = get_mean_harmonic(zprob_set_scl)
        
            
    zphoto_mean, zphoto_mode, zphoto_median = get_z_point_estimates(zlist, zprob_test_refit, n_test)
    std, skewness, kurtosis = get_moments(zlist, zprob_test_refit)
    resne_collect['zphoto_mean_refit'][j] = zphoto_mean
    resne_collect['zphoto_mode_refit'][j] = zphoto_mode
    resne_collect['zphoto_median_refit'][j] = zphoto_median
    resne_collect['pit_refit'][j] = get_pit_indiv(zprob_test_refit, y_test_cdf, y_test)
    resne_collect['wasser_refit'][j] = get_wasser_indiv(wbin, zprob_test_refit, y_test_cdf)
    resne_collect['crps_refit'][j] = get_crps_indiv(wbin, zprob_test_refit, y_test_cdf)
    resne_collect['ce_refit'][j] = get_ce_indiv(zprob_test_refit, y_test)
    resne_collect['entropy_refit'][j] = get_entropy_indiv(zprob_test_refit)
    resne_collect['std_refit'][j] = std
    resne_collect['skewness_refit'][j] = skewness
    resne_collect['kurtosis_refit'][j] = kurtosis
    
    residual, sigma_mad, eta = get_z_stats(zphoto_mean, zspec_test)
    print ('test statistics (refit):', residual, sigma_mad, eta)
    
    zphoto_mean, zphoto_mode, zphoto_median = get_z_point_estimates(zlist, zprob_test_scl, n_test)
    std, skewness, kurtosis = get_moments(zlist, zprob_test_scl)
    resne_collect['zphoto_mean_scl'][j] = zphoto_mean
    resne_collect['zphoto_mode_scl'][j] = zphoto_mode
    resne_collect['zphoto_median_scl'][j] = zphoto_median
    resne_collect['pit_scl'][j] = get_pit_indiv(zprob_test_scl, y_test_cdf, y_test)
    resne_collect['wasser_scl'][j] = get_wasser_indiv(wbin, zprob_test_scl, y_test_cdf)
    resne_collect['crps_scl'][j] = get_crps_indiv(wbin, zprob_test_scl, y_test_cdf)
    resne_collect['ce_scl'][j] = get_ce_indiv(zprob_test_scl, y_test)
    resne_collect['entropy_scl'][j] = get_entropy_indiv(zprob_test_scl)
    resne_collect['std_scl'][j] = std
    resne_collect['skewness_scl'][j] = skewness
    resne_collect['kurtosis_scl'][j] = kurtosis

    residual, sigma_mad, eta = get_z_stats(zphoto_mean, zspec_test)
    print ('test statistics (scl):', residual, sigma_mad, eta)
    
savepath = directory + 'res_collect_refitting+SCL_gaussfilt' + str(alpha_gaussfilt).replace('.', 'p') + '_' + recali_label
np.savez(savepath, **resne_collect)
