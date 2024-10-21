import numpy as np



class GetData:
    def __init__(self, directory, texp, size_latent_main, img_size, bands, bins, z_min, wbin, survey, nsample_dropout, phase):
        self.texp = texp
        self.size_latent_main = size_latent_main
        self.img_size = img_size
        self.bands = bands
        self.bins = bins
        self.z_min = z_min
        self.wbin = wbin
        self.survey = survey
        self.nsample_dropout = nsample_dropout
        self.phase = phase
        
        ##### User-defined catalogs and multi-band cutout images #####
        # The catalogs contain digit IDs for the training, test and validation samples, 
        # spectroscopic redshifts (zspec), and additional inputs including the galactic reddening E(B-V) and NIR magnitudes (for KiDS).
        # All images in a dataset are loaded at once. If not, the "load_img" method (see below) should be redefined and used instead.
            
        if survey == 1:  # SDSS
            catalog = np.load(directory + 'zphoto_catalog_SDSS.npz')
            imgs = np.load(directory + 'images_SDSS.npz')
        elif survey == 2:  # CFHTLS
            catalog = np.load(directory + 'zphoto_catalog_CFHTLS.npz')
            imgs = np.load(directory + 'images_CFHTLS.npz')
        elif survey == 3:  # KiDS
            catalog = np.load(directory + 'zphoto_catalog_KiDS+VIKING.npz')
            imgs = np.load(directory + 'images_KiDS.npz')

        id_train = catalog['id_train']
        id_test = catalog['id_test']
        id_validation = catalog['id_validation']
        zspec = catalog['zspec']
        if survey == 3:  # KiDS
            inputadd = np.stack([catalog['ebv'], catalog['mag_Z'], catalog['mag_Y'], catalog['mag_J'], catalog['mag_H'], catalog['mag_Ks']], 1)
        elif survey == 1 or survey == 2:  # SDSS, CFHTLS
            inputadd = np.expand_dims(catalog['ebv'], -1)
            
        self.id_train = id_train
        self.id_test = id_test
        self.id_validation = id_validation
        self.n_train = len(id_train)
        self.n_test = len(id_test)
        self.n_val = len(id_validation)
        self.zspec = zspec
        self.imgs = imgs
        self.inputadd = inputadd
        print ('Training,Test,Validation:', self.n_train, self.n_test, self.n_val)


    def load_img(self, id_):
        ##### User-defined image load method (for each galaxy) #####
        # Useful if all images for a dataset cannot be loaded at once
        img = np.array(list(self.imgs[id_]))
        return img
          

    def img_rescale(self, img):
        index_neg = img < 0
        index_pos = img > 0
        img[index_pos] = np.log(img[index_pos] + 1.0)
        img[index_neg] = -np.log(-img[index_neg] + 1.0)
        return img


    def img_reshape(self, img):
        mode = np.random.random()
        if mode < 0.25: img = np.rot90(img, 1)
        elif mode < 0.50: img = np.rot90(img, 2)
        elif mode < 0.75: img = np.rot90(img, 3)
        else: pass
            
        mode = np.random.random()
        if mode < 0.5: img = np.flip(img, 0)
        else: pass            
        return img
    

    def img_morph_aug(self, img):
        mode = np.random.random()
        if mode < 0.25: img = np.rot90(img, 1)
        elif mode < 0.50: img = np.rot90(img, 2)
        elif mode < 0.75: img = np.rot90(img, 3)
            
        if mode > 0.75: mode = 0
        else: mode = np.random.random()
        if mode < 0.5: img = np.flip(img, 0)
        return img


    def get_z_stats(self, zphoto_q, zspec_q, zprob_q, y_q):
        deltaz = (zphoto_q - zspec_q) / (1 + zspec_q)
        residual = np.mean(deltaz)
        sigma_mad = 1.4826 * np.median(abs(deltaz - np.median(deltaz)))
        
        if self.survey == 1: eta_th = 0.05  # SDSS
        elif self.survey == 2 or self.survey == 3: eta_th = 0.15  # CFHTLS, KiDS
        
        eta = len(deltaz[abs(deltaz) > eta_th]) / float(len(deltaz))
        crps = np.mean(np.sum((np.cumsum(zprob_q, 1) - np.cumsum(y_q, 1)) ** 2, 1)) * self.wbin                        
        return residual, sigma_mad, eta, crps


    def get_z_point_estimates(self, zlist, zprob_q, N_sample):
        zphoto_mean = np.sum(zprob_q * np.expand_dims(zlist, 0), 1)

        zphoto_max = np.zeros(N_sample)
        for i in range(N_sample):
            zphoto_max[i] = zlist[np.argmax(zprob_q[i])]

        zphoto_median = np.zeros(N_sample)
        for i in range(N_sample):
            zphoto_median[i] = zlist[np.argmin(abs(np.cumsum(zprob_q[i]) - 0.5))]
        return zphoto_mean, zphoto_max, zphoto_median
                

    def get_batch_data(self, id_):
        if self.phase == 0: 
            id_batch = self.id_validation#[:2000]
        elif self.phase == 1:
            id_batch = id_
           
        x_batch = np.zeros((len(id_batch), self.img_size, self.img_size, self.bands))  # rescaled images as inputs
        for i in range(len(id_batch)):
            img = self.load_img(id_batch[i])
            x_batch[i] = self.img_rescale(img)
         
        inputadd_batch = self.inputadd[id_batch]
        zspec_batch = self.zspec[id_batch]
        y_batch = np.zeros((len(id_batch), self.bins))  # one-hot labels
        for i in range(len(id_batch)):
            z_index = max(0, min(self.bins - 1, int((zspec_batch[i] - self.z_min) / self.wbin)))
            y_batch[i, z_index] = 1.0
        return x_batch, zspec_batch, y_batch, inputadd_batch
    
    
    def get_cost_z_stats(self, data_q, session, x, y, inputadd, x2, y2, inputadd2, p_set, cost_zp_set, latent_set):
        if self.phase == 0:  # training
            x_q, zspec_q, y_q, inputadd_q = data_q
            N_sample = len(zspec_q)
        elif self.phase == 1:  # testing
            id_all = np.concatenate([self.id_test, self.id_validation, self.id_train])
            N_sample = len(id_all)
            latent_q = np.zeros((N_sample, self.size_latent_main))
                
        batch = 512
        N_set = len(p_set)  # 1 or 2    # [p11, p12]
        zprob_q = np.zeros((N_set, N_sample, self.bins))
        cost_zp_q = np.zeros(N_set)
        
        for i in range(0, N_sample, batch):
            index_i = np.arange(i, min(i + batch, N_sample))
            if self.phase == 0:
                x_batch = x_q[index_i]                
                y_batch = y_q[index_i]
                inputadd_batch = inputadd_q[index_i]
            elif self.phase == 1:
                x_batch, zspec_batch, y_batch, inputadd_batch = self.get_batch_data(id_all[index_i])
                            
            x_batch2 = np.concatenate([x_batch[int(batch/2):], x_batch[:int(batch/2)]])
            inputadd_batch2 = np.concatenate([inputadd_batch[int(batch/2):], inputadd_batch[:int(batch/2)]])
            y_batch2 = np.concatenate([y_batch[int(batch/2):], y_batch[:int(batch/2)]])
            feed_dict = {x:x_batch, y:y_batch, inputadd:inputadd_batch, x2:x_batch2, y2:y_batch2, inputadd2:inputadd_batch2}
            output_batch = session.run(p_set + cost_zp_set + latent_set, feed_dict = feed_dict)
            for j in range(N_set):
                zprob_q[j][index_i] = output_batch[j]
                cost_zp_q[j] = cost_zp_q[j] + output_batch[N_set+j] * len(index_i)                
            if self.phase == 1:
                latent_q[index_i] = output_batch[-1]
        cost_zp_q = cost_zp_q / N_sample
                       
        zlist = (0.5 + np.arange(self.bins)) * self.wbin + self.z_min
        if self.phase == 0:
            residual = np.zeros(N_set)
            sigma_mad = np.zeros(N_set)
            eta = np.zeros(N_set)
            crps = np.zeros(N_set)
            for j in range(N_set):
                zphoto_mean = np.sum(zprob_q[j] * np.expand_dims(zlist, 0), 1)
                residual_j, sigma_mad_j, eta_j, crps_j = self.get_z_stats(zphoto_mean, zspec_q, zprob_q[j], y_q)
                residual[j] = residual_j
                sigma_mad[j] = sigma_mad_j
                eta[j] = eta_j
                crps[j] = crps_j     
            return cost_zp_q, residual, sigma_mad, eta, crps
        elif self.phase == 1:
            zphoto_mean = np.zeros((N_set, N_sample))
            zphoto_max = np.zeros((N_set, N_sample))
            zphoto_median = np.zeros((N_set, N_sample))
            for j in range(N_set):
                zphoto_mean_j, zphoto_max_j, zphoto_median_j = self.get_z_point_estimates(zlist, zprob_q[j], N_sample)
                zphoto_mean[j] = zphoto_mean_j
                zphoto_max[j] = zphoto_max_j
                zphoto_median[j] = zphoto_median_j
            return cost_zp_q, latent_q, zprob_q[0], zphoto_mean, zphoto_max, zphoto_median


    def get_mean_harmonic(self, p_set):
        n = len(p_set)
        s = np.sum(np.array([1 / p_set[i] for i in range(n)]), 0)
        ph = n / s
        ph[np.isnan(ph)] = 0
        ph = ph / np.sum(ph, 1, keepdims=True)
        return ph
    

    ### test with dropout
    def get_cost_z_stats_dropout(self, session, x, y, inputadd, x2, y2, inputadd2, p_set, cost_zp_set, latent_set):
        id_all = np.concatenate([self.id_test, self.id_validation, self.id_train])
        N_sample = len(id_all)
                
        batch = 512
        zprob_q = np.zeros((N_sample, self.bins))
        cost_zp_q = 0
        latent_q = 0
            
        for i in range(0, N_sample, batch):
            index_i = np.arange(i, min(i + batch, N_sample))
            x_batch, zspec_batch, y_batch, inputadd_batch = self.get_batch_data(id_all[index_i])
   
            x_batch2 = np.concatenate([x_batch[int(batch/2):], x_batch[:int(batch/2)]])
            inputadd_batch2 = np.concatenate([inputadd_batch[int(batch/2):], inputadd_batch[:int(batch/2)]])
            y_batch2 = np.concatenate([y_batch[int(batch/2):], y_batch[:int(batch/2)]])
            feed_dict = {x:x_batch, y:y_batch, inputadd:inputadd_batch, x2:x_batch2, y2:y_batch2, inputadd2:inputadd_batch2}

            zprob_n_batch = np.zeros((self.nsample_dropout, len(index_i), self.bins))
            for k in range(self.nsample_dropout):
                output_n_batch = session.run([p_set[0], cost_zp_set[0]], feed_dict = feed_dict)
                zprob_n_batch[k] = output_n_batch[0]
                cost_zp_q = cost_zp_q + output_n_batch[1] * len(index_i)
            zprob_q[index_i] = self.get_mean_harmonic(zprob_n_batch)
        cost_zp_q = cost_zp_q / N_sample / self.nsample_dropout
                                                
        zlist = (0.5 + np.arange(self.bins)) * self.wbin + self.z_min
        zphoto_mean, zphoto_max, zphoto_median = self.get_z_point_estimates(zlist, zprob_q, N_sample)   
        return cost_zp_q, latent_q, zprob_q, zphoto_mean, zphoto_max, zphoto_median

 
    def get_id_nonoverlap(self, id_all, id_pre, subbatch):
        id_select = np.random.choice(id_all, subbatch)
        for i in range(subbatch):
            while id_select[i] == id_pre[i]:
                id_select[i] = np.random.choice(id_all)
        return id_select
    
        
    def get_next_subbatch(self, subbatch):
        if self.texp == 0:
            id1_subbatch = np.random.choice(self.id_train, subbatch)
            id_list = [id1_subbatch]
            get_aug = [False]
        elif self.texp == 1:
            id1_subbatch = np.random.choice(self.id_train, subbatch)
            id2_subbatch = self.get_id_nonoverlap(self.id_train, id1_subbatch, subbatch) 
            id_list = [id1_subbatch, id2_subbatch]
            get_aug = [True, True]
            
        x_list = []
        y_list = []
        inputadd_list = []
        
        for k, id_subbatch in enumerate(id_list):        
            id_subbatch = np.array(id_subbatch)
            z = self.zspec[id_subbatch]
            inputadd = self.inputadd[id_subbatch]
            
            y = np.zeros((subbatch, self.bins))
            x = np.zeros((subbatch, self.img_size, self.img_size, self.bands))
            x_morph = np.zeros((subbatch, self.img_size, self.img_size, self.bands))
            for i in range(subbatch):
                img = self.load_img(id_subbatch[i])               
                img = self.img_reshape(img)
                x[i] = self.img_rescale(np.array(list(img)))
                z_index = max(0, min(self.bins - 1, int((z[i] - self.z_min) / self.wbin)))
                y[i, z_index] = 1.0
                
                if get_aug[k]:
                    img_morph = self.img_morph_aug(np.array(list(img)))
                    x_morph[i] = self.img_rescale(img_morph)

            x = [x, x_morph]
            x_list.append(x)
            y_list.append(y)
            inputadd_list.append(inputadd)
        return x_list, y_list, inputadd_list

      