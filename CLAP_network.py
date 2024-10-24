import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import numpy as np
        

 
def prelu(x):
    with tf.name_scope('PRELU'):
        _alpha = tf.get_variable('prelu', shape=x.get_shape()[-1], dtype = x.dtype, initializer=tf.constant_initializer(0.0))
    return tf.maximum(0.0, x) + _alpha * tf.minimum(0.0, x)



def conv2d(input, name, num_output_channels=None, kernel_size=3, kernel_size2=None, strides=[1,1,1,1], padding='SAME', uniform=True, square_kernel=True, add_bias=True, act='prelu', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if padding == 'SYMMETRIC':
            padding_size = int(kernel_size / 2)
            if square_kernel: padding_size2 = padding_size
            else: padding_size2 = int(kernel_size2 / 2)
            input = tf.pad(input, paddings=tf.constant([[0,0], [padding_size,padding_size], [padding_size2,padding_size2], [0,0]]), mode='SYMMETRIC')
            padding = 'VALID'
        if padding == 'REFLECT':
            padding_size = int(kernel_size / 2)
            if square_kernel: padding_size2 = padding_size
            else: padding_size2 = int(kernel_size2 / 2)
            input = tf.pad(input, paddings=tf.constant([[0,0], [padding_size,padding_size], [padding_size2,padding_size2], [0,0]]), mode='REFLECT')
            padding = 'VALID'            

        num_in_channels = input.get_shape()[-1].value
        if square_kernel: kernel_shape = [kernel_size, kernel_size, num_in_channels, num_output_channels]
        else: kernel_shape = [kernel_size, kernel_size2, num_in_channels, num_output_channels]
            
   #     weights = tf.get_variable('weights', shape=kernel_shape, initializer=tf.contrib.layers.xavier_initializer(uniform=uniform))
        weights = tf.get_variable('weights', shape=kernel_shape, initializer=tf.glorot_uniform_initializer())                 
        outputs = tf.nn.conv2d(input, weights, strides=strides, padding=padding)
        
        if add_bias:
            biases = tf.get_variable('biases', shape=[num_output_channels], initializer=tf.constant_initializer(0.1))
            outputs = tf.nn.bias_add(outputs, biases)
        
        if act == 'prelu': outputs = prelu(outputs)
        elif act == 'relu': outputs = tf.nn.relu(outputs)
        elif act == 'tanh': outputs = tf.nn.tanh(outputs)
        elif act == 'sigmoid': outputs = tf.sigmoid(outputs)
        elif act == 'leakyrelu': outputs = tf.nn.leaky_relu(outputs)
        elif act == None: pass
        return outputs      
    
    
    
def pool2d(input, kernel_size, stride, name, padding='SAME', use_avg=True, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        if use_avg: 
            return tf.nn.avg_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name = name)
        else: 
            return tf.nn.max_pool(input, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name = name)
        


def fully_connected(input, num_outputs, name, add_bias=True, expand_weights=False, act='relu', reuse=False):           
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        num_input_units = input.get_shape()[-1].value
        weights_shape = [num_input_units, num_outputs]
    #    weights = tf.get_variable('weights', shape=weights_shape, initializer=tf.contrib.layers.xavier_initializer())
        weights = tf.get_variable('weights', shape=weights_shape, initializer=tf.glorot_uniform_initializer())        
        if expand_weights:
            weights = tf.expand_dims(weights, 0)
        outputs = tf.matmul(input, weights)
        
        if add_bias:
            biases = tf.get_variable('biases', shape=[num_outputs], initializer=tf.constant_initializer(0.1))
            outputs = tf.nn.bias_add(outputs, biases)

        if act == 'prelu': outputs = prelu(outputs)
        elif act == 'relu': outputs = tf.nn.relu(outputs)
        elif act == 'tanh': outputs = tf.nn.tanh(outputs)
        elif act == 'sigmoid': outputs = tf.sigmoid(outputs)
        elif act == 'leakyrelu': outputs = tf.nn.leaky_relu(outputs)
        elif act == None: pass
        return outputs
    

    
def inception_Pasquet(input, nbS1, nbS2, name, output_name, without_kernel_5=False, act='prelu', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        s1_0 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, act=act, name=name+'S1_0')
        s2_0 = conv2d(input=s1_0, num_output_channels=nbS2, kernel_size=3, act=act, name=name+'S2_0')
        s1_2 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, act=act, name=name+'S1_2')
        pool0 = pool2d(input=s1_2, kernel_size=2, stride=1, name=name+'pool0', use_avg=True)
        if not(without_kernel_5):
            s1_1 = conv2d(input=input, num_output_channels=nbS1, kernel_size=1, act=act, name=name+'S1_1')
            s2_1 = conv2d(input=s1_1, num_output_channels=nbS2, kernel_size=5, act=act, name=name+'S2_1')
        s2_2 = conv2d(input=input, num_output_channels=nbS2, kernel_size=1, act=act, name=name+'S2_2')

        if not(without_kernel_5):
            outputs = tf.concat(values=[s2_2, s2_1, s2_0, pool0], name=output_name, axis=3)
        else:
            outputs = tf.concat(values=[s2_2, s2_0, pool0], name=output_name, axis=3)
    return outputs



def inception_Treyer(input, kernels, name, act='relu', reuse=False):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
            
        a0 = conv2d(input=input, num_output_channels=int(kernels*0.65), kernel_size=1, name='a0', act=act)
        a1 = conv2d(input=a0, num_output_channels=kernels, kernel_size=5, name='a1', act=act)
        b0 = conv2d(input=input, num_output_channels=int(kernels*0.65), kernel_size=1, name='b0', act=act)
        b1 = conv2d(input=b0, num_output_channels=kernels, kernel_size=3, name='b1', act=act)
        c0 = conv2d(input=input, num_output_channels=int(kernels*0.65), kernel_size=1, name='c0', act=act)
        c1 = pool2d(input=c0, kernel_size=2, name='c1', stride=1, use_avg=True)
        d1 = conv2d(input=input, num_output_channels=int(kernels*0.7), kernel_size=1, name='d1', act=act)
    return tf.concat([a1, b1, c1, d1], 3)



class Model:
    def __init__(self, texp, survey, img_size, bands, bins, size_latent_main, size_latent_ext, net, num_gmm, rate_dropout, nsample_dropout, name, reuse):
        self.texp = texp
        self.img_size = img_size
        self.bands = bands
        self.bins = bins
        self.size_latent_main = size_latent_main
        self.size_latent_ext = size_latent_ext
        self.net = net
        self.num_gmm = num_gmm
        self.rate_dropout = rate_dropout
        self.nsample_dropout = nsample_dropout
        self.name = name
        self.reuse = reuse
        
        if survey == 3:  # KiDS
            c_inputadd = 6  # E(B-V), mag_Z, mag_Y, mag_J, mag_H, mag_Ks
        elif survey == 1 or survey == 2:  # SDSS, CFHTLS
            c_inputadd = 1  # E(B-V)
                        
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')

        self.x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, bands], name='x')
        self.inputadd = tf.placeholder(tf.float32, shape=[None, c_inputadd], name='inputadd')
        self.y = tf.placeholder(tf.float32, shape=[None, bins], name='y')

        self.x2 = tf.placeholder(tf.float32, shape=[None, img_size, img_size, bands], name='x2')
        self.inputadd2 = tf.placeholder(tf.float32, shape=[None, c_inputadd], name='inputadd2')
        self.y2 = tf.placeholder(tf.float32, shape=[None, bins], name='y2')

        self.x_morph = tf.placeholder(tf.float32, shape=[None, img_size, img_size, bands], name='x_morph')
        self.x2_morph = tf.placeholder(tf.float32, shape=[None, img_size, img_size, bands], name='x2_morph')


    def encoder(self, x, inputadd, name, reuse):      
        if self.net == 0: act = 'prelu'   # InceptionNet_Pasquet
        elif self.net == 1: act = 'relu'  # InceptionNet_Treyer
        
        with tf.variable_scope(self.name + name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                
            if self.net == 0:  # InceptionNet_Pasquet             
                conv0 = conv2d(input=x, num_output_channels=64, kernel_size=5, name='conv0', act=act)
                conv0p = pool2d(input=conv0, kernel_size=2, stride=2, name='conv0p', use_avg=True)
                i0 = inception_Pasquet(conv0p, 48, 64, name='I0_', output_name='inception_Pasquet0', act=act)
                i1 = inception_Pasquet(i0, 64, 92, name='I1_', output_name='inception_Pasquet1', act=act)
                i1p = pool2d(input=i1, kernel_size=2, name='inception_Pasquet1p', stride=2, use_avg=True)
                i2 = inception_Pasquet(i1p, 92, 128, name='I2_', output_name='inception_Pasquet2', act=act)
                i3 = inception_Pasquet(i2, 92, 128, name='I3_', output_name='inception_Pasquet3', act=act)
                i3p = pool2d(input=i3, kernel_size=2, name='inception_Pasquet3p', stride=2, use_avg=True)
                i4 = inception_Pasquet(i3p, 92, 128, name='I4_', output_name='inception_Pasquet4', act=act, without_kernel_5=True)
                flat = tf.layers.Flatten()(i4)
                concat = tf.concat([flat, inputadd], 1)
                fc0 = fully_connected(input=concat, num_outputs=1024, name='fc0', act='relu')

            elif self.net == 1:  # InceptionNet_Treyer
                fc1e = fully_connected(input=inputadd, num_outputs=96, name='fc1e', act=None) 
                conv1 = conv2d(input=x, num_output_channels=96, kernel_size=5, name='conv1', act=None)                    
                conv1e = tf.expand_dims(tf.expand_dims(fc1e, 1), 1)
                print (inputadd, fc1e, conv1e)
                conv1 = tf.nn.relu(conv1 + conv1e)
                conv2 = conv2d(input=conv1, num_output_channels=96, kernel_size=3, name='conv2', act='tanh')
                pool2 = pool2d(input=conv2, kernel_size=2, stride=2, name='pool2', use_avg=True)
                inc1 = inception_Treyer(input=pool2, kernels=156, name='inc1', act=act)
                inc2 = inception_Treyer(input=inc1, kernels=156, name='inc2', act=act)
                inc2b = inception_Treyer(input=inc2, kernels=156, name='inc2b', act=act)
                pool3 = pool2d(input=inc2b, kernel_size=2, stride=2, name='pool3', use_avg=True)
                inc3 = inception_Treyer(input=pool3, kernels=156, name='inc3', act=act)
                inc3b = inception_Treyer(input=inc3, kernels=156, name='inc3b', act=act)
                pool4 = pool2d(input=inc3b, kernel_size=2, stride=2, name='pool4', use_avg=True)
                inc4 = inception_Treyer(input=pool4, kernels=156, name='inc4', act=act)
                conv5 = conv2d(input=inc4, num_output_channels=96, kernel_size=3, name='conv5', act=act, padding='VALID')
                conv6 = conv2d(input=conv5, num_output_channels=96, kernel_size=3, name='conv6', act=act, padding='VALID')
                conv7 = conv2d(input=conv6, num_output_channels=96, kernel_size=3, name='conv7', act=act, padding='VALID')
                pool7 = pool2d(input=conv7, kernel_size=2, stride=1, name='pool7', use_avg=True)
                flat = tf.layers.Flatten()(pool7)
                fc0 = fully_connected(input=flat, num_outputs=1024, name='fc0', act=act)                    
                        
            if self.rate_dropout > 0:
                fc0 = tf.nn.dropout(fc0, rate = self.rate_dropout)
            fc1a = fully_connected(input=fc0, num_outputs=self.size_latent_main, name='fc1a', act=None)                    
            if self.size_latent_ext == 0:
                return fc1a
            else:
                fc1b = fully_connected(input=fc0, num_outputs=self.size_latent_ext, name='fc1b', act=None)                    
                return tf.concat([fc1a, fc1b], 1)
                

    def decoder(self, latent, name, reuse):        
        act = 'leakyrelu'
        with tf.variable_scope(self.name + name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            fc_de = fully_connected(input=latent, num_outputs=1024, name='fc_de', act=act)
            fm1 = tf.reshape(fc_de, [-1, 8, 8, 16])
            fm2 = conv2d(input=fm1, num_output_channels=32, kernel_size=3, name='fm2', act=act)
            fm3 = conv2d(input=fm2, num_output_channels=32, kernel_size=3, name='fm3', act=act)
            fm4 = tf.image.resize_images(fm3, size=[16, 16], method=tf.image.ResizeMethod.BILINEAR)
            fm5 = conv2d(input=fm4, num_output_channels=32, kernel_size=3, name='fm5', act=act)
            fm6 = conv2d(input=fm5, num_output_channels=32, kernel_size=3, name='fm6', act=act)
            fm7 = tf.image.resize_images(fm6, size=[32, 32], method=tf.image.ResizeMethod.BILINEAR)
            fm8 = conv2d(input=fm7, num_output_channels=32, kernel_size=3, name='fm8', act=act)
            fm9 = conv2d(input=fm8, num_output_channels=32, kernel_size=3, name='fm9', act=act)
            fm10 = tf.image.resize_images(fm9, size=[64, 64], method=tf.image.ResizeMethod.BILINEAR)
            fm11 = conv2d(input=fm10, num_output_channels=32, kernel_size=3, name='fm11', act=act)
            fm12 = conv2d(input=fm11, num_output_channels=self.bands, kernel_size=3, name='fm12', act=None)
            return fm12
        
    
    def estimator(self, latent, name, reuse):        
        if self.num_gmm == 0:  # softmax output
            dim_output = self.bins
        else:  # GMM output
            dim_output = self.num_gmm * 3
            
        with tf.variable_scope(self.name + name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
       
            fc2 = fully_connected(input=latent[:, :self.size_latent_main], num_outputs=1024, name='fc2', act='relu')
            if self.rate_dropout > 0:
                fc2 = tf.nn.dropout(fc2, rate = self.rate_dropout)
            fc3 = fully_connected(input=fc2, num_outputs=dim_output, name='fc3', act=None)
            return fc3
                
        
    def get_mse(self, img, img_recon):
        return tf.reduce_mean(tf.pow(img - img_recon, 2))

    
    def get_p_ce(self, y, zlogits):
        p = tf.nn.softmax(zlogits)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=zlogits))
        return p, cost


    def get_p_gmm_crps(self, y, zlogits):
        weight = tf.nn.softmax(zlogits[:, :self.num_gmm])
        sigma = tf.exp(zlogits[:, self.num_gmm:2*self.num_gmm]) + 10**(-5)
        mean = zlogits[:, 2*self.num_gmm:]
        
        zcoords = tf.expand_dims(tf.constant(np.arange(self.bins) + 0.5, dtype=np.float32), 0) * tf.ones_like(y)
        p = tf.zeros_like(y)
        
        for i in range(self.num_gmm):
            p = p + weight[:, i:i+1] * tf.exp(-0.5 * tf.pow((zcoords - mean[:, i:i+1]) / sigma[:, i:i+1], 2)) / (np.sqrt(2*np.pi) * sigma[:, i:i+1])
        cost = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.cumsum(p, 1) - tf.cumsum(y, 1), 2), 1))
        return p, cost
    

    def get_zlogits_contrast(self, zlogits1, zlogits2):
        p1 = tf.stop_gradient(tf.nn.softmax(zlogits1))
        p2 = tf.stop_gradient(tf.nn.softmax(zlogits2))
        cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p1, logits=zlogits2))
        cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=p2, logits=zlogits1))
        return cost1 + cost2

    
    def get_outputs_single_pre(self, x, inputadd, use_2nd=False):
        latent = self.encoder(x, inputadd, name='encoder', reuse=use_2nd)
        x_recon = self.decoder(latent, name='decoder', reuse=use_2nd)
        zlogits = self.estimator(latent, name='estimator', reuse=use_2nd)
        return latent, x_recon, zlogits
        
    
    def get_outputs_single_cyc(self, x, y, inputadd, use_2nd=False):
        latent_cyc1, x_recon_cyc1, zlogits_cyc1 = self.get_outputs_single_pre(x, inputadd, use_2nd=use_2nd)
        latent_cyc2, x_recon_cyc2, zlogits_cyc2 = self.get_outputs_single_pre(x_recon_cyc1, inputadd, use_2nd=True)
        p1, cost_zp_single_cyc1 = self.get_p_ce(y, zlogits_cyc1)
        p2, cost_zp_single_cyc2 = self.get_p_ce(y, zlogits_cyc2)
        cost_recon_single_cyc1 = self.get_mse(x, x_recon_cyc1)
        cost_recon_single_cyc2 = self.get_mse(x, x_recon_cyc2)
        cost_zcontra = self.get_zlogits_contrast(zlogits_cyc1, zlogits_cyc2)
        cost_zp = cost_zp_single_cyc1 + cost_zp_single_cyc2
        cost_recon = cost_recon_single_cyc1 + cost_recon_single_cyc2
        return cost_zp, cost_recon, cost_zcontra, p1, p2, zlogits_cyc1, cost_zp_single_cyc1, cost_zp_single_cyc2, latent_cyc1, latent_cyc2
            
                
    def get_outputs_aug(self, x, x_morph, inputadd, latent, zlogits, use_2nd=False):
        latent_morphaug, x_recon_morphaug, zlogits_morphaug = self.get_outputs_single_pre(x_morph, inputadd, use_2nd=use_2nd)
        cost_recon_morphaug = self.get_mse(x_morph, x_recon_morphaug)
        cost_zcontra_morphaug = self.get_zlogits_contrast(zlogits, zlogits_morphaug)  
        return cost_recon_morphaug, cost_zcontra_morphaug, latent_morphaug
        

    def exp_minus_lmse(self, latent1_list, latent2_list):
        latent1_main = tf.concat(latent1_list, 0)[:, :self.size_latent_main]
        latent2_main = tf.concat(latent2_list, 0)[:, :self.size_latent_main]
        return tf.exp(-1 * tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.pow(latent1_main - latent2_main, 2), 1))))


    def get_outputs(self):
        if self.texp == 0:  # end-to-end baseline model, e.g., vanilla CNN
            latent = self.encoder(self.x, self.inputadd, name='encoder', reuse=False)
            zlogits = self.estimator(latent, name='estimator', reuse=False)
            if self.num_gmm == 0:  # softmax output
                p, cost_zp_single = self.get_p_ce(self.y, zlogits)
            else:  # GMM output
                p, cost_zp_single = self.get_p_gmm_crps(self.y, zlogits)
            return [p], [cost_zp_single], [latent[:, :self.size_latent_main]]
            
        elif self.texp == 1:  # supervised contrastive learning
            cost_zp1, cost_recon1, cost_zcontra1, p11, p12, zlogits1, cost_zp_single11, cost_zp_single12, latent1, latent1_ = self.get_outputs_single_cyc(self.x, self.y, self.inputadd, use_2nd=False)
            cost_zp2, cost_recon2, cost_zcontra2, _, _, zlogits2, _, _, latent2, latent2_ = self.get_outputs_single_cyc(self.x2, self.y2, self.inputadd2, use_2nd=True)
            cost_recon_morphaug1, cost_zcontra_morphaug1, latent_morph1 = self.get_outputs_aug(self.x, self.x_morph, self.inputadd, latent1, zlogits1, use_2nd=True)
            cost_recon_morphaug2, cost_zcontra_morphaug2, latent_morph2 = self.get_outputs_aug(self.x2, self.x2_morph, self.inputadd2, latent2, zlogits2, use_2nd=True)
            
            cost_zp = 0.5 * (cost_zp1 + cost_zp2)
            cost_recon = 0.5 * (cost_recon1 + cost_recon2 + cost_recon_morphaug1 + cost_recon_morphaug2)
            cost_zcontra = 0.5 * (cost_zcontra1 + cost_zcontra2 + cost_zcontra_morphaug1 + cost_zcontra_morphaug2)
                    
            lmse_p = self.exp_minus_lmse([latent1, latent1, latent2, latent2], [latent1_, latent_morph1, latent2_, latent_morph2])
            lmse_n = self.exp_minus_lmse([latent1], [latent2])
            cost_contra = -1 * tf.log(lmse_p / (lmse_p + lmse_n))
            print (cost_contra)
                
            p_set = [p11, p12]
            cost_zp_set = [cost_zp_single11, cost_zp_single12]
            latent_set = [latent1[:, :self.size_latent_main]]
            return cost_zp, cost_recon, cost_contra, cost_zcontra, p_set, cost_zp_set, latent_set
