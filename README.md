# CLAP. I. Resolving miscalibration for deep learning-based galaxy photometric redshift estimation

We present "the **C**ontrastive **L**earning and **A**daptive KNN for **P**hotometric Redshift (CLAP)", a novel method for obtaining well-calibrated galaxy photometric redshift (photo-z) probability density estimates. It empowers state-of-the-art image-based deep learning methods with the advantages of k-nearest neighbors (KNN), featured by high accuracy, high computational efficiency and a substantial improvement on the calibration of probability density estimates over conventional methods. This is the first paper in the CLAP series.

CLAP has been tested using data from three surveys: the Sloan Digital Sky Survey (SDSS), the Canada-France-Hawaii Telescope Legacy Survey (CFHTLS) and the Kilo-Degree Survey (KiDS).

A CLAP model is developed via a few procedures: supervised contrastive learning (SCL), adaptive KNN, recalibration and refitting. In order to reduce epistemic uncertainties, we combine probability density estimates from an ensemble of individual CLAP models/realizations via the harmonic mean. These procedures have to be run consecutively (detailed below).

The CLAP paper is available at http://arxiv.org/abs/2410.19390.

The photo-z catalogs produced by CLAP are available at https://zenodo.org/records/13954481.

The code is tested using:
- CPU: Intel(R) Xeon(R) Platinum 8260 CPU @ 2.40GHz
- GPU: Tesla V100 / T4
- Python 3.7.11
- TensorFlow 2.2.0 (developed with TensorFlow 1 but run with TensorFlow 2 by "tf.disable_v2_behavior()"; future versions will be migrated to TensorFlow 2)

## Supervised contrastive learning (SCL)

- Apply supervised contrastive learning by running "CLAP_scl_baseline.py", which requires "CLAP_data.py" and "CLAP_network.py".

- Users should prepare and load multi-band cutout images and catalogs for the three surveys (see "CLAP_data.py").

- Set "--texp=1" for supervised contrastive learning; "--phase=0" for training a model from scratch; "--net=1" for using the inception net from Treyer et al. (2024); "--survey=1" for using the SDSS dataset; "--survey=2" for using the CFHTLS dataset; "--survey=3" for using the KiDS dataset (including the VIKING NIR magnitudes). The argument "size_latent_main" specifies the number of dimensions of the latent vector that encodes redshift information (default: 16). The argument "size_latent_ext" specifies the number of dimensions of the latent vector that encodes other information (default: 512). The argument "coeff_recon" specifies the coefficient/weight of the image reconstruction loss term (default: 100 for SDSS and KiDS; 1 for CFHTLS). The argument "batch_train" specifies the size of a mini-batch in training (default: 64). Train individual models by setting "--ne=1", "--ne=2", …, "--ne=10" (10 realizations in an ensemble); different realizations have to be run separately. 

** Example with SDSS:
> python CLAP_scl_baseline.py --ne=1 --survey=1 --net=1 --size_latent_main=16 --size_latent_ext=512 --coeff_recon=100 --batch_train=64 --texp=1 --phase=0

** Example with CFHTLS:
> python CLAP_scl_baseline.py --ne=1 --survey=2 --net=1 --size_latent_main=16 --size_latent_ext=512 --coeff_recon=1 --batch_train=64 --texp=1 --phase=0

** Example with KiDS:
> python CLAP_scl_baseline.py --ne=1 --survey=3 --net=1 --size_latent_main=16 --size_latent_ext=512 --coeff_recon=100 --batch_train=64 --texp=1 --phase=0

- After training, rerun "CLAP_scl_baseline.py" by setting "--phase=1" for restoring the trained SCL model and producing inference results (saved in an output .npz file).

** Example with SDSS:
> python CLAP_scl_baseline.py --ne=1 --survey=1 --net=1 --size_latent_main=16 --size_latent_ext=512 --coeff_recon=100 --batch_train=64 --texp=1 --phase=1

## Adaptive KNN & Recalibration

- Apply adaptive KNN and recalibration by running "CLAP_knn_recalibration.py", producing raw probability density estimates (saved in an output file named "..._zprob_raw_....npy").

- The user-defined catalogs and the output .npz files from SCL are required.
 
- The arguments "ne" and "survey" are the same as in SCL, corresponding to certain SCL models/realizations. That is, before setting "--survey=1" and "--ne=1" for "CLAP_knn_recalibration.py", make sure an .npz file has been produced by SCL with "--survey=1" and "--ne=1". The argument "recali_op" specifies whether and how to apply recalibration after KNN (0: no recalibration; 1: only using the training sample for recalibration (default for KiDS); 2: using both the training and the validation samples for recalibration (default for SDSS and CFHTLS)).

** Example with SDSS:
> python CLAP_knn_recalibration.py --ne=1 --survey=1 --recali_op=2

## Refitting

- Apply refitting by running "CLAP_refitting.py", producing refit (but unsmoothed and uncombined) probability density estimates (save in an output .npz file).

- The user-defined catalogs, the output .npz files from SCL, and the output .npy files (containing raw probability density estimates) from KNN & recalibration are required.

- The arguments "ne", "survey" and "recali_op" are the same as in the previous procedures.

** Example with SDSS:
> python CLAP_refitting.py --ne=1 --survey=1 --recali_op=2

## Producing final results

- Run "CLAP_res_collect.py" to collect and combine the probability density estimates produced by multiple realizations from the previous procedures, including both SCL and refitting. The photo-z point estimates and summary statistics are also computed. The probability density estimates from refitting are smoothed by a Gaussian filter before being collected. The results from individual realizations and those produced by combination via the arithmetic mean and the harmonic mean are all provided, while those produced by combining the smoothed refit probability density estimates via the harmonic mean are regarded as the final results from the default implementation of CLAP.

- The user-defined catalogs, the output .npz files from SCL, and the output .npz files (containing refit probability density estimates) from refitting are required.

- The arguments "survey" and "recali_op" are the same as in the previous procedures. All realizations whose "ne" is specified in "CLAP_res_collect.py" are collected.

** Example with SDSS:
> python CLAP_res_collect.py --survey=1 --recali_op=2

## Baseline methods

- For comparison with CLAP, run "CLAP_scl_baseline.py" to produce baseline (end-to-end) models.

- Set "--texp=0" for training an end-to-end model. The argument "num_gmm" specifies the number of Gaussian mixture components in the density output (using the softmax output by setting "--num_gmm=0"). The argument "rate_dropout" specifies the dropout rate used in the dropout layers (not using dropout by setting "--rate_dropout=0"). The arguments "size_latent_ext" and "coeff_recon" now have no impact.

- The user-defined catalogs and multi-band cutout images are required.

** Example of vanilla CNN (end-to-end, softmax output, no dropout) with SDSS:
> python CLAP_scl_baseline.py --ne=1 --survey=1 --net=1 --size_latent_main=16 --batch_train=64 --texp=0 --phase=0

** Example of using 5 Gaussian mixture components:
> python CLAP_scl_baseline.py --ne=1 --survey=1 --net=1 --size_latent_main=16 --batch_train=64 --texp=0 --num_gmm=5 --phase=0

** Example of using a dropout rate of 0.5:
> python CLAP_scl_baseline.py --ne=1 --survey=1 --net=1 --size_latent_main=16 --batch_train=64 --texp=0 --rate_dropout=0.5 --phase=0

- For all these methods, after training, rerun "CLAP_scl_baseline.py" by setting "--phase=1" to produce inference results (saved in output .npz files).
