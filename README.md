# [Deep learning using a biophysical model for Robust and Accelerated Reconstruction (RoAR) of quantitative, artifact-free and denoised R2* images](https://arxiv.org/abs/1912.07087)

## Dependencies
This code was written using:

1. python 3.6.3 Anaconda
2. tensorflow-gpu 1.13.1
3. numpy 1.16.4
4. cudnn 7.6.0
5. cudatoolkit 9.0

## Training
### RoAR (normal input)
1. Create one MAT file per patient, with numerical file names 1, 2, 3, etc. Each file should contain the following variables (with the below names and dimensions)
      * echos (y, x, z, number echos) - gradient recalled echos for input
      * brainmsk (y, x, z) - a mask for the echos with 0's masking out the background and skull
      * Ffun (y,x,z, number echos) - the normalized F-function for the echos
2. Create a folder containing these MAT files.
3. In config.ini
      * data_path = folder containing the MAT files
      * denoise = false
4. Run train.py
### RoAR(high noise input)
1. Same as step 1 for normal input with an optional addition of:
      * S0 (y,x,z) - S0 predicted by either RoAR trained on normal input (above) or NLLS. This is used to standardize noise level.
2. Same as step 2 above
3. In config.ini
      * data_path = folder containing the MAT files
      * denoise = true
      * lower_bound = lowest SNR noise to add
      * upper_bound = highest SNR noise to add
      * If you did not choose to calculate S0 for noise standardization you may choose which echo to standardize noise on by setting standardize_denoise = echo # of your choice (1, 3, etc)
      * If you did calculate S0 set standardize_denoise = 0
4. Same as step 4 above

## Test
We've included links to download model files [here](https://www.dropbox.com/sh/qzqx7mv7s3pynb5/AAD4RtoFsExynzUZPLDFyBl9a?dl=0), where we have both the model trained on in vivo high SNR echos and the model trained on noisy synthetic echos. Run test.py after filling in the parameters of the generate_results method:
* Full path to the saved model
* Path to a MAT file with input echos and a brain mask
* Path to a folder to save results to
* Filename to save results as
* Boolean indicating usage of GPU or CPU to run the network
