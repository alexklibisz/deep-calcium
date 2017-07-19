# deep-calcium

Deep Learning Models for Calcium Imaging Data

## Example

This will be streamlined in the future.

```
# Clone the repository.
git clone https://github.com/alexklibisz/deep-calcium && cd deep-calcium

# Setup a virtual environment (google "virtual env wrapper" and install it).
mkvirtualenv deep-calcium

# Install dependencies inside the virtual environment.
pip install --upgrade pip
pip install -r requirements.txt

# Make data and checkpoints directories.
mkdir data checkpoints

# Run one of the examples - training UNet2DS on neurofinder.00.00 dataset.
# Open the script and read the code to understand what's happening.
# Note the following will require ~3.1GB of disk space to store the data.
CUDA_VISIBLE_DEVICES="0" python examples/unet2ds_nf.py train neurofinder.00.00

```

## Models for Neuron Segmentation

**UNet2DS: [U-Net](https://arxiv.org/abs/1505.04597) with 2D Summary Images**

- Trained on data from the [Neurofinder challenge](http://neurofinder.codeneuro.org/) with the following results.

| Date | Summary | Mean F<sub>1</sub> Score | All Scores | Model & Weights | Training Artifacts | Commit |
|---|---|---|---|---|---|---|
|6/16/17|UNet with a single batchnorm layer at the input. Images scaled to [0,1]. |0.5356|[Image](https://github.com/alexklibisz/deep-calcium/blob/dev/media/nf_scores_unet2ds_0.5356.png)|[Dropbox](https://www.dropbox.com/sh/tqbclt7muuvqfw4/AACqVVA8oJlZNIYvfc6x6gO2a/weights_val_nf_f1_mean.hdf5?dl=1)|[Dropbox](https://www.dropbox.com/sh/tqbclt7muuvqfw4/AADET6ZVlUbHZsqHKgwDOysXa?dl=0)|[0bda9d4](https://github.com/alexklibisz/deep-calcium/commit/0bda9d4b9cad71fb3685671c2e699c88d9195a24)|
|7/12/17|Same as 6/16/17, but with 8x test-time augmentation. |0.5422|[Image](https://github.com/alexklibisz/deep-calcium/blob/dev/media/nf_scores_unet2ds-tta_0.5422.png)|[Dropbox](https://www.dropbox.com/s/x5bv4klz16ai6wa/model_val_nf_f1_mean.hdf5?dl=1)|[Dropbox](https://www.dropbox.com/sh/tqbclt7muuvqfw4/AADET6ZVlUbHZsqHKgwDOysXa?dl=0)|[f1b33bf](https://github.com/alexklibisz/deep-calcium/commit/f1b33bfe48425d0d7a33f7f74ded19905a24b88f)|
|7/13/17|UNet with batchnorm between each conv and ReLU. Mean subtraction and normalization on each summary image. Mask-summary erosion to eliminate merged neurons in ground-truth mask.|0.5611|[Image](https://github.com/alexklibisz/deep-calcium/blob/dev/media/nf_scores_unet2ds_0.5611.png)|[Dropbox](https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AAAteOMVC45Ovf6g2iu10c_Ya/1499980441_model_07_0.843.hdf5?dl=1)|[Dropbox](https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AABW_ksvueR3GdJIVCyNdFxIa?dl=0)|[2b15d1b](https://github.com/alexklibisz/deep-calcium/blob/2b15d1b07a780ff4b2477524f255e41533fc6205/deepcalcium/models/neurons/unet_2d_summary.py)|
|7/13/17|Same as 7/13/17, but with 8x test-time augmentation.|0.5689|[Image](https://github.com/alexklibisz/deep-calcium/blob/dev/media/nf_scores_unet2ds-tta_0.5689.png)|[Dropbox](https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AAAteOMVC45Ovf6g2iu10c_Ya/1499980441_model_07_0.843.hdf5?dl=1)|[Dropbox](https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AABW_ksvueR3GdJIVCyNdFxIa?dl=0)|[2b15d1b](https://github.com/alexklibisz/deep-calcium/blob/2b15d1b07a780ff4b2477524f255e41533fc6205/deepcalcium/models/neurons/unet_2d_summary.py)|
