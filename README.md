# deep-calcium

Deep Learning Models for Calcium Imaging Data

## Installation and example

Install the package and make predictions on Neurofinder using a pre-trained UNet2DS model.

*This assumes python2.7 and pip2. python3.4 compatibility will hopefully be available soon.*

```
# Install from Github repo.
$ pip2 install --upgrade --user -I pip
$ pip2 install --user git+https://github.com/alexklibisz/deep-calcium.git

# Make a checkpoints directories to save outputs.
mkdir checkpoints

# Download the model and weights.
$ wget https://goo.gl/EjpijZ -O model.hdf5

# Download the example script and evaluate predictions on the first training dataset.
# This will download and preprocess the dataset to ~/.deep-calcium-datasets, requiring ~3.1GB of disk space.
$ wget https://raw.githubusercontent.com/alexklibisz/deep-calcium/dev/examples/neurons/unet2ds_nfdata.py
$ CUDA_VISIBLE_DEVICES="0" python unet2ds_nfdata.py evaluate neurofinder.00.00 --model model.hdf5

```

## Models for Neuron Segmentation

**UNet2DS: [U-Net](https://arxiv.org/abs/1505.04597) with 2D Summary Images**

- Model described in the paper: [Fast, Simple Calcium Imaging Segmentation with Fully Convolutional Networks](https://arxiv.org/abs/1707.06314) by Aleksander Klibisz, Derek Rose, Matthew Eicholtz, Jay Blundon, Stanislav Zakharenko.
- See notebooks for [figures](https://github.com/alexklibisz/deep-calcium/blob/36bd9d1824b6a44c9eac3bb6ce8e25f913c6a6d5/notebooks/dlmia_workshop_figures.ipynb) and [supplementary material](https://github.com/alexklibisz/deep-calcium/blob/36bd9d1824b6a44c9eac3bb6ce8e25f913c6a6d5/notebooks/dlmia_workshop_supplementary.ipynb).
- Trained on data from the [Neurofinder challenge](http://neurofinder.codeneuro.org/) with results below.

| Date | Summary | Mean F<sub>1</sub> Score | All Scores | Model & Weights | Training Artifacts | Commit |
|---|---|---|---|---|---|---|
|6/16/17|UNet with a single batchnorm layer at the input. Images scaled to [0,1]. |0.5356|[Image](https://github.com/alexklibisz/deep-calcium/blob/dev/media/nf_scores_unet2ds_0.5356.png)|[Dropbox](https://www.dropbox.com/sh/tqbclt7muuvqfw4/AACqVVA8oJlZNIYvfc6x6gO2a/weights_val_nf_f1_mean.hdf5?dl=1)|[Dropbox](https://www.dropbox.com/sh/tqbclt7muuvqfw4/AADET6ZVlUbHZsqHKgwDOysXa?dl=0)|[0bda9d4](https://github.com/alexklibisz/deep-calcium/commit/0bda9d4b9cad71fb3685671c2e699c88d9195a24)|
|7/12/17|Same as 6/16/17, but with 8x test-time augmentation. |0.5422|[Image](https://github.com/alexklibisz/deep-calcium/blob/dev/media/nf_scores_unet2ds-tta_0.5422.png)|[Dropbox](https://www.dropbox.com/s/x5bv4klz16ai6wa/model_val_nf_f1_mean.hdf5?dl=1)|[Dropbox](https://www.dropbox.com/sh/tqbclt7muuvqfw4/AADET6ZVlUbHZsqHKgwDOysXa?dl=0)|[f1b33bf](https://github.com/alexklibisz/deep-calcium/commit/f1b33bfe48425d0d7a33f7f74ded19905a24b88f)|
|7/13/17|UNet with batchnorm between each conv and ReLU. Mean subtraction and normalization on each summary image. Mask-summary erosion to eliminate merged neurons in ground-truth mask.|0.5611|[Image](https://github.com/alexklibisz/deep-calcium/blob/dev/media/nf_scores_unet2ds_0.5611.png)|[Dropbox](https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AAAteOMVC45Ovf6g2iu10c_Ya/1499980441_model_07_0.843.hdf5?dl=1)|[Dropbox](https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AABW_ksvueR3GdJIVCyNdFxIa?dl=0)|[2b15d1b](https://github.com/alexklibisz/deep-calcium/blob/2b15d1b07a780ff4b2477524f255e41533fc6205/deepcalcium/models/neurons/unet_2d_summary.py)|
|7/13/17|Same as 7/13/17, but with 8x test-time augmentation. Replaced UNet2DS submission with this one. |0.5689|[Image](https://github.com/alexklibisz/deep-calcium/blob/dev/media/nf_scores_unet2ds-tta_0.5689.png)|[Dropbox](https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AAAteOMVC45Ovf6g2iu10c_Ya/1499980441_model_07_0.843.hdf5?dl=1)|[Dropbox](https://www.dropbox.com/sh/5nwrxj1pvsbxvwn/AABW_ksvueR3GdJIVCyNdFxIa?dl=0)|[2b15d1b](https://github.com/alexklibisz/deep-calcium/blob/2b15d1b07a780ff4b2477524f255e41533fc6205/deepcalcium/models/neurons/unet_2d_summary.py)|
