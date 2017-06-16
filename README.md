# deep-calcium
Deep Learning Models for Calcium Imaging Data

## Models: Neuron Segmentation

### UNet2DS: UNet with 2D Summary Images

- Slightly-modified UNet model trained on Neurofinder labeled datasets with no problem-specific pre- or post-processing.
- Implemented with Keras using Tensorflow backend.
- Latest implementation scored 0.5356 on Neurofinder competition
  - Commit: [0bda9d4b9cad71fb3685671c2e699c88d9195a24](https://github.com/alexklibisz/deep-calcium/commit/0bda9d4b9cad71fb3685671c2e699c88d9195a24)
  - Submission md5sum d9b47b2c42e4f04cfd76b308b79680a6.
  - [Training, submission artifacts on Dropbox.](https://www.dropbox.com/sh/tqbclt7muuvqfw4/AADET6ZVlUbHZsqHKgwDOysXa?dl=0)
  - [Pre-trained weights `weights_val_nf_f1_mean.hdf5`](https://www.dropbox.com/sh/tqbclt7muuvqfw4/AACqVVA8oJlZNIYvfc6x6gO2a/weights_val_nf_f1_mean.hdf5?dl=1), md5sum ffaa4c3a5110eae024114d3fbdd438f2.

![UNet2DS 0.5355 scores](media/nf_scores_unet2ds_0.5356.png)