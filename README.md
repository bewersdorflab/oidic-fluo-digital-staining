# oidic-smlm-digital-staining
This repository contains code for [1]. The neural network, DuRDN, was first published in [2]. Please cite both [1] and [2] when using this repository.

## Setup
MATLAB

Python 3.12.12

PyTorch 2.5.1

## Directories and files
`Data/`: directory to put raw and pre-processed data.

`img_proc_code/`: code for pre-processing data.

`v1_OIDIC_2d_code/`: directory of OI-DIC digital staining model.

`v6_4DIC_2d_code/`: directory of 4-DIC digital staining model.

`v7_1DIC_2d_code/`: directory of 1-DIC digital staining model.

`.sh` scripts: example scripts run in Yale Bouchet cluster. The corresponding output is in the `/outputs/` folder in the same directory as the `.sh` file.

## How to reproduce example outputs
1. Clone this repository.
2. Reproduce from pre-processed data: download data from () Fig. 6 folder `Data` to the `Data` folder in the local repository and then run `.sh` scripts.
3. Reproduce from raw data: download data from () Fig. 6 folder `Data/RAW/0_1_RAW/` to the `Data` folder  in the local repository, sequentially run pre-processing code `_Norm.m`, `_Arrange.m`, and `_Processing.m`, and then run `.sh` scripts.
4. Please note that for the given example:
 - For nucleus model, raw images number 22-76, 103-154, 181-570 are for training (497 image pairs); 1-21, 571-596 (47 image pairs) are for testing/validation.
 - For mitochondria model, raw images number 1-29, 49-504 (excluding black frames 314, 319, 410, 432), 534-572 (520 image pairs) are for training; 30-40, 505-533 (48 image pairs) are for testing/validation.
 - For lipid-droplet model, raw images number 1-312 (excluding black frames 9, 14, 127, 170, 277) (307 image pairs) are for training; 313-338 (excluding black frame 326) (25 image pairs) are for testing/validation.

## References
[1] Bao, Y. and Marin, Z. et al. A correlative quantitative phase and super-resolution fluorescence microscope for imaging cellular structures and dynamics. *Light, Science & Applications* (conditionally accepted).

[2] Chen, X. C. et al. CT-free attenuation correction for dedicated cardiac SPECT using a 3D dual squeeze-and-excitation residual dense network. *Journal of Nuclear Cardiology* 29, 2235-2250 (2022). 
