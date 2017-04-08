# PorousMediaGAN 
<img style="float: right;" width="25%" height="25%", src="https://github.com/LukasMosser/PorousMediaGan/blob/master/misc/render_transp.png">
---
## Abstract


## Paper

## Results
Cross-Sectional Views of the three trained models
- Beadpack Sample  
![Beadpack Comparison](https://github.com/LukasMosser/PorousMediaGan/blob/master/paper/figures/beadpack_comparison.png)
- Berea Sample  
![Berea Comparison](https://github.com/LukasMosser/PorousMediaGan/blob/master/paper/figures/berea_comparison.png)
- Ketton Sample  
![Ketton Comparison](https://github.com/LukasMosser/PorousMediaGan/blob/master/paper/figures/ketton_comparison.png)

## Instructions
### Pre-requisites
- To run any of the `jupyter` notebooks follow instructions [here](http://jupyter.org/install.html) or install via pip.
```bash
pip install jupyter
```
- In addition we make heavy use of `pandas`, `numpy`, `scipy` and `numba`
- We recommend the use of [anaconda](https://www.continuum.io/downloads)
- For numba instructions, you can find a tutorial and installation guideline [here](http://numba.pydata.org/numba-doc/dev/user/installing.html).
- For the torch version of the code training and generating code please follow the instructions [here](https://github.com/soumith/dcgan.torch)
- In addition you will need to have installed torch packages `hdf5` and `dpnn`
```bash
luarocks install hdf5
luarocks install dpnn
```
- For the pytorch version you will need to have installed `h5py` and `tifffile`
```bash
pip install h5py
pip install tifffile
```
- Clone this repo
```bash
git clone https://github.com/LukasMosser/PorousMediaGAN
cd PorousMediaGAN
```

## Pre-trained model (Pytorch version only)
We have included a pre-trained model used for the Berea sandstone example in the paper in the repository.
To create the training image dataset from the full CT image perform the following steps:
- Unzipping of the CT image
```bash
cd ./data/berea/original/raw
#unzip using your preferred unzipper
unzip berea.zip
```


## Directory Overview


# Citation
---
If you use our code for your own research, we would be grateful if you cite our publication
[ArXiv]()
```
@article{PorousMediaGAN
	title={},
	author={Mosser, Lukas and Dubrule, Olivier and Blunt, Martin J.}
	journal={arXiv preprint arXiv:1703.update},
	year={2017}
}
```


## Acknowledgment
---
The code used for our research is based on [DCGAN](https://github.com/soumith/dcgan.torch)
for the [torch](http://torch.ch/) version and the [pytorch](https://github.com/pytorch) example on how to implement a [GAN](https://github.com/pytorch/examples/tree/master/dcgan)
Our dataloader has been modified from [DCGAN](https://github.com/soumith/dcgan.torch).
