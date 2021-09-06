# Datasets used with ACTOR

## Credits and license agreement
For all the datasets, be sure to **read** and **follow** their license agreements, and **cite** them accordingly.

## How was the data prepared?
### NTU13 ("Refined NTU-RGBD")
Donwload the data from the [Action2Motion webpage](https://ericguo5513.github.io/action-to-motion/) and extract it to the ``data/ntu13/`` folder.

**Update** (from Action2Motion website): "Due to the Release Agreement of NTU-RGBD dataset, we are not allowed to and will no longer provide the access to our re-estimated NTU-RGBD data."

Unfortunately, we cannot provide it either.

#### Bibtex
```bibtex
@inproceedings{nturgbd2016,
  title = {{NTU RGB+D}: A Large Scale Dataset for 3D Human Activity Analysis},
  author = {Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2016},
  pages = {1010-1019}
}
```


### UESTC dataset
Please download our post-processed [VIBE](https://github.com/mkocabas/VIBE) estimations of the SMPL parameters [there](https://drive.google.com/file/d/1LE-EmYNzECU8o7A2DmqDKtqDMucnSJsy/view?usp=sharing]), and extract it to the ``data/uestc/`` folder.

#### Note on post-processing
In many of the videos in the UESTC dataset, there are non-active people in the background. Since VIBE produces tracks for all people in the video, we remove these background person tracks and only keep one track. In addition, there are sometimes cases where there are missing frames for a track. In this case, we identify the correct track and perform a rotation interpolation to fill the gap.

If you are interested on how this processing was performed, you can see the script ``src/preprocess/uestc_vibe_postprocessing.py`` and the raw [VIBE estimation](https://lsh.paris.inria.fr/surreact/vibe_uestc.tar.gz) from the [SURREACT](https://github.com/gulvarol/surreact/blob/master/datageneration/README.md) webpage.

#### Bibtex
```bibtex
@inproceedings{uestc2018,
author = {Ji, Yanli and Xu, Feixiang and Yang, Yang and Shen, Fumin and Shen, Heng Tao and Zheng, Wei-Shi},
title = {A Large-Scale {RGB-D} Database for Arbitrary-View Human Action Recognition},
year = {2018},
doi = {10.1145/3240508.3240675},
booktitle = {ACM International Conference on Multimedia (ACMMM)},
pages = {1510–1518}
}
```


### HumanAct12 dataset
This dataset is from the authors of [Action2Motion](https://ericguo5513.github.io/action-to-motion/). It consists of temporal cropping action annotating of the [PHSPDataset](https://jimmyzou.github.io/publication/2020-PHSPDataset). Action2Motion provides 3D joints and labels from their [website](https://ericguo5513.github.io/action-to-motion/). We extract the SMPL poses from the PHSPDataset with the same temporal cropping as Action2Motion.

You can find our post-processed version [here](https://drive.google.com/file/d/1130gHSvNyJmii7f6pv5aY5IyQIWc3t7R/view?usp=sharing), and you should extract it to ``data/HumanAct12Poses/``.


#### Note on post-processing
In order to obtain SMPL parameters, you can download from the [PHSPD Google Drive](https://drive.google.com/drive/folders/1ZGkpiI99J-4ygD9i3ytJdmyk_hkejKCd):
- ``pose.zip``, extract it move/rename the ``pose`` folder to ``data/PHPSDposes``
- ``CamParams0906.pkl`` and ``CamParams0909.pkl``, and move them to a freshly new created folder ``data/phspdCameras/``
- ``HumanAct12.zip``, extract it and move it to ``data/HumanAct12/``

The script ``python src/preprocess/humanact12_process.py`` extracts the SMPL poses of the HumanAct12 dataset. The results should be in ``data/HumanAct12Poses/humanact12poses.pkl``

#### Bibtex
```bibtex
@inproceedings{zou2020polarization,
  title="{3D} Human Shape Reconstruction from a Polarization Image",
  author={Zou, Shihao and Zuo, Xinxin and Qian, Yiming and Wang, Sen and Xu, Chi and Gong, Minglun and Cheng, Li},
  booktitle="European Conference on Computer Vision (ECCV)",
  pages="351--368",
  year="2020"
}
```

```bibtex
@inproceedings{chuan2020action2motion,
  title="Action2Motion: Conditioned Generation of 3D Human Motions",
  author={Guo, Chuan and Zuo, Xinxin and Wang, Sen and Zou, Shihao and Sun, Qingyao and Deng, Annan and Gong, Minglun and Cheng, Li},
  booktitle="ACM International Conference on Multimedia (ACMMM)",
  pages="2021--2029",
  year="2020"
}
```
