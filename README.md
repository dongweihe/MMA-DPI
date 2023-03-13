# Multi-modality attribute learning-based method for drug-protein interaction prediction based on deep neural network
Identification of active candidate compounds for target proteins, also called drug-protein interaction prediction (DPI), is an essential but time-consuming and expensive step, which leads to fostering the development of drug discovery. In recent years, deep network-based learning methods were frequently proposed in DPIs due to their powerful capability of feature representation. However, the performance of existing DPI methods is still limited by insufficiently labeled pharmacological data and neglected intermolecular information. Therefore, overcoming these difficulties to perfect the performance of DPIs is an urgent challenge for researchers. In this article, we designed an innovative “multi-modality attributes” learning-based framework for DPIs with molecular transformer and graph convolutional networks, termed, multi-modality attributes (MMA)-DPI. Specifically, intermolecular sub-structural information and chemical semantic representations were extracted through an augmented transformer module from biomedical data. A tri-layer graph convolutional neural network module was applied to associate the neighbor topology information and learn the condensed dimensional features by aggregating a heterogeneous network that contains multiple biological representations of drugs, proteins, diseases and side effects. Then, the learned representations were taken as the input of a fully connected neural network module to further integrate them in molecular and topological space. Finally, the attribute representations were fused with adaptive learning weights to calculate the interaction score for the DPIs tasks. MMA-DPI was evaluated in different experimental conditions and the results demonstrate that the proposed method achieved higher performance than existing state-of-the-art (SoTA) frameworks.

## The environment of MMA-DPI
```
python==3.7.12
tensorflow==1.15.0
visualdl==2.1.1
scikit-learn==1.0.2
scipy==1.7.3
subword-nmt==0.3.8
PyYAML==5.4.1
numpy==1.21.6
pandas==1.3.5
networkx==2.1
pgl==2.2.4
paddlepaddle==2.0.2
matplotlib==3.5.2
```

## Dataset description
In this paper, four datasets are used, i.e., BindingDB, Luo, Davis and YAM. The directory structure are shown below:

```txt
data
|-- macro attribute
|   |-- Luo
|   |   |--sevenNets
|   |   |--sim_network
|   |   |--oneTooneIndex
|   |-- YAM
|       |-- enzyme
|       |-- gpcr
|       |-- ic
|       |-- nr
|-- micro attribute
    |--BindingDB  
    |--Davis
    |--vocabulary
```

The data file of 'data/micro attribute/vocabulary' contains the sub-structure information for drugs and proteins.

## Run the MMA-DPI for DPI prediction
You can run our model using BindingDB dataset with:
```sh
python evaluation.py
```

Also run it using Davis dataset with:
```sh
python evaluation.py --dataset ${'Davis'}
```

### Note
If you want to run our model with Luo et al. or Yam dataset, please change the data path in the file of evaluation.py



# Reference

**IIFDTI**
```
@article{doi:10.1093/bioinformatics/btac485,
    author = {Zhongjian Cheng, Qichang Zhao, Yaohang Li and Jianxin Wang},
    title = {IIFDTI: predicting drug–target interactions through interactive and independent features based on attention mechanism},
    journal = {Bioinformatics},
    volume = {38},
    number = {17},
    pages = {4153-4161},
    year = {2022},
    publisher={Oxford University Press}
}
```


**TransfomerCPI**
```
@article{doi:10.1093/bioinformatics/btaa524,
    author = {Lifan Chen, Xiaoqin Tan, Dingyan Wang, Feisheng Zhong, Xiaohong Liu, Tianbiao Yang, Xiaomin Luo, Kaixian Chen, Hualiang Jiang and Mingyue Zheng},
    title = {TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments},
    journal = {Bioinformatics},
    volume = {38},
    number = {16},
    pages = {4406-4414},
    year = {2020},
    publisher={Oxford University Press}
}
```

**MolTrans**
```
@article{doi:10.1093/bioinformatics/xxxxxx,
  title={MolTrans: Molecular Interaction Transformer for drug--target interaction prediction},
  author={Huang, Kexin and Xiao, Cao and Glass, Lucas M and Sun, Jimeng},
  journal={Bioinformatics},
  volume={37},
  number={6},
  pages={830--836},
  year={2021},
  publisher={Oxford University Press}
}
```
