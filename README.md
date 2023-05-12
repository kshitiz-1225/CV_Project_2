# Comparative Analysis of One-Shot Learning

The aim for this repository is to contain clean, readable and tested code to reproduce one-shot learning research.

This project is written in python 3.7 and Pytorch and assumes you have a GPU.

We implemented the following few shot models:

- [Prototypical Networks](https://arxiv.org/abs/1703.05175)
- [Siamese Networks](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Matching Networks](https://arxiv.org/abs/1606.04080)
- [Relation Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf)
- [Model Agnostic Meta Learning ( MAML )](https://arxiv.org/abs/1703.03400)

### Data

We have implemented our models on the following dataset Omniglot, Mini-ImageNet and CIFAR-FS

**Omniglot** dataset. Download from [Link](https://github.com/brendenlake/omniglot/tree/master/python) place the extracted files into `DATA_PATH/Omniglot_Raw` and run`scripts/preprocessing_omniglot.py`

**Mini-ImageNet** dataset. Download files from [Link](https://drive.google.com/file/d/1-31FtYmm42a1MbU67weeP1Juh7SgGq7e/view?usp=sharing)
place in `DATA_PATH/miniImageNet/images` and run`preprocessing_miniImageNet.py`

**CIFAR-FS** dataset. Download files using `sh download_cifarfs.sh` [Link](https://www.dropbox.com/s/wuxb1wlahado3nq/cifar-fs-splits.zip?dl=0)

## Steps to run locally

- Clone this repository and launch code:

- Use pip to install other dependencies from `requirements.txt`

```
pip install -r requirements.txt
```

- Set the DATA_PATH in config.py

```
pip install -r requirements.txt
```

- Download the dataset

### Training 

- Train Network

```
python <python_file.py> --dataset <dataset/path> --n-train 1 --n-test 1 --k-train 5 --k-test 5
```

### Testing

- Test Network

```
python <python_file.py> --dataset <dataset/path> --n-train 1 --n-test 1 --k-train 5 --k-test 5
```
### Training/Testing
```
python main.py <train_model/test_model>
```
## Team
- [Kshitiz ](https://github.com/kshitiz-1225)
- [Harsh Agarwal](https://github.com/harsh-ux)
