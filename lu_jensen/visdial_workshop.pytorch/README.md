# visDial.pytorch
Visual Dialog model in pytorch

### Introduction
This is the pytorch implementation of our NIPS 2017 paper ["Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model"](https://arxiv.org/abs/1706.01554)


### Disclaimer

This is the reimplementation code of visual dialog model based on Pytorch. Our original code was implemented during the first author's internship. All the results presented in our paper were obtained based on the original code, which cannot be released since the firm restriction. This project is an attempt to reproduce the results in our paper.

### Citation
If you find this code useful, please cite the following paper:

    @article{lu2017best,
        title={Best of Both Worlds: Transferring Knowledge from Discriminative Learning to a Generative Visual Dialog Model},
        author={Lu, Jiasen and Kannan, Anitha and and Yang, Jianwei and Parikh, Devi and Batra, Dhruv},
        journal={NIPS},
        year={2017}
    }
### Dependencies

1. PyTorch. Install [PyTorch](http://pytorch.org/) with proper commands. Make sure you also install *torchvision*.

### Evaluation

* The preprocessed feature can be found [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/data/)
* The pre-trained model can be found [here](https://filebox.ece.vt.edu/~jiasenlu/codeRelease/visDial.pytorch/model/)

To evaluate the pre-trained model on validation set, first use the script to download the feature and pre-trained model.
```
python script/download.py --path [path_to_download]
```
After download the feature and pre-trained model, you can run the evaluation script by using following command

* Evaluate the discriminative model:
```
python eval/eval_D.py --data_dir [path_to_root] --model_path [path_to_root]/save/HCIAE-D-MLE.pth --cuda
```

* Evaluate the MLE trained generative model:
```
python eval/eval_G.py --data_dir [path_to_root] --model_path [path_to_root]/save/HCIAE-G-MLE.pth --cuda
```

* Evaluate the DIS trained generative model:
```
python eval/eval_G_DIS.py --data_dir [path_to_root] --model_path [path_to_root]/save/HCIAE-G-DIS.pth --cuda
```
You will get the similar results as in the paper :)

### Train a visual dialog model.

#### Preparation

#### Training


