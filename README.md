# FedCCS

The source code of the FedCCS:

# Requirement
Python packages:
- Tensorflow (>2.0)
- Jupyter Notebook
- scikit-learn
- matplotlib
- tqdm
 
 You need to download the dataset (e.g. FEMNIST, MNIST, FashionMNIST, Synthetic) and specify a GPU id follows the guidelines of [FedProx](https://github.com/litian96/FedProx) & [Ditto](https://github.com/litian96/ditto). 

### 📌 Please download `mnist, nist, sent140, synthetic` from the `FedProx` repository and rename nist to fmnist, download `femnist` from the `Ditto` repository. The nist in FedProx is 10-class, but the femnist in Ditto is 62-class. We use the 10-class version in this project.

The directory structure of the datasets should look like this:

```
CFL-->data-->mnist-->data-->train--> ***train.json
                |              |->test--> ***test.json
                |
                |->femnist-->data-->train--> ***train.json
                |                  |->test--> ***test.json
                |
                |->fmnist-->data-->...
                |
                |->synthetic_1_1-->data-->...
                |
                ...
```
# Quick Start

You can also run FedCCS with `python main.py`. Please modify `config` according to your needs.

# Experimental Results
All evaluation results will save in the `CFL-->results-->...` directory as `excel` format files.


# Acknowledgements
The code is modified based on `FlexCFL`, thanks for their contribution to CFL work.😊

- [Flexible Clustered Federated Learning for Client-Level Data Distribution Shift](https://arxiv.org/abs/2108.09749)


