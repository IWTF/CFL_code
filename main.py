from httplib2 import FailedToDecompressContent
from flearn.trainer.fedgroup import FedGroup
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.fedours import FedCos
from utils.trainer_utils import TrainConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    config = TrainConfig('femnist', 'mlp', 'fedavg')
    config.trainer_config['dynamic'] = False
    trainer = FedAvg(config)
    trainer.train()
    # trainer.train_locally()


def main_flexcfl(dataset, model):
    config = TrainConfig(dataset, model, 'fedgroup')
    config.trainer_config['dynamic'] = False
    trainer = FedGroup(config)
    trainer.train()


def main_fedcos(dataset, model):
    config = TrainConfig(dataset, model, 'fedcos')
    config.trainer_config['eval_global_model'] = False
    trainer = FedCos(config)
    trainer.train()


# main_flexcfl('cifar', 'cnn')
main_fedcos('cifar', 'cnn')
