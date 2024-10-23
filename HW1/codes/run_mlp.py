from typing import Dict, Type
from network import Network
from utils import LOG_INFO
from layers import Selu, HardSwish, Linear, Tanh
from loss import KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import wandb
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser


# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

def attain_args():
    parser = ArgumentParser()
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', "-wd", type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--momentum', "-mo", type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--batch_size', "-b", type=int, default=100, help='Batch size for training')
    parser.add_argument('--max_epoch', "-me", type=int, default=10, help='Maximum number of training epochs')
    parser.add_argument('--disp_freq', "-d", type=int, default=50, help='Frequency of displaying the training status')
    parser.add_argument('--test_epoch', "-te", type=int, default=1, help='Number of epochs between testing')
    args = parser.parse_args()

    arg_dict = vars(args)

    return arg_dict


def init(conf, activate_name, loss_name):
    if loss_name == "Focal":
        conf["learning_rate"] = 1e-2
    wandb.init(
        project=f"mnist_mlp_training",
        config=conf,
        group="function_change_tests",
        name=f"activation: {activate_name} and loss: {loss_name} for 50 epoch"
    )

    # Your model defintion here
    # You should explore different model architecture
    modelx = Network()
    modelx.add(Linear('fc1', 784, 128, 0.01))
    modelx.add(layer_func[activate_name](activate_name))
    modelx.add(Linear('fc2', 128, 10, 0.01))
    lossx = loss_func[loss_name](name="loss")

    return modelx, lossx

def processor(mdl, los, train_data, test_data, train_label, test_label, config):
    # for epoch in tqdm(range(int(config['max_epoch'])), desc='Training'):
    for epoch in range(int(config['max_epoch'])):
        iteration = train_net(mdl, los, config, train_data, train_label, config['batch_size'], config['disp_freq'])

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % epoch)
            test_net(mdl, los, test_data, test_label, config['batch_size'])

layer_func = {
    "tanh": Tanh,
    "SeLu": Selu,
    "HardSwish": HardSwish,
}

loss_func = {
    "KLDiv": KLDivLoss,
    "CrossEntropy": SoftmaxCrossEntropyLoss,
    "Hinge": HingeLoss,
    "Focal": FocalLoss,
}

name_losses = ["KLDiv", "CrossEntropy", "Hinge", "Focal"]
name_activations = ["tanh", "SeLu", "HardSwish"]

if __name__ == '__main__':
    # do argument parse
    wandb.login()
    train_config = attain_args()
    tr_data, te_data, tr_label, te_label = load_mnist_2d('data')

    activate_name = "SeLu"
    loss_name = "Focal"
    
    for i in range(12):
        activate_name = name_activations[i // 4]
        
        loss_name = name_losses[i % 4]
        
        # single model
        # activate_name = "tanh"
        # wandb.init(
        #     project=f"mnist_mlp_training",
        #     config=train_config,
        #     group="function_change_tests",
        #     name=f"activation: {activate_name} and loss: {loss_name} with double layer"
        # )
        # act_func = layer_func[activate_name](activate_name)
        # modelx = Network()
        # modelx.add(Linear('fc1', 784, 256, 0.01))
        # modelx.add(Tanh("selu1"))
        # modelx.add(Linear('fcx', 256, 128, 0.01))
        # modelx.add(Tanh("selu2"))
        # modelx.add(Linear('fc2', 128, 10, 0.01))
        # lossx = loss_func[loss_name](name=loss_name)
        
        # multiple funcs and single layer
        modelx, lossx = init(train_config, activate_name, loss_name)

    
        processor(modelx, lossx, tr_data, te_data, tr_label, te_label, train_config)
        wandb.finish()
