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

wandb.init(project="mnist_mlp_training", config={
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "momentum": 0.9,
    "batch_size": 100,
    "max_epoch": 100,
    "disp_freq": 50,
    "test_epoch": 1,
    "name": "default",
})

def attain_args():
    parser = ArgumentParser()
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', "-wd", type=float, default=1e-5, help='Weight decay for regularization')
    parser.add_argument('--momentum', "-mo", type=float, default=0.9, help='Momentum for the optimizer')
    parser.add_argument('--batch_size', "-b", type=int, default=100, help='Batch size for training')
    parser.add_argument('--max_epoch', "-me", type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--disp_freq', "-d", type=int, default=50, help='Frequency of displaying the training status')
    parser.add_argument('--test_epoch', "-te", type=int, default=1, help='Number of epochs between testing')
    args = parser.parse_args()
    return args


def init(conf, name):
    now = datetime.now().strftime("%Y_%M_%D_%H_%M_%S")
    wandb.init(
        project=f"mnist_mlp_training_{now}",
        config=conf,
        name=name
    )

    # Your model defintion here
    # You should explore different model architecture
    model = Network()
    model.add(Linear('fc1', 784, 128, 0.01))
    model.add(Tanh('tanh'))
    model.add(Linear('fc2', 128, 10, 0.01))
    loss = HingeLoss(name='loss')

    return model, loss

def processor(mdl, los, train_data, test_data, train_label, test_label, config):

    for epoch in tqdm(range(config['max_epoch']), desc='Training'):
        LOG_INFO('Training @ %d epoch...' % epoch)
        iteration = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

        if epoch % config['test_epoch'] == 0:
            LOG_INFO('Testing @ %d epoch...' % epoch)
            test_net(model, loss, test_data, test_label, config['batch_size'])


if __name__ == '__main__':
    # do argument parse
    train_config = attain_args()

    model, loss = init(vars(train_config))

    tr_data, te_data, tr_label, te_label = load_mnist_2d('data')

    processor(model, loss, tr_data, te_data, tr_label, te_label, train_config)
