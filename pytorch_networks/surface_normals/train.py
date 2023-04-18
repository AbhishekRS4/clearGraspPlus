'''
Train for surface normals prediction task
'''


## load from python and thrid party modules
import io
import os
import glob
import time
import oyaml
import torch
import shutil
import argparse
import numpy as np
import torch.nn as nn


from tqdm import tqdm
from termcolor import colored
from pyats.datastructures import NestedAttrDict
from torch.optim.lr_scheduler import _LRScheduler


## load from our own modules
import dataset_scannet
from utils import utils
import dataset_matterport
from modeling import deeplab
from loss_functions import *
from dataset import SurfaceNormalsDataset, load_concat_sub_datasets, get_data_loader, get_augumentation_list, resize_tensor


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_epochs = max_epochs
        self.min_lr = min_lr # avoid zero lr
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch/self.max_epochs)**self.power, self.min_lr)
                for base_lr in self.base_lrs]


def get_optimizer_and_loss_func(config, model):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(config.train.optimSgd.learningRate),
        momentum=float(config.train.optimSgd.momentum),
        weight_decay=float(config.train.optimSgd.weight_decay)
    )

    if config.train.lossFunc == 'cosine':
        criterion = loss_fn_cosine
    elif config.train.lossFunc == 'radians':
        criterion = loss_fn_radians
    else:
        raise ValueError(f"Invalid lossFunc from config file. Can only be ['cosine', 'radians']. " +
                         f"Value passed is: {config.train.lossFunc}")
    return optimizer, criterion


def get_lr_scheduler(config, optimizer):
    lr_scheduler = None
    if not config.train.lrScheduler:
        pass
    elif config.train.lrScheduler == "CyclicLR":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=config.train.lrSchedulerCyclic.base_lr,
            max_lr=config.train.lrSchedulerCyclic.max_lr,
            step_size_up=config.train.lrSchedulerCyclic.step_size_up,
            step_size_down=config.train.lrSchedulerCyclic.step_size_down,
            mode="triangular"
        )
    elif config.train.lrScheduler == "StepLR":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.train.lrSchedulerStep.step_size,
            gamma=config.train.lrSchedulerStep.gamma
        )
    elif config.train.lrScheduler == "ReduceLROnPlateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.train.lrSchedulerPlateau.factor,
            patience=config.train.lrSchedulerPlateau.patience,
            verbose=True
        )
    elif config.train.lrScheduler == "PolyLR":
        lr_scheduler =  PolynomialLR(
            optimizer, config.train.numEpochs+1, power=config.train.lrSchedulerPoly.power,
        )
    else:
        print(f"unindentified option: {config.train.lrScheduler}")
    return lr_scheduler


def train_loop(model, train_loader, optimizer, criterion, device, model_type, num_classes, batch_size):
    model.train()

    running_loss_mean = 0.0
    running_loss_median = 0.0
    running_percent_1 = 0.0
    running_percent_2 = 0.0
    running_percent_3 = 0.0

    num_batches = len(train_loader)

    for iter_num, batch in enumerate(tqdm(train_loader)):
        inputs, labels, masks = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)

        normal_vectors = model(inputs)
        normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
        loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
        loss /= batch_size

        loss.backward()
        optimizer.step()

        normal_vectors_norm = normal_vectors_norm.detach().cpu()
        inputs = inputs.detach().cpu()
        labels = labels.detach().cpu()
        mask_tensor = masks.squeeze(1)

        loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = \
            metric_calculator_batch(normal_vectors_norm, labels.double(), mask_tensor)

        running_loss_mean += loss_deg_mean
        running_loss_median += loss_deg_median
        running_percent_1 += percentage_1
        running_percent_2 += percentage_2
        running_percent_3 += percentage_3

    epoch_loss_mean = running_loss_mean / num_batches
    epoch_loss_median = running_loss_median / num_batches
    epoch_percent_1 = running_percent_1 / num_batches
    epoch_percent_2 = running_percent_2 / num_batches
    epoch_percent_3 = running_percent_3 / num_batches
    return (epoch_loss_mean.cpu().detach().numpy(),
            epoch_loss_median.cpu().detach().numpy(),
            epoch_percent_1.cpu().detach().numpy(),
            epoch_percent_2.cpu().detach().numpy(),
            epoch_percent_3.cpu().detach().numpy(),)


def validation_loop(model, validation_loader, criterion, device, num_classes):
    model.eval()

    running_loss_mean = 0.0
    running_loss_median = 0.0
    running_percent_1 = 0.0
    running_percent_2 = 0.0
    running_percent_3 = 0.0

    num_batches = len(validation_loader)

    with torch.no_grad():
        for iter_num, sample_batched in enumerate(tqdm(validation_loader)):
            inputs, labels = sample_batched
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            normal_vectors = model(inputs)
            normal_vectors_norm = nn.functional.normalize(normal_vectors.double(), p=2, dim=1)
            loss = criterion(normal_vectors_norm, labels.double(), reduction='sum')
            loss /= batch_size

            normal_vectors_norm = normal_vectors_norm.detach().cpu()
            inputs = inputs.detach().cpu()
            labels = labels.detach().cpu()
            mask_tensor = masks.squeeze(1)

            loss_deg_mean, loss_deg_median, percentage_1, percentage_2, percentage_3 = \
                metric_calculator_batch(normal_vectors_norm, labels.double(), mask_tensor)

            running_loss_mean += loss_deg_mean
            running_loss_median += loss_deg_median
            running_percent_1 += percentage_1
            running_percent_2 += percentage_2
            running_percent_3 += percentage_3

    epoch_loss_mean = running_loss_mean / num_batches
    epoch_loss_median = running_loss_median / num_batches
    epoch_percent_1 = running_percent_1 / num_batches
    epoch_percent_2 = running_percent_2 / num_batches
    epoch_percent_3 = running_percent_3 / num_batches
    return (epoch_loss_mean.cpu().detach().numpy(),
            epoch_loss_median.cpu().detach().numpy(),
            epoch_percent_1.cpu().detach().numpy(),
            epoch_percent_2.cpu().detach().numpy(),
            epoch_percent_3.cpu().detach().numpy(),)


def test_loop(model, test_loader, criterion, device, num_classes):
    epoch_loss, mean_iou = validation_loop(
        model, test_loader, criterion, device, num_classes,
    )
    return epoch_loss, mean_iou


def start_training(ARGS):
    FILE_PATH_CONFIG = ARGS.config_file
    with open(FILE_PATH_CONFIG) as fd_config_yaml:
        # Returns an ordered dict. Used for printing
        config_yaml = oyaml.load(fd_config_yaml, Loader=oyaml.Loader)
        config_dict = dict(config_yaml)

    config = NestedAttrDict(**config_dict)
    print(colored(f"Config being used for training:\n{oyaml.dump(config_yaml)}\n\n", "green"))

    # Create a new directory to save logs
    runs = sorted(glob.glob(os.path.join(config.train.logsDir, "object_segmentation", "exp-*")))
    prev_run_id = int(runs[-1].split("-")[-1]) if runs else 0
    DIR_CHECKPOINT = os.path.join(config.train.logsDir, "object_segmentation", f"exp-{prev_run_id+1}")
    FILE_NAME_LOGS_CSV = os.path.join(DIR_CHECKPOINT, "train_logs.csv")
    os.makedirs(DIR_CHECKPOINT)
    print(f"Saving logs to folder: " + colored(f"{DIR_CHECKPOINT}", "blue"))

    # Save a copy of config file in the logs
    shutil.copy(FILE_PATH_CONFIG, os.path.join(DIR_CHECKPOINT, "config.yaml"))

    # Create a csv writer
    logging_column_names = [
        "epoch",
        "learning_rate",
        "train_loss_mean",
        "train_loss_median",
        "train_pc_1",
        "train_pc_2",
        "train_pc_3",
        "valid_loss_mean",
        "valid_loss_median",
        "valid_pc_1",
        "valid_pc_2",
        "valid_pc_3",
        "test_real_loss_mean",
        "test_real_loss_median",
        "test_real_pc_1",
        "test_real_pc_2",
        "test_real_pc_3",
        "test_syn_loss_mean",
        "test_syn_loss_median",
        "test_syn_pc_1",
        "test_syn_pc_2",
        "test_syn_pc_3",
    ]
    csv_writer = utils.CSVWriter(
        FILE_NAME_LOGS_CSV,
        logging_column_names,
    )

    ###################### DataLoader #############################
    input_only = [
        "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
        "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
    ]

    augs_train = get_augumentation_list("train",
        config.train.imgHeight, config.train.imgWidth
    )
    augs_test = get_augumentation_list("validation",
        config.train.imgHeight, config.train.imgWidth
    )

    # train set
    db_train = None
    if config.train.datasetsTrain is not None:
        db_train = load_concat_sub_datasets(
            config.train.datasetsTrain,
            augs_train,
            percent_data=config.train.percentageDataForTraining,
            input_only=input_only,
        )

    # validation set
    db_validation = None
    if config.train.datasetsVal is not None:
        db_validation = load_concat_sub_datasets(
            config.train.datasetsVal,
            augs_test,
            percent_data=config.train.percentageDataForValidation,
            input_only=None,
        )

    # test set - real
    db_test_real = None
    if config.train.datasetsTestReal is not None:
        db_test_real = load_concat_sub_datasets(
            config.train.datasetsTestReal,
            augs_test,
            percent_data=None,
            input_only=None,
        )

    # test set - synthetic
    db_test_syn = None
    if config.train.datasetsTestSynthetic is not None:
        db_test_syn = load_concat_sub_datasets(
            config.train.datasetsTestSynthetic,
            augs_test,
            percent_data=None,
            input_only=None,
        )

    # Create dataloaders
    assert (config.train.batchSize <= len(db_train)), \
        (f"batchSize ({config.train.batchSize}) cannot be more than the " +
         f"number of images in training dataset ({len(db_train)})")

    train_loader = get_data_loader(
        db_train,
        config.train.batchSize,
        shuffle=True,
        pin_memory=True
    )

    if db_validation:
        assert (config.train.validationBatchSize <= len(db_validation)), \
            (f'validationBatchSize ({config.train.validationBatchSize}) cannot be more than the ' +
             f'number of images in validation dataset: {len(db_val)}')

        validation_loader = get_data_loader(
            db_validation,
            batch_size=config.train.validationBatchSize,
        )

    if db_test_real:
        assert (config.train.testBatchSize <= len(db_test_real)), \
            (f'testBatchSize ({config.train.testBatchSize}) cannot be more than the ' +
             f'number of images in test real dataset: {len(db_test_real)}')

        test_real_loader = get_data_loader(
            db_test_real,
            batch_size=config.train.testBatchSize,
        )

    if db_test_syn:
        assert (config.train.testBatchSize <= len(db_test_synthetic)), \
            (f'testBatchSize ({config.train.testBatchSize}) cannot be more than the ' +
             f'number of images in test synthetic dataset: {len(db_test_syn)}')

        test_syn_loader = get_data_loader(
            db_test_syn,
            batch_size=config.train.testBatchSize,
        )


    ###################### ModelBuilder #############################
    if config.train.model == 'deeplab_xception':
        model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='xception', sync_bn=True,
                                freeze_bn=False)
    elif config.train.model == 'deeplab_resnet':
        model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='resnet', sync_bn=True,
                                freeze_bn=False)
    elif config.train.model == 'drn':
        model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone='drn', sync_bn=True,
                                freeze_bn=False)  # output stride is 8 for drn
    elif config.train.model == "drn_psa":
        model = deeplab.DeepLab(num_classes=config.train.numClasses, backbone="drn_psa", sync_bn=True,
                                freeze_bn=False)  # output stride is 8 for drn_psa
    else:
        raise ValueError(
            'Invalid model "{}" in config file. Must be one of ["drn", "drn_psa" "deeplab_xception", "deeplab_resnet"]'
            .format(config.train.model))

    if config.train.continueTraining:
        print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
        if not os.path.isfile(config.train.pathPrevCheckpoint):
            raise ValueError('Invalid path to the given weights file for transfer learning.\
                    The file {} does not exist'.format(config.train.pathPrevCheckpoint))

        CHECKPOINT = torch.load(config.train.pathPrevCheckpoint, map_location='cpu')

        if 'model_state_dict' in CHECKPOINT:
            # Our weights file with various dicts
            model.load_state_dict(CHECKPOINT['model_state_dict'])
        elif 'state_dict' in CHECKPOINT:
            # Original Author's checkpoint
            CHECKPOINT['state_dict'].pop('decoder.last_conv.8.weight')
            CHECKPOINT['state_dict'].pop('decoder.last_conv.8.bias')
            model.load_state_dict(CHECKPOINT['state_dict'], strict=False)
        else:
            # Our old checkpoint containing only model's state_dict()
            model.load_state_dict(CHECKPOINT)

    # Enable Multi-GPU training
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ###################### Setup Optimizer #############################
    optimizer, criterion =  get_optimizer_and_loss_func(config, model)
    lr_scheduler = get_lr_scheduler(config, optimizer)

    # Continue Training from prev checkpoint if required
    if config.train.continueTraining and config.train.initOptimizerFromCheckpoint:
        if 'optimizer_state_dict' in CHECKPOINT:
            optimizer.load_state_dict(CHECKPOINT['optimizer_state_dict'])
        else:
            print(
                colored(
                    'WARNING: Could not load optimizer state from checkpoint as checkpoint does not contain ' +
                    '(optimizer_state_dict). Continuing without loading optimizer state. ', 'red'))

    ###################### Train Model #############################
    # Set total iter_num (number of batches seen by model, used for logging)
    total_iter_num = 0
    START_EPOCH = 1
    END_EPOCH = config.train.numEpochs + 1

    if (config.train.continueTraining and config.train.loadEpochNumberFromCheckpoint):
        if 'model_state_dict' in CHECKPOINT:
            # TODO: remove this second check for 'model_state_dict' soon. Kept for ensuring backcompatibility
            total_iter_num = CHECKPOINT['total_iter_num'] + 1
            START_EPOCH = CHECKPOINT['epoch'] + 1
            END_EPOCH = CHECKPOINT['epoch'] + config.train.numEpochs
        else:
            print(
                colored(
                    'Could not load epoch and total iter nums from checkpoint, they do not exist in checkpoint.\
                           Starting from epoch num 0', 'red'))

    for epoch in range(START_EPOCH, END_EPOCH):
        train_loss_mean, train_loss_median, train_pc_1, train_pc_2, train_pc_3 = 0.0, 0.0, 0.0, 0.0, 0.0
        valid_loss_mean, valid_loss_median, valid_pc_1, valid_pc_2, valid_pc_3 = 0.0, 0.0, 0.0, 0.0, 0.0
        test_real_loss_mean, test_real_loss_median, test_real_pc_1, test_real_pc_2, test_real_pc_3 = 0.0, 0.0, 0.0, 0.0, 0.0
        test_syn_loss_mean, test_syn_loss_median, test_syn_pc_1, test_syn_pc_2, test_syn_pc_3 = 0.0, 0.0, 0.0, 0.0, 0.0

        t_1 = time.time()
        print(f"\n\nEpoch {epoch}/{END_EPOCH - 1}")
        print("=" * 20)

        ###################### Training Cycle #############################
        train_loss_mean, train_loss_median, train_percent_1, train_percent_2, train_percent_3 = train_loop(
            model, train_loader, optimizer, criterion, device, config.train.model,
            config.train.numClasses, config.train.batchSize,
        )

        print(f"\ntraining set, mean loss: {train_loss_mean:.4f}, median loss: {train_loss_median:.4f}, percent 1: {train_percent_1:.4f}"+\
            f", percent 2: {train_percent_2:.4f}, percent 3: {train_percent_3:.4f}")
        print("=" * 20)

        # Log Current Learning Rate
        # TODO: NOTE: The lr of adam is not directly accessible. Adam creates a loss for every parameter in model.
        #    The value read here will only reflect the initial lr value.
        try:
            lr_epoch = lr_scheduler.get_last_lr()
            lr_epoch = lr_epoch[0]
        except:
            lr_epoch = config.optimSgd.learningRate

        # Save the model checkpoint every N epochs
        if (epoch % config.train.saveModelInterval) == 0:
            file_path_model = os.path.join(DIR_CHECKPOINT, f"sur_normal_epoch_{epoch}.pth")
            if torch.cuda.device_count() > 1:
                model_params = model.module.state_dict()  # Saving nn.DataParallel model
            else:
                model_params = model.state_dict()

            if config.train.saveOptimizerState:
                torch.save(
                    {
                        "model_state_dict": model_params,
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    file_path_model,
                )
            else:
                torch.save(
                    model_params,
                    file_path_model,
                )
        ###################### Validation Cycle #############################
        if db_validation:
            valid_loss_mean, valid_loss_median, valid_percent_1, valid_percent_2, valid_percent_3 = validation_loop(
                model, validation_loader, criterion, device, config.train.numClasses
            )
            print(f"\validation set, mean loss: {valid_loss_mean:.4f}, median loss: {valid_loss_median:.4f}, percent 1: {valid_percent_1:.4f}"+\
                f", percent 2: {valid_percent_2:.4f}, percent 3: {valid_percent_3:.4f}")
            print("=" * 20)
        ###################### Test Cycle - Real #############################
        if db_test_real:
            test_real_loss_mean, test_real_loss_median, test_real_percent_1, test_real_percent_2, test_real_percent_3 = test_loop(
                model, test_real_loader, criterion, device, config.train.numClasses
            )
            print(f"\test real set, mean loss: {test_real_loss_mean:.4f}, median loss: {test_real_loss_median:.4f}, percent 1: {test_real_percent_1:.4f}"+\
                f", percent 2: {test_real_percent_2:.4f}, percent 3: {test_real_percent_3:.4f}")
            print("=" * 20)
        ###################### Test Cycle - Synthetic #############################
        if db_test_syn:
            test_syn_loss_mean, test_syn_loss_median, test_syn_percent_1, test_syn_percent_2, test_syn_percent_3 = test_loop(
                model, test_syn_loader, criterion, device, config.train.numClasses
            )
            print(f"\test synthetic set, mean loss: {test_syn_loss_mean:.4f}, median loss: {test_syn_loss_median:.4f}, percent 1: {test_syn_percent_1:.4f}"+\
                f", percent 2: {test_syn_percent_2:.4f}, percent 3: {test_syn_percent_3:.4f}")
            print("=" * 20)
        t_2 = time.time()
        print(f"time: {(t_2-t_1):.2f} sec.")
        print("=" * 40)
        csv_writer.write_row(
            [
                epoch,
                round(lr_epoch, 6),
                np.around(train_loss_mean, 6),
                np.around(train_loss_median, 6),
                np.around(train_pc_1, 6),
                np.around(train_pc_2, 6),
                np.around(train_pc_3, 6),
                np.around(valid_loss_mean, 6),
                np.around(valid_loss_median, 6),
                np.around(valid_pc_1, 6),
                np.around(valid_pc_2, 6),
                np.around(valid_pc_3, 6),
                np.around(test_real_loss_mean, 6),
                np.around(test_real_loss_median, 6),
                np.around(test_real_pc_1, 6),
                np.around(test_real_pc_2, 6),
                np.around(test_real_pc_3, 6),
                np.around(test_syn_loss_mean, 6),
                np.around(test_syn_median, 6),
                np.around(test_syn_pc_1, 6),
                np.around(test_syn_pc_2, 6),
                np.around(test_syn_pc_3, 6),
            ]
        )
        # Update Learning Rate Scheduler
        if config.train.lrScheduler == "StepLR" or config.train.lrScheduler == "PolyLR":
            lr_scheduler.step()
        elif config.train.lrScheduler == "ReduceLROnPlateau":
            lr_scheduler.step(train_loss)
    csv_writer.close()
    return


def main():
    ###################### Load Config File #############################
    parser = argparse.ArgumentParser(description='Run training of surface normal prediction model')
    parser.add_argument('-c', '--config_file', required=True, help='Path to config yaml file', metavar='path/to/config')
    ARGS = parser.parse_args()
    start_training(ARGS)
    return

if __name__ == "__main__":
    main()
