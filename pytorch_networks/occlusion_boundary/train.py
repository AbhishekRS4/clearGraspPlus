'''
Train for transparent object occlusion boundary detection task
'''


## load from python and thrid party modules
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
from utils import utils
from modeling import deeplab
from dataset import OcclusionBoundaryDataset, load_concat_sub_datasets, get_data_loader, get_augumentation_list


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
    #criterion = utils.cross_entropy2d
    criterion = utils.FocalLoss
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


def train_loop(model, train_loader, optimizer, criterion, device, model_type, num_classes):
    model.train()

    running_loss = 0.0
    total_iou = 0.0
    mean_iou = 0.0

    num_batches = len(train_loader)

    for iter_num, batch in enumerate(tqdm(train_loader)):
        inputs, labels = batch
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        # Forward + Backward Prop
        optimizer.zero_grad()
        torch.set_grad_enabled(True)
        outputs = model.forward(inputs)

        #pred_labels = torch.max(outputs, 1)[1]
        pred_labels = torch.argmax(outputs, 1)
        weight = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float32)
        #weight = weight.to(device)
        loss = criterion(outputs.cpu(), labels.cpu(), weight=weight)

        loss.backward()
        optimizer.step()
        running_loss += loss

        _total_iou = utils.compute_mean_IOU(
            labels, pred_labels, num_classes=num_classes,
        )
        total_iou += _total_iou

    epoch_loss = running_loss / num_batches
    mean_iou = total_iou / num_batches
    return epoch_loss.cpu().detach().numpy(), mean_iou


def validation_loop(model, validation_loader, criterion, device, num_classes):
    model.eval()

    running_loss = 0.0
    total_iou = 0.0
    mean_iou = 0.0

    num_batches = len(validation_loader)

    with torch.no_grad():
        for iter_num, sample_batched in enumerate(tqdm(validation_loader)):
            inputs, labels = sample_batched
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model.forward(inputs)

            #pred_labels = torch.max(outputs, 1)[1]
            pred_labels = torch.argmax(outputs, 1)
            weight = torch.tensor([1.0, 5.0, 3.0], dtype=torch.float32)
            #weight = weight.to(device)
            loss = criterion(outputs.cpu(), labels.cpu(), weight=weight)
            running_loss += loss

            _total_iou = utils.compute_mean_IOU(
                labels, pred_labels, num_classes=num_classes,
            )
            total_iou += _total_iou

    epoch_loss = running_loss / num_batches
    mean_iou = total_iou / num_batches
    return epoch_loss.cpu().detach().numpy(), mean_iou


def test_loop(model, test_loader, criterion, device, num_classes):
    epoch_loss, mean_iou = validation_loop(
        model, test_loader, criterion, device, num_classes,
    )
    return epoch_loss, mean_iou


def start_training(ARGS):
    CONFIG_FILE_PATH = ARGS.config_file
    with open(CONFIG_FILE_PATH) as fd_config_yaml:
        config_yaml = oyaml.load(fd_config_yaml, Loader=oyaml.Loader)  # Returns an ordered dict. Used for printing
        config_dict = dict(config_yaml)

    config = NestedAttrDict(**config_dict)
    print(colored(f'Config being used for training:\n{oyaml.dump(config_yaml)}\n\n', 'green'))

    # Create a new directory to save logs
    runs = sorted(glob.glob(os.path.join(config.train.logsDir, "occlusion_boundary", 'exp-*')))
    prev_run_id = int(runs[-1].split('-')[-1]) if runs else 0
    DIR_CHECKPOINT = os.path.join(config.train.logsDir, "occlusion_boundary", f'exp-{prev_run_id+1}')
    FILE_NAME_LOGS_CSV = os.path.join(DIR_CHECKPOINT, "train_logs.csv")
    os.makedirs(DIR_CHECKPOINT)
    print(f'Saving logs to folder: ' + colored(f'{DIR_CHECKPOINT}', 'blue'))

    # Save a copy of config file in the logs
    shutil.copy(CONFIG_FILE_PATH, os.path.join(DIR_CHECKPOINT, 'config.yaml'))

    # Create a csv writer
    logging_column_names = [
        "epoch",
        "learning_rate"
        "train_loss",
        "train_iou",
        "valid_loss",
        "valid_iou",
        "test_real_loss",
        "test_real_iou",
        "test_syn_loss",
        "test_syn_iou",
    ]
    csv_writer = utils.CSVWriter(
        FILE_NAME_LOGS_CSV,
        logging_column_names,
    )

    ###################### DataLoader #############################
    input_only = [
        "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
        "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout", "cdropout_black"
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
    # NOTE: Calculation of statistics like epoch_loss depend on the param drop_last being True. They calculate total num
    #       of images as num of batches * batchSize, which is true only when drop_last=True.
    assert (config.train.batchSize <= len(db_train)), \
        (f'batchSize ({config.train.batchSize}) cannot be more than ' +
         f'the number of images in training dataset ({len(db_train)})')

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
             f'number of images in test dataset: {len(db_test_real)}')

        test_real_loader = get_data_loader(
            db_test_real,
            batch_size=config.train.testBatchSize,
        )

    if db_test_syn:
        assert (config.train.testBatchSize <= len(db_test_syn)), \
            (f'testBatchSize ({config.train.testBatchSize}) cannot be more than the ' +
             f'number of images in test dataset: {len(db_test_syn)}')

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
        raise ValueError(f'Invalid model ({config.train.model}) in config file. Must be one of ["deeplab_xception", "deeplab_resnet", "drn", "drn_psa"]')

    if config.train.continueTraining:
        print('Transfer Learning enabled. Model State to be loaded from a prev checkpoint...')
        if not os.path.isfile(config.train.pathPrevCheckpoint):
            raise ValueError('Invalid path to the given weights file for transfer learning.\
                    The file {} does not exist'.format(config.train.pathPrevCheckpoint))

        CHECKPOINT = torch.load(config.train.pathPrevCheckpoint, map_location='cpu')

        if 'model_state_dict' in CHECKPOINT:
            # Newer weights file with various dicts
            print(colored('Continuing training from checkpoint...Loaded data from checkpoint:', 'green'))
            # print('Config Used to train Checkpoint:\n', oyaml.dump(CHECKPOINT['config']), '\n')
            # print('From Checkpoint: Last Epoch Loss:', CHECKPOINT['epoch_loss'], '\n\n')

            model.load_state_dict(CHECKPOINT['model_state_dict'])
        else:
            # Old checkpoint containing only model's state_dict()
            model.load_state_dict(CHECKPOINT)

    # Enable Multi-GPU training
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
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
        train_loss, train_iou = 0.0, 0.0
        valid_loss, valid_iou = 0.0, 0.0
        test_real_loss, test_real_iou = 0.0, 0.0
        test_syn_loss, test_syn_iou = 0.0, 0.0

        t_1 = time.time()
        print(f"\n\nEpoch {epoch}/{END_EPOCH - 1}")
        print("=" * 20)

        ###################### Training Cycle #############################
        print('Train:')
        print('=' * 10)

        # Update Learning Rate Scheduler
        if config.train.lrScheduler == 'StepLR' or config.train.lrScheduler == 'PolyLR':
            lr_scheduler.step()
        elif config.train.lrScheduler == 'ReduceLROnPlateau':
            lr_scheduler.step(train_loss)

        train_loss, train_iou = train_loop(
            model, train_loader, optimizer, criterion, device, config.train.model,
            config.train.numClasses,
        )

        print(f"\ntraining set, loss: {train_loss:.4f}, mean IoU: {train_iou:.4f}")
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
            file_path_model = os.path.join(DIR_CHECKPOINT, f"occ_boundary_epoch_{epoch}.pth")
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
            valid_loss, valid_iou = validation_loop(
                model, validation_loader, criterion, device, config.train.numClasses
            )
            print(f"\nvalidation set, loss: {valid_loss:.4f}, mean IoU: {valid_iou:.4f}")
            print("=" * 20)
        ###################### Test Cycle - Real #############################
        if db_test_real:
            test_loss, test_real_iou = test_loop(
                model, test_real_loader, criterion, device, config.train.numClasses
            )
            print(f"\ntesting real set, loss:{test_loss:.4f}, mean IoU: {test_real_iou:.4f}")
            print("=" * 20)
        ###################### Test Cycle - Synthetic #############################
        if db_test_syn:
            test_syn_loss, test_syn_iou = test_loop(
                model, test_syn_loader, criterion, device, config.train.numClasses
            )
            print(f"\ntesting synthetic set, loss: {test_syn_loss:.4f},  mean IoU: {test_syn_iou:.4f}")
            print("=" * 20)
        t_2 = time.time()
        print(f"time: {(t_2-t_1):.2f} sec.")
        print("=" * 40)
        csv_writer.write_row(
            [
                epoch,
                round(lr_epoch, 6),
                np.around(train_loss, 6),
                np.around(train_iou, 6),
                np.around(valid_loss, 6),
                np.around(valid_iou, 6),
                np.around(test_real_loss, 6),
                np.around(test_real_iou, 6),
                np.around(test_syn_loss, 6),
                np.around(test_syn_iou, 6),
            ]
        )
    # close the csv writer
    csv_writer.close()
    return


def main():
    ###################### Load Config File #############################
    parser = argparse.ArgumentParser(description='Run training of transparent object occlusion boundary detection model')
    parser.add_argument('-c', '--config_file', required=True, help='Path to config yaml file', metavar='path/to/config')
    ARGS = parser.parse_args()
    start_training(ARGS)
    return


if __name__ == "__main__":
    main()
