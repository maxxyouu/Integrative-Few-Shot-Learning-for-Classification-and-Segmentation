import torch
import argparse

from pytorch_lightning import Trainer

from model.asnet import AttentiveSqueezeNetwork
from model.panet import PrototypeAlignmentNetwork
from model.pfenet import PriorGuidedFeatureEnrichmentNetwork
from model.hsnet import HypercorrSqueezeNetwork
from model.asnethm import AttentiveSqueezeNetworkHM
from model.universeg_lightning import UniverSeg
from data.dataset import FSCSDatasetModule
from common.callbacks import MeterCallback, CustomProgressBar, CustomCheckpoint, OnlineLogger
from pytorch_lightning.callbacks.progress import TQDMProgressBar


def main(args):

    # Method
    modeldict = dict(panet=PrototypeAlignmentNetwork,
                        pfenet=PriorGuidedFeatureEnrichmentNetwork,
                        hsnet=HypercorrSqueezeNetwork,
                        asnethm=AttentiveSqueezeNetworkHM,
                        asnet=AttentiveSqueezeNetwork,
                        universeg=UniverSeg)
    modelclass = modeldict[args.method]

    # Dataset initialization
    dm = FSCSDatasetModule(args)

    # Pytorch-lightning main trainer
    checkpoint_callback = CustomCheckpoint(args)
    trainer = Trainer(accelerator='auto', #accelerator='dp',  # DataParallel TODO: run on different accelerator in the lab env
                    callbacks=[MeterCallback(args), CustomCheckpoint(args), TQDMProgressBar(refresh_rate=1)], #
                    # num_nodes=torch.cuda.device_count(), # gpus=torch.cuda.device_count(),
                    logger=False if args.nowandb or args.eval else OnlineLogger(args),
                    # progress_bar_refresh_rate=1,
                    max_epochs=args.niter,
                    num_sanity_val_steps=0,
                    enable_model_summary=True,
                    # resume_from_checkpoint=checkpoint_callback.lastmodelpath 
                    # profiler='advanced',  # this awesome profiler is easy to use
                    )

    if args.eval:
        # Loading the best model checkpoint from args.logpath
        modelpath = checkpoint_callback.modelpath
        model = modelclass.load_from_checkpoint(modelpath, args=args)
        trainer.test(model, test_dataloaders=dm.test_dataloader())
    else:
        # Train
        model = modelclass(args=args)
        # dm is the train_dataloaders
        trainer.fit(model, dm, ckpt_path=checkpoint_callback.lastmodelpath)


if __name__ == '__main__':

    # Arguments parsing
    # from the labL E:\\Integrative-Few-Shot-Learning-for-Classification-and-Segmentation\\datasets\\pascal\VOCdevkit
    # /Users/maxxyouu/Desktop/Integrative-Few-Shot-Learning-for-Classification-and-Segmentation/datasets/pascal/VOCdevkit
    parser = argparse.ArgumentParser(description='Methods for Integrative Few-Shot Classification and Segmentation')
    parser.add_argument('--datapath', type=str,
                        default='/Users/maxxyouu/Desktop/Integrative-Few-Shot-Learning-for-Classification-and-Segmentation/datasets/pascal/VOCdevkit', 
                        help='Dataset path containing the root dir of pascal & coco')
    parser.add_argument('--method', type=str, default='hsnet', choices=['panet', 'pfenet', 'hsnet', 'asnet', 'asnethm', 'universeg'], help='FS-CS methods')
    parser.add_argument('--benchmark', type=str, default='pascal', choices=['pascal', 'coco'], help='Experiment benchmark')
    parser.add_argument('--logpath', type=str, default='', help='Checkpoint saving dir identifier')
    parser.add_argument('--way', type=int, default=1, help='N-way for K-shot evaluation episode')
    parser.add_argument('--shot', type=int, default=1, help='K-shot for N-way K-shot evaluation episode: fixed to 1 for training')
    parser.add_argument('--bsz', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--niter', type=int, default=2000, help='Max iterations')
    parser.add_argument('--fold', type=int, default=0, choices=[0, 1, 2, 3], help='4-fold validation fold')
    parser.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'resnet101'], help='Backbone CNN network')
    parser.add_argument('--nowandb', action='store_true', help='Flag not to log at wandb')
    parser.add_argument('--eval', action='store_true', help='Flag to evaluate a model checkpoint')
    parser.add_argument('--weak', action='store_true', help='Flag to train with cls (weak) labels -- reduce learning rate by 10 times')
    parser.add_argument('--resume', action='store_true', help='Flag to resume a finished run')
    parser.add_argument('--vis', action='store_true', help='Flag to visualize. Use with --eval')
    args = parser.parse_args()
    args.nowandb = True
    args.weak = False
    main(args)
