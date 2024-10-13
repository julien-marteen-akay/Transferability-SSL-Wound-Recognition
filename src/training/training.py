# external packages
import os
import sys
import argparse
import pandas as pd
from pathlib import Path
from functools import partial
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

# internals
from ..datasets.core import DataModule
from ..models.classifier import Classifier
from ..utils import argparse_helpers as arghelp
from ..utils.project_structure import create_experiment_folder


class Constants:
    project_path = Path(sys.argv[0]).parents[1]
    save_path = os.path.join(project_path.parents[0], "out")  # from now on the experiments are pushed into the folder "out/<project_name>" which is on the same level as the project folder


def get_callbacks(path, stop_patience: int = 30):
    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(path, 'models') + "/", monitor="val_loss", verbose=True),
        EarlyStopping(monitor="val_loss", patience=stop_patience, verbose=True)
        ]
    return callbacks


def get_args():
    parser = argparse.ArgumentParser(description="Program for running classification experiments, allowing to use pretrained backbones for linear probing, etc.")
    parser.add_argument("--custom_name", default=False,
                        help="specify a custom name for this experiment, if not specified it will be "
                             "a datetime stamp of when the experiment started")
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed is used everywhere throughout the project, so it makes sense to set it differently to influence datasets, learnable param inits, etc.")
    # ---------------------------
    # Lightning Trainer arguments
    # ---------------------------
    # epochs
    parser.add_argument("--epochs", default=300, type=int,
                        help="Number of epochs.")
    parser.add_argument("--min_epochs", default=60, type=int,
                        help="Minimum number of epochs to run.")
    # performance
    parser.add_argument("--gpu", type=arghelp.gpu_from_string, default="auto",
                        help=("""
                        gpu (Union[List[int], str, int])
                        The GPUs to use.
                        Can be set to
                        - A positive number specifying the *amount* of GPUs to be used.
                        - A sequence of device indices (list-like, i.e. '[0, 1]' for the two GPUs indexed by 0 and 1). White-space is removed automatically.
                        - The value '-1' to indicate all available GPUs should be used.
                        - "auto" for automatic selection based on the chosen accelerator (default).
                        """)
                    )
    parser.add_argument("--precision", type=str, default="32-true",
                        help=(
            """
            precision (Union[Literal[64, 32, 16], Literal['16-mixed', 'bf16-mixed', '32-true', '64-true'], Literal['64', '32', '16', 'bf16']])
            - double precision (64, '64' or '64-true')
            - full precision (32, '32' or '32-true')
            - 16bit mixed precision (16, '16', '16-mixed') or bfloat16 mixed precision ('bf16', 'bf16-mixed').
            Can be used on CPU, GPU, TPUs, HPUs or IPUs. Default: '32-true'.
            """)
            )
    # debugging
    parser.add_argument("--fast_dev_run", type=arghelp.boolean_string, default=False,
                        help="Works like a compiler. Does not capture logs, save checkpoints, etc. but touches every line of code.")
    parser.add_argument("--overfit_batches", type=int, default=0, 
                        help="Overfits the specified amount of batches, for the user to see if the constructed model is actually capable of learning.")
    # --------------------
    # Datamodule arguments
    # --------------------
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="The name of the dataset. Currently only supports 'medetec' and 'azh'. Soon to come: 'connext'.")
    # batch sizes
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for the experiment. Applied to the training dataset.")
    parser.add_argument("--val_batch_size", type=int, default=32,
                        help="Batch size for the experiment. Applied to the validation dataset.")
    parser.add_argument("--test_batch_size", type=int, default=32,
                        help="Batch size for the experiment. Applied to the test dataset.")
    # splits
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Batch size for the experiment. Applied to the validation dataset.")
    parser.add_argument("--test_split", type=float, default=0.2,
                        help="Batch size for the experiment. Applied to the test dataset.")
    # preprocessing
    parser.add_argument("--keep_classes", type=partial(arghelp.to_list, strip_whitespace=False), default=None,
                        help="""
                        Only keep samples belonging to the specified classes.
                        Must be list-like, i.e. providing --keep_classes [classA,someotherclass,X] would lead to only keeping the samples
                        in the dataset belonging to the classes with names 'classA', 'someotherclass', 'X'. Seperate with commas (whitespace sensitive).
                        For Medetec you may want to try: [pressure-ulcer,leg-ulcer-images,foot-ulcers]
                        """)
    parser.add_argument("--min_samples", type=int, default=0,
                        help="Discard samples, whose class appears less than min_samples times in the whole dataset.")
    parser.add_argument("--augmentation", type=arghelp.boolean_string, default=True, required=True,
                        help="Whether to apply augmentation or not. Only affects the training dataset. 'True' or 'False', string is lowered automatically.")
    parser.add_argument("--augmentation_p", type=float, default=0.05,
                        help="Probability of each random augmentation to be applied.")
    parser.add_argument("--keep_aspect_ratio", type=arghelp.boolean_string, default=False, required=True,
                        help="""
                        Applied to the training dataset only: If true images, where hight != width, are zero-padded to a 
                        squared image (side length = max(hight, width)), such that the aspect ratio is kept when resizing to 224 x224.
                        """)
    parser.add_argument("--keep_aspect_ratio_eval", type=arghelp.boolean_string, default=False, required=True,
                        help="Same as the --keep_aspect_ratio flag, but wrt. to the validation and test datasets.")
    # performance
    parser.add_argument("--num_workers", type=int, default=0,
                        help=(
                        """
                        Number of workers for each dataloader (training, validation, test).
                        You can specify any amount.
                        Special cases:
                        -  0 means to only use the main process (default).
                        -  -1 means to let python count the amount of availabe CPUs and take a fraction of that, precisely: num_workers := os.cpu_count() // 4
                        """))
    # ------------------------------
    # Model-specific hyperparameters
    # ------------------------------
    parser.add_argument("--backbone", type=str, required=True,
                        help="""
                        Provide one of the following strings
                        - 'resnet_imagenet' ... ResNet pretrained on ImageNet classification
                        - 'resnet' ... ResNet architecture, not pretrained.
                        - 'vicreg' ... ResNet pretrained with VICReg
                        - 'vicregl' ... Add to it one of the following strings below, i.e. 'vicregl_convnext_small' means ConvNext (small version) pretrained with VICRegL
                        --> 'resnet' ... Same ResNet as above
                        --> 'convnext_small' ... ConvNext small version
                        --> 'convnext_base' ... ConvNext base (=medium) version
                        --> 'convnext_xlarge' ... ConvNext xlarge version
                        """)
    parser.add_argument("--num_mlp_layers", type=int, default=1, required=True,
                        help="""
                        Specifies how many layers the classification head should have (minimum 1). Select 1 to do linear probing, i.e. have only one linear 
                        layer in the classification head. In that case the number of neurons is set to the number of classes, calculated dynamically 
                        based on the chosen dataset, etc.
                        When providing a value greater than 1, layers are added accordingly and between every pair of layers a ReLU activation function is also applied.
                        """)
    parser.add_argument("--num_mlp_neurons", type=int, default=512,
                        help="Setting this value will define the number of neurons contained in each layer of the classification head. See other flag --num_mlp_layers.")
    parser.add_argument("--freeze_backbone", type=arghelp.boolean_string, default=True, required=True,
                        help="Whether to freeze the backbone or not. If frozen, only the linear classification layer is trained (=linear probing).")
    
    args = parser.parse_args()
    return args


def main(args, path):
    # set seeds for numpy, torch and python.random
    seed_everything(args.seed, workers=True)

    datamodule = DataModule(dataset_name=args.dataset_name,
                            batch_size=args.batch_size,
                            val_batch_size=args.val_batch_size,
                            test_batch_size=args.test_batch_size,
                            val_split=args.val_split,
                            test_split=args.test_split,
                            keep_classes=args.keep_classes,
                            min_samples=args.min_samples,
                            augmentation=args.augmentation,
                            augmentation_p=args.augmentation_p,
                            keep_aspect_ratio=args.keep_aspect_ratio,
                            keep_aspect_ratio_eval=args.keep_aspect_ratio_eval,
                            num_workers=args.num_workers,
                            seed=args.seed
                            )
    # when preparing the data of the datamodule, we can access new information, as the preprocessing is done, i.e. how many classes are left after filtering
    datamodule.prepare_data()
    
    # get model
    model = Classifier(backbone=args.backbone,
                       num_mlp_layers=args.num_mlp_layers,
                       num_classes=datamodule.num_classes,
                       num_mlp_neurons=args.num_mlp_neurons,
                       freeze_backbone=args.freeze_backbone,
                       seed=args.seed
                       )

    # get callbacks and friends
    callbacks = get_callbacks(path)
    logger = TensorBoardLogger(save_dir=os.path.join(path, 'tensorboard'))

    # trainer
    trainer = Trainer(accelerator="gpu", devices=args.gpu, min_epochs=args.min_epochs, max_epochs=args.epochs, fast_dev_run=args.fast_dev_run,
                      callbacks=callbacks, logger=logger, precision=args.precision)
    
    # fit
    trainer.fit(model, datamodule)
    
    # test
    out_dict = trainer.test(dataloaders=datamodule, ckpt_path='best')

    out_dict = {**out_dict[0], **vars(args)}
    s = pd.Series(out_dict, name="param_value")
    s.index.name = "param_name"
    s.to_csv(os.path.join(path, "experiment_info.csv"))


if __name__ == "__main__":
    args = get_args()

    # experiment folder
    path = create_experiment_folder(os.path.join(Constants.save_path, 'experiments', args.dataset_name, args.backbone), custom_name=args.custom_name)

    # run experiment
    main(args, path)
