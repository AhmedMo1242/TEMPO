import argparse


def get_parser():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description="The pytorch implementation for Visual Alignment Constraint "
        "for Continuous Sign Language Recognition."
    )
    parser.add_argument(
        "--work-dir",
        default="./work_dir/test/",
        help="the work folder for storing results",
    )
    parser.add_argument(
        "--config",
        default="./configs/baseline.yaml",
        help="path to the configuration file",
    )
    parser.add_argument(
        "--random_fix", type=str2bool, default=True, help="fix random seed or not"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=0,
        help="the indexes of GPUs for training or testing",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=None,
        help="this arg is only used to test utilization of different signals",
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # processor
    parser.add_argument(
        "--phase", default="train", help="can be train, test and features"
    )

    parser.add_argument("--train_args", default=dict(), help="clip_grad or anneal")

    parser.add_argument(
        "--save-interval",
        type=int,
        default=200,
        help="the interval for storing models (#epochs)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=0, help="the default value for random seed."
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="the interval for evaluating models (#epochs)",
    )
    parser.add_argument(
        "--print-log", type=str2bool, default=True, help="print logging or not"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="the interval for printing messages (#iteration)",
    )
    parser.add_argument("--evaluate-tool", default="python", help="sclite or python")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # feeder
    parser.add_argument(
        "--feeder",
        default="dataloader_video.BaseFeeder",
        help="data loader will be used",
    )
    parser.add_argument("--dataset", default=None, help="data loader will be used")
    parser.add_argument(
        "--dataset-part",
        type=str,
        default="all",
        choices=["all", "first_half", "second_half"],
        help="Train on specific part of the dataset: all, first_half, or second_half",
    )
    parser.add_argument(
        "--train-max-batches",
        type=int,
        default=None,
        help="stop training after this many batches per epoch (useful for debugging)",
    )
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="limit evaluation to this many batches (useful for quick checks)",
    )
    parser.add_argument(
        "--num-worker", type=int, default=4, help="the number of worker for data loader"
    )
    parser.add_argument(
        "--feeder-args", default=dict(), help="the arguments of data loader"
    )
    # batch sizes
    parser.add_argument(
        "--batch-size", type=int, default=16, help="training batch size"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=8, help="test batch size"
    )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # model
    parser.add_argument("--model", default=None, help="the model will be used")
    parser.add_argument(
        "--model-args", type=dict, default=dict(), help="the arguments of model"
    )
    parser.add_argument(
        "--load-weights", default=None, help="load weights for network initialization"
    )
    parser.add_argument(
        "--load-checkpoints",
        default=None,
        help="load checkpoints for continue training",
    )
    parser.add_argument(
        "--ignore-weights",
        type=str,
        default=[],
        nargs="+",
        help="the name of weights which will be ignored in the initialization",
    )
    parser.add_argument(
        "--freeze-layers",
        type=str,
        default=[],
        nargs="+",
        help="prefix patterns of parameter names to freeze (require_grad=False). "
        "E.g. 'visual_module' will freeze the entire CoSign2s backbone.",
    )
    parser.add_argument(
        "--reset-optimizer",
        type=str2bool,
        default=False,
        help="When True, discard checkpoint optimizer/scheduler state and start fresh. "
        "Useful for fine-tuning runs where LR is intentionally changed.",
    )
    parser.add_argument(
        "--test-trigger",
        type=float,
        default=15.0,
        help="Dev WER threshold (%) below which test-set inference is run automatically.",
    )
    parser.add_argument(
        "--eval-start-epoch",
        type=int,
        default=0,
        help="Wait until this epoch before starting evaluation on dev set.",
    )

    default_optimizer_dict = {
        "base_lr": 1e-2,
        "optimizer": "SGD",
        "nesterov": False,
        "step": [5, 10],
        "weight_decay": 0.00005,
        "start_epoch": 1,
    }
    default_loss_dict = {
        "SeqCTC": 1.0,
    }

    parser.add_argument(
        "--optimizer-args",
        default=default_optimizer_dict,
        help="the arguments of optimizer",
    )

    parser.add_argument(
        "--num-epoch", type=int, default=80, help="stop training in which epoch"
    )
    return parser


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
