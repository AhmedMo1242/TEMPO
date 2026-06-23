import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from temposlr import utils
import numpy as np
from temposlr import modules
import torch
import torch.nn as nn
from temposlr import datasets
import yaml
import json
import faulthandler
from collections import OrderedDict

faulthandler.enable()

from temposlr.loops import seq_train, seq_eval
from temposlr import model as slr_network

# CUDA Optimizations
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()  # Clear CUDA cache at startup


class SLRProcessor(object):
    def __init__(self, arg):
        super().__init__()

        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(
            self.arg.work_dir, self.arg.print_log, self.arg.log_interval
        )
        self.dataset = {}
        self.data_loader = {}

        self.load_dataset_info()
        with open(self.arg.dataset_info["dict_path"], "r") as f:
            self.gloss_dict = json.load(f)
        self.model, self.optimizer = self.loading()
        self.scaler = torch.cuda.amp.GradScaler(init_scale=2048.0)
        self.best_dev_wer = 1000
        self.tasks = self.arg.dataset[-2:]

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open("{}/config.yaml".format(self.arg.work_dir), "w") as f:
            yaml.dump(arg_dict, f)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model = self.build_module(self.arg.model_args)
        model = self.model_to_device(model)

        # Ensure start_epoch exists in optimizer_args
        if "start_epoch" not in self.arg.optimizer_args:
            self.arg.optimizer_args["start_epoch"] = 0

        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            checkpoint = self.load_checkpoint_weights(model)
            if getattr(self.arg, "reset_optimizer", False):
                # Fine-tune mode: fresh optimizer + fresh LR schedule from epoch 0
                self.arg.optimizer_args["start_epoch"] = 0
                print(
                    "[reset_optimizer] Discarding checkpoint optimizer state. "
                    "Starting from epoch 0 with config LR."
                )
            else:
                self.load_optimizer_state(optimizer, checkpoint)

        # ── Freeze layers specified in config ──────────────────────────────────
        freeze_patterns = getattr(self.arg, "freeze_layers", [])
        if freeze_patterns:
            frozen_params = 0
            for name, param in model.named_parameters():
                # Strip DataParallel 'module.' prefix for matching
                match_name = (
                    name[len("module.") :] if name.startswith("module.") else name
                )
                for pattern in freeze_patterns:
                    if match_name.startswith(pattern):
                        param.requires_grad = False
                        frozen_params += param.numel()
                        break
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(
                f"[freeze] Frozen {frozen_params/1e6:.2f}M params | "
                f"Trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M total"
            )
        # ───────────────────────────────────────────────────────────────────────

        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def adjust_state_dict(self, state_dict, model):
        has_module = any(k.startswith("module.") for k in state_dict.keys())
        is_dataparallel = isinstance(model, nn.DataParallel)
        if has_module and not is_dataparallel:
            # Strip module.
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            return new_state_dict
        elif not has_module and is_dataparallel:
            # Add module.
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = "module." + k
                new_state_dict[name] = v
            return new_state_dict
        else:
            return state_dict

    def model_to_device(self, model):
        return self.device.model_to_device(model)

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(
            weight_path, map_location=torch.device(self.device.output_device)
        )
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        state_dict = self.adjust_state_dict(state_dict, model)

        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print("Successfully Remove Weights: {}.".format(w))
                else:
                    print("Can Not Remove Weights: {}.".format(w))
        model.load_state_dict(state_dict, strict=False)

    def load_checkpoint_weights(self, model):
        self.recoder.print_log(f"Resuming from checkpoint: {self.arg.load_checkpoints}")
        _load_kwargs = {"map_location": torch.device(self.device.output_device)}
        if tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (1, 13):
            _load_kwargs["weights_only"] = False
        checkpoint = torch.load(self.arg.load_checkpoints, **_load_kwargs)

        # Load model state
        state_dict = checkpoint["model_state_dict"]
        state_dict = self.adjust_state_dict(state_dict, model)
        # Allow user to explicitly ignore certain checkpoint keys via config
        if hasattr(self.arg, "ignore_weights") and len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    self.recoder.print_log(f"Removed checkpoint weight: {w}")
                else:
                    self.recoder.print_log(
                        f"Checkpoint weight not found to remove: {w}"
                    )

        # Filter out parameters that don't match in shape between checkpoint and model.
        model_state = model.state_dict()
        filtered_state = OrderedDict()
        skipped = []
        for k, v in state_dict.items():
            if k in model_state:
                try:
                    if v.shape == model_state[k].shape:
                        filtered_state[k] = v
                    else:
                        skipped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
                except Exception:
                    # Some entries may not be tensors (buffers etc.) — try to load directly
                    if isinstance(v, torch.Tensor) and isinstance(
                        model_state[k], torch.Tensor
                    ):
                        if v.numel() == model_state[k].numel():
                            # reshape attempt (best-effort)
                            try:
                                filtered_state[k] = v.view(model_state[k].shape)
                            except Exception:
                                skipped.append((k, None, tuple(model_state[k].shape)))
                        else:
                            skipped.append((k, None, tuple(model_state[k].shape)))
                    else:
                        skipped.append((k, None, None))
            else:
                skipped.append(
                    (k, tuple(v.shape) if hasattr(v, "shape") else None, None)
                )

        if skipped:
            for key, src_shape, dst_shape in skipped:
                if dst_shape is None:
                    self.recoder.print_log(
                        f"Skipping checkpoint param (missing in model): {key} -> {src_shape}"
                    )
                else:
                    self.recoder.print_log(
                        f"Skipping checkpoint param (shape mismatch): {key} -> checkpoint {src_shape}, model {dst_shape}"
                    )

        # Load matching parameters only (non-strict to allow missing keys)
        model.load_state_dict(filtered_state, strict=False)

        # Set start epoch for the loop
        self.arg.optimizer_args["start_epoch"] = checkpoint["epoch"] + 1
        self.recoder.print_log(f"Resumed from epoch {checkpoint['epoch']}")
        return checkpoint

    def load_optimizer_state(self, optimizer, checkpoint):
        # Load optimizer and scheduler state if compatible, otherwise reset
        if "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError as err:
                # mismatch between checkpoint and current optimizer groups
                self.recoder.print_log(
                    f"[optimizer] WARNING: could not load optimizer state ({err})."
                    " Starting with fresh optimizer."
                )
        if "scheduler_state_dict" in checkpoint and hasattr(optimizer, "scheduler"):
            try:
                optimizer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            except Exception as err:
                self.recoder.print_log(
                    f"[optimizer] WARNING: could not load scheduler state ({err})."
                    " Scheduler restarted."
                )

        # Move optimizer state to device (if any was loaded)
        for state in optimizer.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device.output_device)

        # Restore Random State
        if "rng_state" in checkpoint and hasattr(self, "rng"):
            self.rng.set_rng_state(checkpoint["rng_state"])

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=(
                self.arg.batch_size if mode == "train" else self.arg.test_batch_size
            ),
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
            pin_memory=True,  # Faster data transfer to GPU
        )

    def build_module(self, args):
        model_class = getattr(slr_network, self.arg.model)
        model = model_class(**args, gloss_dict=self.gloss_dict)
        return model

    def load_data(self):
        print("Loading data")
        self.feeder = getattr(datasets, self.arg.feeder)

        # Load train and dev data. Skip test for now.
        dataset_list = zip(["train", "dev"], [True, False])
        g2i_dict = {k: v["index"] for k, v in self.gloss_dict["gloss2id"].items()}
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = (
                self.arg.feeder_args.copy()
            )  # Use copy to prevent cross-mode pollution
            arg["mode"] = mode
            arg["transform_mode"] = train_flag
            arg["dataset"] = self.arg.dataset

            # Initialize the feeder
            current_ds = self.feeder(gloss_dict=g2i_dict, **arg)

            # Slicing logic for fine-tuning/quick iteration
            if mode == "train" and hasattr(self.arg, "dataset_part"):
                if self.arg.dataset_part != "all":
                    total_size = len(current_ds.inputs_list)
                    half = total_size // 2
                    if self.arg.dataset_part == "first_half":
                        print(
                            f"[info] Slicing dataset: Using FIRST half ({half} samples)"
                        )
                        current_ds.inputs_list = current_ds.inputs_list[:half]
                    elif self.arg.dataset_part == "second_half":
                        print(
                            f"[info] Slicing dataset: Using SECOND half ({total_size - half} samples)"
                        )
                        current_ds.inputs_list = current_ds.inputs_list[half:]

            self.dataset[mode] = current_ds
            self.data_loader[mode] = self.build_dataloader(
                self.dataset[mode], mode, train_flag
            )
        print("Loading data finished.")

    def load_dataset_info(self):
        with open(f"./configs/dataset_configs/{self.arg.dataset}.yaml", "r") as f:
            self.arg.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    def judge_save_eval(self, epoch):
        save_model = (epoch % self.arg.save_interval == 0) and (
            epoch >= 0.5 * self.arg.num_epoch
        )
        # Evaluation enabled after eval_start_epoch
        eval_model = (epoch % self.arg.eval_interval == 0) and (
            epoch >= self.arg.eval_start_epoch
        )
        return save_model, eval_model

    def save_model(self, epoch, save_path):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.optimizer.scheduler.state_dict(),
        }
        if hasattr(self, "rng"):
            state["rng_state"] = self.rng.save_rng_state()
        torch.save(state, save_path)

    def custom_save_model(self, dev_wer, epoch, save_dir):
        # Save as last model every time it's called
        last_path = os.path.join(save_dir, "last_model.pt")
        self.save_model(epoch, last_path)

        # Update best model if improvement found and dev_wer is available
        if dev_wer is not None and dev_wer <= self.best_dev_wer:
            self.best_dev_wer = dev_wer
            best_path = os.path.join(save_dir, "best_model.pt")
            self.save_model(epoch, best_path)
            self.recoder.print_log(f"New best model saved at epoch {epoch}")

    def train(self):
        self.recoder.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))
        for epoch in range(self.arg.optimizer_args["start_epoch"], self.arg.num_epoch):
            save_model, eval_model = self.judge_save_eval(epoch)
            # Prepare training kwargs: allow explicit `train_max_batches` to override
            # any `max_batches` value inside `train_args` to avoid duplicate keywords.
            train_kwargs = (
                self.arg.train_args.copy() if hasattr(self.arg, "train_args") else {}
            )
            if getattr(self.arg, "train_max_batches", None) is not None:
                train_kwargs["max_batches"] = self.arg.train_max_batches

            seq_train(
                self.data_loader["train"],
                self.model,
                self.optimizer,
                self.device,
                epoch,
                self.recoder,
                scaler=self.scaler,
                **train_kwargs,
            )

            dev_error = None
            if eval_model:
                dev_error = self.test("dev", epoch)
                self.recoder.print_log(
                    f"╔══════════════════════════════════════════════╗"
                )
                self.recoder.print_log(
                    f"║  EPOCH {epoch:3d}  │  Val WER: {dev_error:6.2f}%"
                    + f"  │  Best: {self.best_dev_wer:6.2f}%  ║"
                )
                self.recoder.print_log(
                    f"╚══════════════════════════════════════════════╝"
                )

            # clear cuda cache periodically to avoid fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if save_model:
                dev_wer = dev_error if eval_model else None
                self.custom_save_model(dev_wer, epoch, self.arg.work_dir)

    def test(self, mode, epoch):
        wer = seq_eval(
            self.arg,
            self.data_loader[mode],
            self.model,
            self.device,
            mode,
            epoch,
            self.arg.work_dir,
            self.recoder,
            self.tasks,
            self.arg.evaluate_tool,
            max_batches=self.arg.eval_max_batches,
        )
        return wer

    def start(self):
        if self.arg.phase == "train":
            self.train()
        elif self.arg.phase == "test":
            self.recoder.print_log("Model:   {}.".format(self.arg.model))
            self.recoder.print_log("Weights: {}.".format(self.arg.load_weights))
            dev_wer = self.test("dev", 0)
            self.recoder.print_log("Dev WER (best of 3): {:05.2f}%".format(dev_wer))


if __name__ == "__main__":
    sparser = utils.get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, "r") as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print("WRONG ARG: {}".format(k))
                assert k in key
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()

    main_processor = SLRProcessor(args)
    main_processor.start()
