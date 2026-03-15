import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        self.gradient_accumulation_steps = optim_dict.get(
            "gradient_accumulation_steps", 1
        )
        self.step_count = 0

        if self.optim_dict["optimizer"] == "SGD":
            self.optimizer = optim.SGD(
                model,
                lr=self.optim_dict["base_lr"],
                momentum=0.9,
                nesterov=self.optim_dict["nesterov"],
                weight_decay=self.optim_dict["weight_decay"],
            )
        elif self.optim_dict["optimizer"] == "Adam":
            alpha = self.optim_dict["learning_ratio"]
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=self.optim_dict["base_lr"],
                weight_decay=self.optim_dict["weight_decay"],
            )
        elif self.optim_dict["optimizer"] == "AdamW":
            alpha = self.optim_dict["learning_ratio"]
            self.optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.optim_dict["base_lr"],
                weight_decay=self.optim_dict["weight_decay"],
                eps=1e-6,  # Increased eps for AMP stability
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(
            self.optimizer, self.optim_dict.get("step", [])
        )

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict.get("scheduler") == "cosine":
            # Use num_epoch from config, default to 30 for memory_safe config
            num_epochs = self.optim_dict.get("num_epoch", 30)
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs, eta_min=1e-6
            )

        if self.optim_dict["optimizer"] in ["SGD", "Adam", "AdamW"]:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=0.1
            )
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        # Only zero gradients at the start of accumulation cycle
        if self.step_count % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        self.step_count += 1
        if self.step_count % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def to(self, device):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
