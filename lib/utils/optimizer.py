import torch


class Optimizer(object):
    def __init__(self, model, solver):
        self.model = model
        self.solver = solver
        # lr
        self.base_lr = self.solver.BASE_LR
        self.bias_lr_factor = self.solver.BIAS_LR_FACTOR
        # weight decay
        self.weight_decay = self.solver.WEIGHT_DECAY
        self.weight_decay_norm = self.solver.WEIGHT_DECAY_NORM
        self.weight_decay_bias = self.solver.WEIGHT_DECAY_BIAS
        # momentum
        self.momentum = self.solver.MOMENTUM
        self.optimizer_type = self.solver.OPTIMIZER

        self.norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

    def get_params(self):
        params = []
        memo = set()
        for module in self.model.modules():
            for key, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                lr = self.base_lr
                weight_decay = self.weight_decay
                if isinstance(module, self.norm_module_types):
                    weight_decay = self.weight_decay_norm
                elif key == "bias":
                    lr = self.base_lr * self.bias_lr_factor
                    weight_decay = self.weight_decay_bias
                params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        return params

    def build(self):
        assert self.optimizer_type in ['SGD', 'RMSPROP', 'ADAM']
        params = self.get_params()

        if self.optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(
                params,
                momentum=self.momentum
            )
            # TODO: CLIP_GRADIENTS
        elif self.optimizer_type == 'RMSPROP':
            optimizer = torch.optim.RMSprop(
                params,
                momentum=self.momentum
            )
        elif self.optimizer_type == 'ADAM':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.base_lr
            )
        else:
            optimizer = None
        return optimizer
