import tabulate
from collections import defaultdict

import torch
import torch.nn as nn

from lib.data.structures.bounding_box import BoxList
from lib.data.structures.image_list import to_image_list
from lib.utils.misc import logging_rank

from .jit_handles import (addmm_flop_jit, batchnorm_flop_jit, conv_flop_jit,
                          einsum_flop_jit, generic_activation_jit,
                          get_jit_model_analysis, matmul_flop_jit)

_IGNORED_OPS = [
    # 'aten::batch_norm',
    'aten::div',
    'aten::div_',
    'aten::meshgrid',
    'aten::rsub',
    'aten::sub',
    'aten::relu_',
    'aten::add_',
    'aten::mul',
    'aten::add',
    'aten::relu',
    'aten::sigmoid',
    'aten::sigmoid_',
    'aten::sort',
    'aten::exp',
    'aten::mul_',
    'aten::max_pool2d',
    'aten::constant_pad_nd',
    'aten::sqrt',
    'aten::softmax',
    'aten::log2',
    'aten::nonzero_numpy',
    'prim::PythonOp',
    'torchvision::nms',
]

# A dictionary that maps supported operations to their flop count jit handles.
_FLOPS_DEFAULT_SUPPORTED_OPS = {
    'aten::addmm': addmm_flop_jit,
    'aten::_convolution': conv_flop_jit,
    'aten::einsum': einsum_flop_jit,
    'aten::matmul': matmul_flop_jit,
    'aten::batch_norm': batchnorm_flop_jit,
}

_ACTIVS_DEFAULT_SUPPORTED_OPS = {
    "aten::_convolution": generic_activation_jit("conv"),
    "aten::addmm": generic_activation_jit("addmm"),
}


class Analyser:
    """
    Calculate Params & FLOPs & Activations for cls, instance and ssd models.

    Usage:
        from {task}.modeling.model_builder import Generalized_CNN(SSD)
        from lib.utils.analyser import Analyser

        # Calculate Params & FLOPs & Activations
        if cfg.MODEL_ANALYSE.ANALYSE:
            model = Generalized_CNN(is_train=False)
            model.eval()

            analyser = Analyser(cfg, model, param_details=False)

            n_params = analyser.get_params()
            conv_flops, model_flops = analyser.get_flops_activs(input_h, input_w, mode='flops')
            conv_activs, model_activs = analyser.get_flops_activs(input_h, input_w, mode='activations')

            logging_rank('Model Params: {}'.format(n_params[1]))
            logging_rank('FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M'.format(model_flops, conv_flops))
            logging_rank('ACTIVATIONs: {:.4f}M / Conv_ACTIVATIONs: {:.4f}M'.format(model_activs, conv_activs))

            del model
    """

    def __init__(self, cfg, model, param_details=False):
        self.cfg = cfg

        if isinstance(
                model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
        ):
            self.model = model.module
        else:
            self.model = model

        self.param_details = param_details

    def compute_params(self):
        """
        Count parameters of a model and its submodules.

        Returns:
            dict (str-> int): the key is either a parameter name or a module name.
                The value is the number of elements in the parameter, or in all
                parameters of the module. The key '' corresponds to the total
                number of parameters of the model.
        """

        r = defaultdict(int)
        for name, prm in self.model.named_parameters():
            size = prm.numel()
            name = name.split('.')
            for k in range(0, len(name) + 1):
                prefix = '.'.join(name[:k])
                r[prefix] += size
        return r

    def get_params(self, max_depth=6):
        """
        Format the parameter count of the model (and its submodules or parameters)
        in a nice table.

        Args:
            max_depth (int): maximum depth to recursively print submodules or parameters

        Returns:
            str: the table to be printed
        """
        count = self.compute_params()
        param_shape = {
            k: tuple(v.shape) for k, v in self.model.named_parameters()
        }
        table = []

        def format_size(x):
            # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
            # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
            if x > 1e5:
                return '{:.2f}M'.format(x / 1e6)
            # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
            # pyre-fixme[6]: Expected `int` for 1st param but got `float`.
            if x > 1e2:
                return '{:.2f}K'.format(x / 1e3)
            return str(x)

        def fill(lvl, prefix):
            if lvl >= max_depth:
                return
            for name, v in count.items():
                if name.count('.') == lvl and name.startswith(prefix):
                    indent = ' ' * (lvl + 1)
                    if name in param_shape:
                        table.append((indent + name, indent + str(param_shape[name])))
                    else:
                        table.append((indent + name, indent + format_size(v)))
                        fill(lvl + 1, name + '.')

        table.append(('model', format_size(count.pop(''))))
        fill(0, '')

        old_ws = tabulate.PRESERVE_WHITESPACE
        tabulate.PRESERVE_WHITESPACE = True
        tab = tabulate.tabulate(
            table, headers=['name', '#elements or shape'], tablefmt='pipe'
        )
        tabulate.PRESERVE_WHITESPACE = old_ws

        if self.param_details:
            logging_rank(tab)

        return table[0]

    def flop_count(self, model, inputs, supported_ops=None, ):
        """
        Given a model and an input to the model, compute the Gflops of the given
        model.

        Args:
            model (nn.Module): The model to compute flop counts.
            inputs (tuple): Inputs that are passed to `model` to count flops.
                Inputs need to be in a tuple.
            supported_ops (dict(str,Callable) or None) : provide additional
                handlers for extra ops, or overwrite the existing handlers for
                convolution and matmul and einsum. The key is operator name and the value
                is a function that takes (inputs, outputs) of the op. We count
                one Multiply-Add as one FLOP.

        Returns:
            tuple[defaultdict, Counter]: A dictionary that records the number of
                gflops for each operation and a Counter that records the number of
                skipped operations.
        """

        assert isinstance(inputs, tuple), 'Inputs need to be in a tuple.'
        supported_ops = {**_FLOPS_DEFAULT_SUPPORTED_OPS, **(supported_ops or {})}

        # Run flop count.
        total_flop_counter, skipped_ops = get_jit_model_analysis(
            model, inputs, supported_ops
        )

        # Log for skipped operations.
        if len(skipped_ops) > 0:
            for op, freq in skipped_ops.items():
                logging_rank('Skipped operation {} {} time(s)'.format(op, freq))

        # Convert flop count to gigaflops.
        final_count = defaultdict(float)
        for op in total_flop_counter:
            final_count[op] = total_flop_counter[op] / 1e6

        return final_count, skipped_ops

    def activation_count(self, model, inputs, supported_ops=None, ):
        """
        Given a model and an input to the model, compute the total number of
        activations of the model.

        Args:
            model (nn.Module): The model to compute activation counts.
            inputs (tuple): Inputs that are passed to `model` to count activations.
                Inputs need to be in a tuple.
            supported_ops (dict(str,Callable) or None) : provide additional
                handlers for extra ops, or overwrite the existing handlers for
                convolution and matmul. The key is operator name and the value
                is a function that takes (inputs, outputs) of the op.

        Returns:
            tuple[defaultdict, Counter]: A dictionary that records the number of
                activation (mega) for each operation and a Counter that records the
                number of skipped operations.
        """
        assert isinstance(inputs, tuple), "Inputs need to be in a tuple."
        supported_ops = {**_ACTIVS_DEFAULT_SUPPORTED_OPS, **(supported_ops or {})}

        # Run activation count.
        total_activation_count, skipped_ops = get_jit_model_analysis(
            model, inputs, supported_ops
        )

        # Log for skipped operations.
        if len(skipped_ops) > 0:
            for op, freq in skipped_ops.items():
                logging_rank("Skipped operation {} {} time(s)".format(op, freq))

        # Convert activation count to mega count.
        final_count = defaultdict(float)
        for op in total_activation_count:
            final_count[op] = total_activation_count[op] / 1e6

        return final_count, skipped_ops

    def get_flops_activs(self, H, W, mode='flops', **kwargs):
        input_tensor = torch.zeros((1, 3, H, W))

        count_model = self.compute_model_flops_activs(input_tensor, mode, **kwargs)

        conv_count = count_model['conv']
        model_count = 0
        for op in count_model:
            model_count += count_model[op]
        return conv_count, model_count

    def compute_model_flops_activs(self, inputs, mode, **kwargs):
        model = WrapModel(self.cfg, model=self.model)

        model_count = self.wrapper_count_operators(model, inputs, mode=mode, **kwargs)

        return model_count

    def wrapper_count_operators(self, model, image_tensor, mode, **kwargs):

        # ignore some ops
        supported_ops = {k: lambda *args, **kwargs: {} for k in _IGNORED_OPS}
        supported_ops.update(kwargs.pop('supported_ops', {}))
        kwargs['supported_ops'] = supported_ops

        old_train = model.training  # flag whether model is under training mode， False
        with torch.no_grad():
            if mode == 'flops':
                ret = self.flop_count(model.train(False), (image_tensor,), **kwargs)
            elif mode == 'activations':
                ret = self.activation_count(model.train(False), (image_tensor,), **kwargs)
            else:
                raise NotImplementedError('Count for mode {} is not supported yet.'.format(mode))

        if isinstance(ret, tuple):
            ret = ret[0]
        model.train(old_train)
        return ret


def flatten_to_tuple(outputs):
    result = []
    if isinstance(outputs, torch.Tensor):
        result.append(outputs)
    elif isinstance(outputs, (list, tuple)):
        for v in outputs:
            result.extend(flatten_to_tuple(v))
    elif isinstance(outputs, dict):
        for _, v in outputs.items():
            result.extend(flatten_to_tuple(v))
    elif isinstance(outputs, BoxList):
        result.extend(flatten_to_tuple(outputs.bbox))
        if outputs.has_field('grid'):
            result.extend(flatten_to_tuple((outputs.get_field('grid'))))
        if outputs.has_field('mask'):
            result.extend(flatten_to_tuple((outputs.get_field('mask'))))
            result.extend(flatten_to_tuple((outputs.get_field('mask_scores'))))
        if outputs.has_field('keypoints'):
            result.extend(flatten_to_tuple((outputs.get_field('keypoints'))))
        if outputs.has_field('parsing'):
            result.extend(flatten_to_tuple((outputs.get_field('parsing'))))
            result.extend(flatten_to_tuple((outputs.get_field('parsing_scores'))))
        if outputs.has_field('uv'):
            result.extend(flatten_to_tuple((outputs.get_field('uv'))))
        if outputs.has_field('hier'):
            result.extend(flatten_to_tuple((outputs.get_field('hier'))))
    else:
        logging_rank('Output of type {} not included in flops/activations count.'.format(type(outputs)))
    return tuple(result)


class WrapModel(nn.Module):
    def __init__(self, cfg, model=None):
        super().__init__()

        self.cfg = cfg

        if isinstance(
                model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
        ):
            self.model = model.module
        else:
            self.model = model

    def forward(self, images):
        # perform forward computation from backbone until rpn
        # jit requires the input/output to be Tensors
        images = to_image_list(images)
        inputs = images.tensors

        outputs = self.model(inputs)

        # Only the subgraph that computes the returned tuple of tensor will be
        # counted. So we flatten everything we found to tuple of tensors.
        return flatten_to_tuple(outputs)
