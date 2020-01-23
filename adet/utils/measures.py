# coding: utf-8
# Adapted from https://github.com/ShichenLiu/CondenseNet/blob/master/utils.py
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import operator

from functools import reduce


def get_num_gen(gen):
    return sum(1 for x in gen)


def is_pruned(layer):
    try:
        layer.mask
        return True
    except AttributeError:
        return False


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, *args):
    global count_ops, count_params

    for x in args:
        delta_ops = 0
        delta_params = 0
        multi_add = 1
        type_name = get_layer_info(layer)

        ### ops_conv
        if type_name in ['Conv2d']:
            out_h = int((x.size()[2] + 2 * layer.padding[0] / layer.dilation[0] - layer.kernel_size[0]) /
                        layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] / layer.dilation[1] - layer.kernel_size[1]) /
                        layer.stride[1] + 1)
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            delta_params = get_layer_param(layer)

        elif type_name in ['ConvTranspose2d']:
            _, _, in_h, in_w = x.size()
            out_h = int((in_h-1)*layer.stride[0] - 2 * layer.padding[0] + layer.kernel_size[0] + layer.output_padding[0])
            out_w = int((in_w-1)*layer.stride[1] - 2 * layer.padding[1] + layer.kernel_size[1] + layer.output_padding[1])
            delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                        layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            delta_params = get_layer_param(layer)

        ### ops_learned_conv
        elif type_name in ['LearnedGroupConv']:
            measure_layer(layer.relu, x)
            measure_layer(layer.norm, x)
            conv = layer.conv
            out_h = int((x.size()[2] + 2 * conv.padding[0] - conv.kernel_size[0]) /
                        conv.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * conv.padding[1] - conv.kernel_size[1]) /
                        conv.stride[1] + 1)
            delta_ops = conv.in_channels * conv.out_channels * conv.kernel_size[0] * conv.kernel_size[1] * out_h * out_w / layer.condense_factor * multi_add
            delta_params = get_layer_param(conv) / layer.condense_factor

        ### ops_nonlinearity
        elif type_name in ['ReLU', 'ReLU6']:
            delta_ops = x.numel()
            delta_params = get_layer_param(layer)

        ### ops_pooling
        elif type_name in ['AvgPool2d', 'MaxPool2d']:
            in_w = x.size()[2]
            kernel_ops = layer.kernel_size * layer.kernel_size
            out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
            out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
            delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
            delta_params = get_layer_param(layer)

        elif type_name in ['LastLevelMaxPool']:
            pass

        elif type_name in ['AdaptiveAvgPool2d']:
            delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
            delta_params = get_layer_param(layer)

        elif type_name in ['ZeroPad2d', 'RetinaNetPostProcessor']:
            pass
            #delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
            #delta_params = get_layer_param(layer)

        ### ops_linear
        elif type_name in ['Linear']:
            weight_ops = layer.weight.numel() * multi_add
            bias_ops = layer.bias.numel()
            delta_ops = x.size()[0] * (weight_ops + bias_ops)
            delta_params = get_layer_param(layer)

        ### ops_nothing
        elif type_name in ['BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'FrozenBatchNorm2d', 'GroupNorm']:
            delta_params = get_layer_param(layer)

        elif type_name in ['SumTwo']:
            delta_ops = x.numel()

        elif type_name in ['AggregateCell']:
            if not layer.pre_transform:
                delta_ops = 2 * x.numel() # twice for each input
            else:
                measure_layer(layer.branch_1, x)
                measure_layer(layer.branch_2, x)
                delta_params = get_layer_param(layer)

        elif type_name in ['Identity', 'Zero']:
            pass

        elif type_name in ['Scale']:
            delta_params = get_layer_param(layer)
            delta_ops = x.numel()

        elif type_name in ['FCOSPostProcessor', 'RPNPostProcessor', 'KeypointPostProcessor',
                           'ROIAlign', 'PostProcessor', 'KeypointRCNNPredictor', 
                           'NaiveSyncBatchNorm', 'Upsample', 'Sequential']:
            pass

        elif type_name in ['DeformConv']:
            # don't count bilinear
            offset_conv = list(layer.parameters())[0]
            delta_ops = reduce(operator.mul, offset_conv.size(), x.size()[2] * x.size()[3])
            out_h = int((x.size()[2] + 2 * layer.padding[0] / layer.dilation[0]
                         - layer.kernel_size[0]) / layer.stride[0] + 1)
            out_w = int((x.size()[3] + 2 * layer.padding[1] / layer.dilation[1]
                         - layer.kernel_size[1]) / layer.stride[1] + 1)
            delta_ops += layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
            delta_params = get_layer_param(layer)

        ### unknown layer type
        else:
            raise TypeError('unknown layer type: %s' % type_name)

        count_ops += delta_ops
        count_params += delta_params
    return


def measure_model(model, x):
    global count_ops, count_params
    count_ops = 0
    count_params = 0

    def should_measure(x):
        return is_leaf(x) or is_pruned(x)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(*args):
                        measure_layer(m, *args)
                        return m.old_forward(*args)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            else:
                restore_forward(child)

    modify_forward(model)
    out = model.forward(x)
    restore_forward(model)

    return out, count_ops, count_params
