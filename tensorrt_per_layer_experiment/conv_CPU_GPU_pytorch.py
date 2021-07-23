import numpy as np
import collections
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pycuda.driver as cuda
from itertools import repeat
import math
import random


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse
_pair= _ntuple(2)
def move_data_input_grad(grad_out, weights):
    grad_out_cuda= grad_out.cuda()
    weights_cuda=weights.cuda()
    return grad_out_cuda, weights_cuda
def move_data_weight_grad(inputs, grad_out):
    inputs_cuda= inputs.cuda()
    grad_out_cuda=grad_out.cuda()
    return inputs_cuda, grad_out_cuda

def conv2d_input_grad(input_shape, weights, grad_out, stride=1, padding=0, groups=1, dilation=1):
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    kernel_size = (weights.shape[2], weights.shape[3])

    if input_shape is None:
        raise ValueError("grad.conv2d_input requires specifying an input_size")

    grad_input_padding = torch.nn.grad._grad_input_padding(grad_out, input_shape, stride,
                                             padding, kernel_size, dilation)

    grad_out_cuda, weights_cuda = move_data_input_grad(grad_out, weights)
    out =torch.conv_transpose2d(
        grad_out_cuda, weights_cuda, None, stride, padding, grad_input_padding, groups,
        dilation)
    return out
def conv2d_weight_grad(inputs, weights_shape, grad_out, stride=1, padding=0, groups=1, dilation=1 ):
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    in_channels = inputs.shape[1]
    out_channels = grad_out.shape[1]
    min_batch = inputs.shape[0]

    grad_out = grad_out.contiguous().repeat(1, in_channels // groups, 1,
                                                  1)
    grad_out = grad_out.contiguous().view(
        grad_out.shape[0] * grad_out.shape[1], 1, grad_out.shape[2],
        grad_out.shape[3])
    inputs = inputs.contiguous().view(1, inputs.shape[0] * inputs.shape[1],
                                    inputs.shape[2], inputs.shape[3])
    inputs_cuda, grad_out_cuda = move_data_weight_grad(inputs, grad_out)
    grad_weight = torch.conv2d(inputs_cuda, grad_out_cuda, None, dilation, padding,
                               stride, in_channels * min_batch)
    #grad_weight_cpu=grad_weight.cpu()
    grad_weight = grad_weight.contiguous().view(
        min_batch, grad_weight.shape[1] // min_batch, grad_weight.shape[2],
        grad_weight.shape[3])

    return grad_weight.sum(dim=0).view(
        in_channels // groups, out_channels,
        grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
            2, 0, weights_shape[2]).narrow(3, 0, weights_shape[3])

def move_data_forward(inputs_cpu, weights):
    inputs_cuda=inputs_cpu.cuda()
    weights_cuda=weights.cuda()
    return inputs_cuda, weights_cuda

def return_result_to_cpu_forward(out):
    return out.cpu()
def return_result_to_cpu_input_grad(x_grad):
    return x_grad.cpu()
def return_result_to_cpu_weight_grad(w_grad):
    return w_grad.cpu()

class custom_conv2d_grad(torch.autograd.Function):
    @staticmethod
    #save input, weight tensors and parameters for back prop, do conv operation, reshape the 2d output to 4d and return
    def forward(ctx, inputs_cpu, weights, model, padding, stride, dilation, groups):
        ctx.save_for_backward(inputs_cpu, weights)
        ctx.padding=padding
        ctx.stride=stride
        ctx.groups=groups
        ctx.dilation=dilation
        
        inputs_cuda, weights_cuda = move_data_forward(inputs_cpu, weights) 
        #model=model.cuda()
        #out=model(inputs_cuda)
        #inputs_cpu = inputs_cuda.cpu()

        #model=model.cpu()
        #print(inputs_cpu.device)
        out=torch.conv2d(inputs_cuda, weights_cuda, None, stride, padding, dilation, groups)
        return return_result_to_cpu_forward(out)
    @staticmethod
    # obtain saved tensors and parameters from the forward pass and compute input and weight gradients
    def backward(ctx, grad_out):
        inputs,weights=ctx.saved_tensors
        padding=ctx.padding
        stride=ctx.stride
        groups=ctx.groups
        dilation=ctx.dilation
        x_grad=w_grad=w_grad_cpu=x_grad_cpu=None
        #compute input gradient
        if ctx.needs_input_grad[0]:
            #x_grad = torch.nn.grad.conv2d_input(inputs.shape, weights, grad_out, stride=stride, padding=padding, dilation=dilation, groups=groups)
            x_grad = conv2d_input_grad(inputs.shape, weights, grad_out, stride=stride, padding=padding, dilation=dilation, groups=groups)
            x_grad_cpu=return_result_to_cpu_input_grad(x_grad)
        #compute weight gradient
        if ctx.needs_input_grad[1]:
            #w_grad = torch.nn.grad.conv2d_weight(inputs, weights.shape, grad_out, stride=stride, padding=padding, dilation=dilation, groups=groups)
            w_grad = conv2d_weight_grad(inputs, weights.shape, grad_out, stride=stride, padding=padding, dilation=dilation, groups=groups)
            w_grad_cpu=return_result_to_cpu_weight_grad(w_grad)
                
        return x_grad_cpu, w_grad_cpu, None, None, None, None, None
class custom_conv2d(nn.Module):
    def __init__(self, module, in_channel=1, out_channel=0, kernel_shape=(3,3), stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
        super(custom_conv2d,self).__init__()
        self.in_channel=in_channel
        self.out_channel=out_channel
        self.padding=padding
        self.stride=stride
        self.groups=groups
        self.dilation=dilation
        self.conv1=module
        self.prev_weight=None
        self.run=0
    def forward(self, inputs):
        
        out= custom_conv2d_grad.apply(inputs, self.conv1.weight, self.conv1, self.padding, self.stride, self.dilation, self.groups)

        return out

