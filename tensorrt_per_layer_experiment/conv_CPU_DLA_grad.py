import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pycuda.driver as cuda
#import pycuda.autoinit
import tensorrt as trt
import time
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common
import warnings

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

"""
A class to create a newtork, populate the network, create an engine, initilize the engine, create a context and do inference on the accelerator 
"""
class use_DLA():
    def __init__(self, input_name="conv", input_shape=(1,28,28), output_channel=0, kernel_shape=(3,3),dtype=trt.float16, stride=(1,1),
            padding=(0,0),dilation=(1,1), weights=None, bias=None, groups=1, grad_weight=False):
        super(use_DLA, self).__init__()
        self.input_name=input_name
        self.input_shape=input_shape
        self.output_channel=output_channel
        self.kernel_shape=kernel_shape
        self.stride=stride
        self.padding=padding
        self.dilation=dilation
        self.dtype=dtype
        self.weights=weights
        self.bias=bias
        self.network=None
        self.sum=0
        self.runs=0
        self.refitter=None
        self.groups=groups
        self.grad_weight=grad_weight
        self.context=self.engine=self.inputs=self.outputs=self.bindings=self.stream=None

    def populate_network(self, network, weights, bias):
        # Configure the forward pass network layers based on the weights provided and the parameters specified.
        input_tensor = network.add_input(name=self.input_name, dtype=self.dtype, shape=trt.DimsCHW(self.input_shape))
        conv1_w = weights.detach().numpy()
        conv1_b=None
        if bias is not None:
            conv1_b = bias.detach().numpy()
        conv1 = network.add_convolution(input=input_tensor, num_output_maps=int(self.output_channel), kernel_shape=trt.DimsHW(self.kernel_shape), kernel=conv1_w, bias=conv1_b)
        conv1.name= "conv_1"
        conv1.stride = self.stride
        conv1.padding= self.padding
        conv1.dilation=self.dilation
        conv1.num_groups=self.groups

        network.mark_output(tensor=conv1.get_output(0))
    def populate_network_deconv(self, network, weights, bias):
         # Configure the back pass input gradient network layers based on the weights provided and the parameters specified.
        input_tensor = network.add_input(name=self.input_name, dtype=self.dtype, shape=trt.DimsCHW(self.input_shape))
        conv1_w = weights.detach().numpy()
        conv1_b=None
        if bias is not None:
            conv1_b = bias.detach().numpy()
        conv1 = network.add_deconvolution(input=input_tensor, num_output_maps=int(self.output_channel), kernel_shape=trt.DimsHW(self.kernel_shape), kernel=conv1_w, bias=conv1_b)
        conv1.name= "conv_1"
        conv1.stride = self.stride
        conv1.post_padding=self.dilation
        conv1.padding= self.padding
        conv1.num_groups=self.groups
        network.mark_output(tensor=conv1.get_output(0))

    def build_engine(self,weights, bias):
        # build an with configuration engine for the forward pass
        with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network() as network:

            builder.max_batch_size = 32
            builder.refittable= True
            config.default_device_type= trt.DeviceType.GPU
            config.set_flag(trt.BuilderFlag.REFIT)
            config.set_flag(trt.BuilderFlag.FP16)
            config.max_workspace_size = 1 << 30
            self.populate_network(network, weights, bias)
            self.network=network
            return builder.build_engine(network, config)

    def build_engine_back(self,weights, bias):
        # build an with configuration engine for the back pass
        with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network() as network:
            # if the engien is for weight gradient computation set the batch size as 1
            if self.grad_weight == True:
                builder.max_batch_size = 1
            else:
                builder.max_batch_size = 32
            builder.refittable =True
            config.default_device_type= trt.DeviceType.GPU
            config.max_workspace_size= 1 << 30
            config.set_flag(trt.BuilderFlag.REFIT)
            config.set_flag(trt.BuilderFlag.FP16)
            if self.grad_weight==True:
                self.populate_network(network, weights, bias)
            else:
                self.populate_network_deconv(network, weights, bias)
            self.network=network
            # Build and return an engine.
            return builder.build_engine(network, config)

    # Loads input from cpu
    def load_input(self, pagelocked_buffer, inputs_cpu):
        img=inputs_cpu.flatten()
        np.copyto(pagelocked_buffer,img)
    # intialize engine: build engine, allocate buffers and  create execution context object for the forward pass engine.
    def initialize_engine(self):
        self.engine=self.build_engine(self.weights, self.bias)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context=self.engine.create_execution_context()

    # intialize engine: build engine, allocate buffers and create execution context object for the forward pass engine.
    def initialize_engine_back(self):
        self.engine=self.build_engine_back(self.weights, self.bias)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context=self.engine.create_execution_context()

    #refit the new update weight onto the engine and do conv operation in the forward pass
    def forward(self, weights, bias,inputs_cpu):

        output=None
        #start=time.time()
        with trt.Refitter(self.engine, TRT_LOGGER) as refitter:
            refitter.set_weights("conv_1", trt.WeightsRole.KERNEL, weights.detach().numpy())
            if bias is not None:
                refitter.set_weights("conv_1", trt.WeightsRole.BIAS, bias.detach().numpy())
            assert refitter.refit_cuda_engine()
            #end=time.time()
            self.load_input(self.inputs[0].host, inputs_cpu)
            [output] = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, batch_size=inputs_cpu.shape[0])
            self.stream.synchronize()
            #if self.runs>5:
            #    self.sum+=end-start
            #self.runs+=1
            #if self.runs==25:
               # print(self.sum)
        return output
    #refit the new update weight onto the engine and do conv operation in the backward pass
    def back(self, weights, bias, inputs_cpu):
        output=None
        #start=time.time()
        with trt.Refitter(self.engine, TRT_LOGGER) as refitter:
            refitter.set_weights("conv_1", trt.WeightsRole.KERNEL, weights.detach().numpy())
            assert refitter.refit_cuda_engine()
            #end=time.time()

            self.load_input(self.inputs[0].host, inputs_cpu)
            [output] = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream, batch_size=inputs_cpu.shape[0])
            self.stream.synchronize()
            #if self.runs>5:
            #    self.sum+=end-start
            #self.runs+=1
            #if self.runs==25:
                #print(self.sum)
        return output


"""
A custom autograd module to maintain the dynamic computational graph for Pytorch
"""
class use_DLA_autograd(torch.autograd.Function):

    @staticmethod
    #save input, weight tensors and parameters for back prop, do conv operation, reshape the 2d output to 4d and return
    def forward(ctx, self, inputs_cpu, weights,bias, model, padding, stride, stage, groups):
        ctx.save_for_backward(inputs_cpu, weights)
        ctx.padding=padding
        ctx.stride=stride
        ctx.stage=stage
        ctx.self=self
        ctx.groups=groups
        results=torch.tensor(model.forward(weights, bias, inputs_cpu))
        h=int(torch.sqrt(torch.tensor(results.shape[0]/(inputs_cpu.shape[0] * weights.shape[0] ))))
        outs=results.reshape(inputs_cpu.shape[0], weights.shape[0], h,h)
        return outs
    @staticmethod
    # obtain saved tensors and parameters from the forward pass and compute input and weight gradients
    def backward(ctx, grad_out):
        inputs,weights=ctx.saved_tensors
        padding=ctx.padding
        stride=ctx.stride
        stage=ctx.stage
        self=ctx.self
        groups=ctx.groups
        x_grad=w_grad=None
        #if this is the first iteration create back prop computating objects
        if stage==1:
            self.grad=Conv2d_DLA_grad(input_name="conv", inputs=inputs, grad_out=grad_out, weight=weights,dtype=trt.float16, stride=stride, padding=padding, groups=groups)
        #compute input gradient
        if ctx.needs_input_grad[1]:
            x_grad=self.grad.grad_input(grad_out, weights)
            #x_grad = torch.nn.grad.conv2d_input(inputs.shape, weights, grad_out, stride=stride, padding=padding)
        #compute weight gradient
        if ctx.needs_input_grad[2]:
            #w_grad = torch.nn.grad.conv2d_weight(inputs, weights.shape, grad_out, stride=stride, padding=padding)
            w_grad=self.grad.grad_weight(inputs, grad_out)
        return None, x_grad, w_grad, None,None, None, None, None, None

"""
A custom convolution layer module whose operation is carried out on inference accelerator

"""
class Conv_2d_DLA(nn.Module):
    def __init__(self, input_name="conv", in_channel=1, output_channel=0, kernel_shape=(3,3),dtype=trt.float16, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=False):
         super(Conv_2d_DLA,self).__init__()
         self.stages=0
         self.inputs_shape=None
         self.in_channel=in_channel
         self.out_channel=output_channel
         self.kernel_shape=kernel_shape
         self.dtype=dtype
         self.stride=stride
         self.padding=padding
         self.dilation=dilation
         self.grad=None
         self.groups=groups
         self.weights=torch.nn.Parameter(torch.randn(output_channel, in_channel,kernel_shape[0], kernel_shape[1]), requires_grad=True)
         self.bias=torch.rand(output_channel)
         self.conv_vanilla=None
         #a pytorch conv layer for the first iteration to obtain input shape
    def forward(self, inputs_cpu):
        out=None
        #On the first iteration do the operation on CPU and obtain the input shape needed to create a TensorRT conv network
        if self.stages==0:
            self.conv_vanilla=nn.Conv2d(self.in_channel, self.out_channel, kernel_size=self.kernel_shape[0], stride=self.stride[0], padding=self.padding[0], groups=self.groups ,bias=False).to("cpu")
            out=self.conv_vanilla(inputs_cpu)
            self.inputs_shape=inputs_cpu.shape
            self.stages=1
        #On the second iteration create tensorRT objects and do the conv operation applying the custom autograd module
        elif self.stages==1:
            self.weights=self.conv_vanilla.weight
            self.bias=self.conv_vanilla.bias
            self.model=use_DLA("conv",(self.inputs_shape[1:4]),self.out_channel, self.kernel_shape, self.dtype, self.stride, self.padding, self.dilation,
                    self.weights, self.bias, self.groups)
            self.model.initialize_engine()
            out = use_DLA_autograd.apply(self, inputs_cpu, self.weights, self.bias, self.model, self.padding, self.stride, self.stages,self.groups)
            self.stages=2
        #For the rest of the iterations just call custom autograd module with the new inputs and updated weights
        else:
            out = use_DLA_autograd.apply(self, inputs_cpu, self.weights, self.bias, self.model, self.padding, self.stride, self.stages,self.groups)

        return out

"""
A custom module to compute input and weight gradients on Accelerator by using TensorRT APIs

"""
class Conv2d_DLA_grad(nn.Module):
        def __init__(self, input_name="conv", inputs=None, grad_out=None, weight=None,dtype=trt.float16, stride=(1,1), padding=(0,0), groups=1):
            super(Conv2d_DLA_grad, self).__init__()
            self.inputs=inputs
            self.grad_out=grad_out
            self.weight=weight
            self.kernel_shape=trt.DimsHW(weight.shape[2], weight.shape[3])
            self.padding=padding
            self.stride=stride
            self.groups=groups
            self.bias=None
            pad_size=int(weight.shape[2]-1)
            #post padding only on the bottom and right edges to compensate for stride > 1
            post_pad= inputs.shape[2] - (stride[0]*(grad_out.shape[2]-1) + weight.shape[2]- 2*padding[0])
            if post_pad<0:
                post_pad=0
            #if post_pad > weight.shape[2]:
            #    post_pad=weight.shape[2]
            self.pad=(0,post_pad, 0, post_pad)
            self.post_pad=(post_pad, post_pad)
            print(post_pad)
            #prepare the grad out to have a shape of (input channel * output channel * batch size, 1 , output height, output width)
            self.grad_out_padded=F.pad(grad_out, self.pad).contiguous()
            self.grad_out1 = self.grad_out.contiguous().repeat(1,self.inputs.shape[1]//self.groups , 1, 1)
            self.grad_out2= self.grad_out1.contiguous().view( self.grad_out1.shape[0] * self.grad_out1.shape[1],
                    1, self.grad_out1.shape[2], self.grad_out1.shape[3])
            #tensorRT deconvolution object to compute input gradient 
            self.grad_in=use_DLA("conv",trt.DimsCHW(self.grad_out_padded.shape[1:4]),self.inputs.shape[1],
                    self.kernel_shape, trt.float16, self.stride, self.padding, self.post_pad, self.weight, self.bias, self.groups)
            #tensorRT convolution object to comput weight gradient
            self.grad_w=use_DLA("conv",trt.DimsCHW(self.inputs.shape[0]* self.inputs.shape[1],
                    self.inputs.shape[2], self.inputs.shape[3]), self.grad_out2.shape[0],
                    (self.grad_out2.shape[2],self.grad_out.shape[3]) ,
                    trt.float16, self.stride, self.padding, (1,1), self.grad_out2, self.bias, self.inputs.shape[1] * self.inputs.shape[0], grad_weight=True)

            self.grad_w.initialize_engine_back()
            self.grad_in.initialize_engine_back()
        #compute weight gradient, implementation based on  torch.nn.grad.conv2d_weight()
        def grad_weight(self, inputs, grad_out):
            grad_out1 = grad_out.contiguous().repeat(1,inputs.shape[1]//self.groups , 1, 1)
            grad_out2= grad_out1.contiguous().view( grad_out1.shape[0] * grad_out1.shape[1],
                    1, grad_out1.shape[2], grad_out1.shape[3])
            inputs = inputs.contiguous().view(1, inputs.shape[0] * inputs.shape[1],
                                    inputs.shape[2],  inputs.shape[3])

            grad_weight=torch.tensor(self.grad_w.back(grad_out2, self.bias, inputs))
            h=int(torch.sqrt(torch.tensor(grad_weight.shape[0]/(inputs.shape[0] * grad_out2.shape[0] ))))

            grad_weight = grad_weight.contiguous().view(
                        self.inputs.shape[0], grad_weight.shape[0] // (self.inputs.shape[0]*h*h),
                        h,  h)

            return  grad_weight.sum(dim=0).view(
                    self.inputs.shape[1]// self.groups, self.grad_out.shape[1],  grad_weight.shape[2], grad_weight.shape[3]).transpose(0, 1).narrow(
                            2, 0, self.weight.shape[2]).narrow(3, 0, self.weight.shape[3])
        # compute input gradient 
        def grad_input(self, grad_out, weights):
            #post padding the grad_out to compensate for cases stride > 1
            grad_out_padded=F.pad(grad_out, self.pad).contiguous()
            results=torch.tensor(self.grad_in.back(weights, self.bias, grad_out_padded))
            #compute the output height and width
            h=int(torch.sqrt(torch.tensor(results.shape[0]/(grad_out.shape[0] * self.inputs.shape[1] ))))
            #reshape and throw away unnecessary bottom rows and right columns
            outs=results.reshape(grad_out.shape[0], self.inputs.shape[1],h,h).narrow(2,0,self.inputs.shape[2]).narrow(3,0,self.inputs.shape[3])

            return outs

