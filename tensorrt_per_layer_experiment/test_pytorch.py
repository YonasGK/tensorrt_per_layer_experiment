import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt 
import conv_CPU_GPU_pytorch as dla
import cProfile, pstats, io


model_custom= dla.custom_conv2d(in_channel=16,out_channel=16, kernel_shape=(3,3), stride= (1,1), padding=(1,1)).cpu()

loss_fn = torch.nn.MSELoss(reduction='sum').cpu()
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model_custom.parameters(), lr=learning_rate)
model_custom.train()
ground_truth=torch.rand(32,16,32,32).cpu()
for i in range(0,50):
    inputs=torch.rand((32,16,32, 32), requires_grad = True).cpu()
    output=model_custom(inputs)
    loss=loss_fn(output, ground_truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



#pr=cProfile.Profile()
#pr.enable()
for i in range(0,100):
    inputs=torch.rand((32,16, 32, 32), requires_grad = True).cpu()
    output=model_custom(inputs)
    loss=loss_fn(output, ground_truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#pr.disable()
#pr.print_stats(sort=2)
