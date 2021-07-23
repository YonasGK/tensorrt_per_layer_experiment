import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt 
import conv_CPU_GPU_pytorch as dla
import cProfile, pstats, io


model = nn.Conv2d(16, 32, 3, stride = 2, padding = 1).cpu()
model_custom= dla.custom_conv2d(model, in_channel=16,out_channel=32, kernel_shape=(3,3), stride= (2,2), padding=(1,1))

loss_fn = torch.nn.MSELoss(reduction='sum').cpu()
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

model.train()
ground_truth=torch.rand(32,32,16,16).cpu()
for i in range(0,50):
    inputs=torch.rand((32,16,32, 32), requires_grad = True).cpu()
    output=model_custom(inputs)
    loss=loss_fn(output, ground_truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



pr=cProfile.Profile()
pr.enable()
for i in range(0,100):
    inputs=torch.rand((32,16, 32, 32), requires_grad = True).cpu()
    output=model_custom(inputs)
    loss=loss_fn(output, ground_truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
pr.disable()
pr.print_stats(sort=2)
