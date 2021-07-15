import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorrt as trt 
import conv_CPU_DLA_grad as dla
import cProfile, pstats, io


model= dla.Conv_2d_DLA(in_channel=64,output_channel=64, kernel_shape=(3,3), stride= (1,1), padding=(1,1), dtype=trt.float16)

loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
inputs=torch.rand((32,64,8,8), requires_grad=True)


ground_truth=torch.rand(32,64,8,8)
for i in range(0,50):
    output=model(inputs)
    loss=loss_fn(output, ground_truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



pr=cProfile.Profile()
pr.enable()
for i in range(0,100):
    output=model(inputs)
    loss=loss_fn(output, ground_truth)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
pr.disable()
pr.print_stats(sort=2)
