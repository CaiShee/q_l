import torch
import numpy as np

x = 5
y = 5
actions = 4
epoch =150
alpha=0.5
gama=0.8

treasure_x=4
treasure_y=4

A = torch.zeros((int(actions), int(x*y), int(x*y)))

r_block = torch.zeros((int(x), int(x)))
r_block[:-1, 1:] = torch.eye(int(x-1))

l_block = torch.zeros((int(x), int(x)))
l_block[1:, :-1] = torch.eye(int(x-1))

for i in range(y):
    A[0, i*x:(i+1)*x, i*x:(i+1)*x] = r_block
    A[1, i*x:(i+1)*x, i*x:(i+1)*x] = l_block

A[2, :-x, x:] = torch.eye((y-1)*x)
A[3, x:, :-x] = torch.eye((y-1)*x)

env_value = torch.zeros((actions,x*y,1))
env_value[:,treasure_x+y*treasure_y,0]=1
env_value_after_trans=torch.matmul(A,env_value)

no_l=torch.arange(0,x*y,x)
no_r=no_l+x-1
no_u=torch.arange(0,x,1)
no_d=no_u+(y-1)*x

env_value_after_trans[0,no_r,0]=-1
env_value_after_trans[1,no_l,0]=-1
env_value_after_trans[2,no_d,0]=-1
env_value_after_trans[3,no_u,0]=-1

q_table=torch.zeros((actions,x*y,1))

for i in range(epoch):
    q_table_softmax=torch.exp(q_table)
    q_exp_sum = torch.sum(q_table_softmax,axis=0)
    posib=q_table_softmax/q_exp_sum
    posib_max,_=torch.max(posib,axis=0)
    posib/=posib_max

    trans_mat_true=A*posib

    q_k =torch.transpose(q_table,0,2)[0]

    env_value_after_trans_true = env_value_after_trans*posib

    pre_trans=torch.matmul(trans_mat_true,q_k)
    best_choice=torch.max(pre_trans,axis=2)
    best_choice=torch.reshape(best_choice.values,(actions,x*y,1))

    q_table=(1-alpha)*q_table+(alpha)*(env_value_after_trans_true+gama*best_choice)
    print("---第"+str(i)+"轮---")
    print(q_k)