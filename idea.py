import numpy as np

total_x=100
q_table = np.zeros((2,total_x,1))
env_value = np.zeros((2,total_x,1))
env_value[:,int(total_x-1),0]=1
trans_mat = np.zeros((2,total_x,total_x))

trans_mat[0,:total_x-1,1:]=np.eye(total_x-1)
trans_mat[1,1:,:total_x-1]=np.eye(total_x-1)

env_value_after_trans=np.matmul(trans_mat,env_value)

env_value_after_trans[0,total_x-1,0]=-1
env_value_after_trans[1,0,0]=-1

epoch =2000
alpha=0.5
gama=0.8

for i in range(epoch):

    q_table_soft_max=np.exp(q_table)
    q_exp_sum = np.sum(q_table_soft_max,axis=0)
    posib=q_table_soft_max/q_exp_sum

    trans_mat_true=trans_mat*posib

    q_right = q_table[0]
    q_left = q_table[1]
    q_k = np.concatenate((q_right,q_left),axis=1)

    env_value_after_trans_true = env_value_after_trans*posib

    pre_trans=np.matmul(trans_mat_true,q_k)
    best_choice=np.max(pre_trans,axis=2)
    best_choice=np.reshape(best_choice,(2,total_x,1))

    q_table=(1-alpha)*q_table+(alpha)*(env_value_after_trans_true+gama*best_choice)

print(q_k)