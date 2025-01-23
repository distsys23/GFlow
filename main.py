"""
Main function
Type in terminal to see writer in web page: tensorboard --logdir=SavedLoss
CUDA version: 11.2
tensorboard: 2.9.0
"""
import os
import matplotlib.pyplot as plt
import time
import torch as T
from workflow import *
from functions import *
from buffer import MultiAgentReplayBuffer
from matd3 import MATD3
from env_new import *
from parameter import args

from torch.utils.tensorboard import SummaryWriter
# from rl_plotter.logger import Logger
def get_edge_index(task_matrix):
    """
    将邻接矩阵转换为 edge_index 格式
    :param task_matrix: 任务的邻接矩阵 [num_tasks, num_tasks]
    :return: 边索引矩阵 edge_index，形状为 [2, num_edges]
    """
    # 获取邻接矩阵中非零元素的索引
    src, dst = np.nonzero(task_matrix)  # src 和 dst 是两个一维数组
    edge_index = np.array([src, dst], dtype=np.int64)  # 转为 [2, num_edges] 格式
    return edge_index

def sat_security(env, allowed_actions):
    valid_actions = {vm_id: [] for vm_id in range(env.vm_num)}  # 为每个虚拟机创建一个任务列表
    # 遍历所有虚拟机
    for vm_id in range(env.vm_num):
        # 遍历所有任务 并进行安全性检查
        for task in allowed_actions:
            selected_task = env.dag.jobs[task]
            # 任务的安全性要求
            task_security_authentication = selected_task['security_authentication']
            task_security_confidentiality = selected_task['security_confidentiality']
            task_security_integrity = selected_task['security_integrity']
            task_privacy_security_level = selected_task['privacy_security_level']
            suc = 0
            # 根据任务的安全性等级进行检查
            # 安全性等级为l1
            if task_privacy_security_level == 1:
                if env.vm_type[vm_id] == 0:  # 只能在私有云上运行
                    suc = 1
            # 安全性等级为l2
            elif task_privacy_security_level == 2:
                if env.vm_type[vm_id] == 0:  # 私有云上运行
                    suc = 1
                else:  # 公有云上运行，且满足认证、机密性、完整性要求
                    if (env.vm_security_authentication[vm_id] >= task_security_authentication and
                        env.vm_security_confidentiality[vm_id] >= task_security_confidentiality and
                        env.vm_security_integrity[vm_id] >= task_security_integrity):
                        suc = 1
            # 安全性等级为l3
            elif task_privacy_security_level == 3:
                if env.vm_type[vm_id] == 0:  # 私有云上运行
                    suc = 1
                else:  # 公有云上运行，且满足完整性要求
                    if env.vm_security_integrity[vm_id] >= task_security_integrity:
                        suc = 1
            #  如果虚拟机通过安全性验证  任务就可以分配给该虚拟机 
            if suc == 1:
                valid_actions[vm_id].append(task)  # 将符合条件的任务加入当前虚拟机
    return valid_actions

if __name__ == '__main__':
    learn_interval = 40 
    fc1_actor = 128
    fc2_actor = 64
    fc1_critic = 128
    fc2_critic = 64

    alpha = 0.00001
    beta = 0.0001
    gamma = 0.8
    tau = 0.01
    # arrival_rate = 1

    embedding_num = 16 
    actor_input_embedding = args.task_num * embedding_num + 5 # concat task & vm
    batch_size = 32

    scientific_workflow = Scientific_Workflow('CyberShake', 30)
    dag = scientific_workflow.get_workflow()
    print('dag',dag)
    env = MultiAgentEnv(dag)
    n_agents = env.vm_num
  
    actor_dims = []
    for idx in range(n_agents):
        actor_dims.append(actor_input_embedding)
    critic_dims = args.task_num * embedding_num + 5 * args.VM_num

    # action space, action: 0 for not allocate, 1 for id of task to allocate
    n_actions = env.task_num
    chkpt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SavedNetwork')

    # 确保目录存在
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)

    marl_agents = MATD3(embedding_num, chkpt_dir, actor_dims, critic_dims, n_agents, n_actions, learn_interval,fc1_actor, fc2_actor, fc1_critic,fc2_critic,alpha, beta, gamma, tau)

    max_size = 10000
    memory = MultiAgentReplayBuffer(max_size, actor_dims, critic_dims, n_agents, n_actions, batch_size)
    steps_epoch = 3000  #10000  # number of maximum episodes
    steps_exp = steps_epoch / 3
    global_step = 0

    epsilon_pro = 0
    epsilon_pro_increment = 0.01
    epsilon_pro_max = 1

    result_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SavedResult14_1-security')
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # result_dir = os.path.dirname(os.path.realpath(__file__)) + '\SavedResult'
    writer = SummaryWriter("SavedLoss")

    makespan_list = []
    cost_list = []
    reward_list = []
    success_rate_list = []
    loss_list = []

    #  生成每个工作流的到达时间

    # arrival_time_intervals = np.random.poisson(arrival_rate, size=steps_epoch)

    # main loop for MATD3
    for episode in range(steps_epoch):
        print('----------------------------Episode', episode, '----------------------------')
        """
        obs: init state of agent: 
        """  
        n_obs = env.reset()
        done_reset = False # 结束条件为工作流中的所有任务都完成
        done_goal = env.task_events # 记录任务完成状态
        '''[False] * self.task_num'''
        training_counts = 0

        if episode < steps_exp:
            Exploration = True
            marl_agents.reset_noise()
        else:
            Exploration = False
        noise_l = 0.2  # valid noise range
        
        step = 0
        reward = np.zeros(n_agents) # reward for each episode
        reward_episode_sum = 0

        # 初始化环境和状态
        task_state = env.get_task_state() # 此刻
        VM_state = env.get_VM_state()  # 此刻
        task_matrix = env.get_task_matrix()  #此刻
        edge_index = get_edge_index(task_matrix)

        if (episode + 1) % 10 == 0:
            epsilon_pro = min(epsilon_pro + epsilon_pro_increment, epsilon_pro_max)   

        while not done_reset:
            print('----------------------Step', step, '----------------------')
            env.task_load() # 加载满足条件的 task 到总队列 Queue 中        
            allowed_actions = env.dag_queue.task_queue.queue # 初始运行的动作： 队列中的任务   

            # 在选择任务之前检查每个虚拟机的可选择任务
            valid_action = sat_security(env, allowed_actions)

            actions = marl_agents.choose_action(task_state, VM_state,edge_index, valid_action, Exploration, noise_l, epsilon_pro)    
            # action: -1 for not allocate, others for id of task to allocate, example: actions = [-1, -1, -1, 29, -1, -1, -1, -1, -1, -1]    
            # data normalization
            actions = np.array(actions)
            vm_speed = np.array(args.vm_speed)  # 获取虚拟机速度
            # 冲突检测与解决
            unique_actions, counts = np.unique(actions, return_counts=True)
            conflicting_tasks = unique_actions[counts > 1]  # 找到被多个Agent选择的任务

            for task in conflicting_tasks:
                # 找到选择该任务的所有虚拟机
                indices = np.where(actions == task)[0]
                best_vm = None
                best_reward = -float('inf')
                # 对每个选择该任务的虚拟机计算模拟奖励

                for vm_idx in indices:
                    simulated_reward = env.simulate_feedback(task, vm_idx)  # 调用模拟奖励计算
                    if simulated_reward > best_reward:
                        best_reward = simulated_reward
                        best_vm = vm_idx
                # 保留奖励值最高的虚拟机，其他虚拟机设为 -1
                for vm_idx in indices:
                    if vm_idx != best_vm:
                        actions[vm_idx] = -1
            # print('actions:', actions)
            '''actions: [-1 -1 -1 -1 -1 -1 13  2 -1 -1 -1 -1 -1 -1 -1 -1]'''
            need_allocate = env.dag_queue.size()          
            allocate_count = 0
            # print('need_allocate',need_allocate)
            # print('actions',actions)

            for j in range(env.vm_num):
                #if actions[j] in env.dag_queue.task_queue.queue:
                if actions[j] != -1 and actions[j] in env.dag_queue.task_queue.queue:
                    popped_task = env.dag_queue.pop_specific(actions[j])  
                    if popped_task is not None:
                        reward_feedback = env.feedback(actions[j], j)
                        reward_episode_sum += reward_feedback
                        reward[j] = reward_feedback
                        allocate_count += 1
                        need_allocate -= 1
                    
            task_state_next = env.get_task_state()
            VM_state_next = env.get_VM_state()
            task_matrix_next = env.get_task_matrix()
            edge_index_next = get_edge_index(task_matrix)

            # when done
            if all(done_goal):  
                done_reset = True
                # reward_list.append((episode, sum(reward)))
                print(f"training_counts: {training_counts}")
                # print('reward_list:', reward_list)
            done = [True] * n_agents
            
            memory.store_transition(task_state, VM_state, edge_index, actions, reward, task_state_next, VM_state_next, edge_index_next, done)
            if not memory.ready():
                pass
            else:
                loss = marl_agents.learn(memory, writer, global_step, batch_size, n_agents, embedding_num)
                loss_list.append(loss)
                training_counts+=1
            task_state = task_state_next
            VM_state = VM_state_next
            edge_index = edge_index_next
            step += 1
            global_step += 1

        reward_list.append(reward_episode_sum)
        makespan = env.get_makespan()
        # makespan = sum(env.response_time_list)
        print('makespan:', makespan)
        if episode % 1 == 0:
            makespan_list.append(makespan)

        cost = sum(env.vm_cost_list)
        print('cost:', cost)
        if episode % 1 == 0:
            cost_list.append(cost)

        success_rate = np.sum(env.success_event) / env.task_num
        print('success_rate',success_rate)
        if episode % 1 == 0:
            success_rate_list.append(success_rate)

    writer.close()
    # save networks
    marl_agents.save_checkpoint()
    # rewards_only = [item[1] for item in reward_list]
    loss_array = np.array([loss.cpu().detach().numpy() for loss in loss_list])
    
  
   

 
 