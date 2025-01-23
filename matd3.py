"""
Function for building Multi-Agent Twin Delayed Deep Deterministic Policy Gradient(MATD3) algorithm.

Using:
numpy: 1.22.4
pytroch: 1.12.0
"""
import numpy as np
import torch as T
import torch.nn.functional as F
from agent import Agent
from networks import HeteroGATLayer
T.set_printoptions(profile="full")

class MATD3:
    def __init__(self, embedding_dim, chkpt_dir, actor_dims, critic_dims, n_agents, n_actions,freq,
                 fc1_actor, fc2_actor, fc1_critic,fc2_critic, alpha, beta, gamma, tau): 
        """
        :param chkpt_dir: check point directory
        :param actor_dims: number of dimensions for the actor
        :param critic_dims: number of dimensions for the critic
        :param n_agents: number of agents
        :param n_actions: number of actions
        :param freq: updating frequency, default value is 100
        :param fc1: number of dimensions for first layer, default value is 128
        :param fc2: number of dimensions for second layer, default value is 64
        :param alpha: learning rate of actor (target) network, default value is 0.01
        :param beta: learning rate of critic (target) network, default value is 0.01
        """
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.freq = freq
        self.embedding_dim = embedding_dim
        self.agents = [Agent(embedding_dim, actor_dims[agent_idx], critic_dims, n_actions, n_agents, agent_idx, chkpt_dir, 
                             alpha, beta, fc1_actor, fc2_actor, fc1_critic,fc2_critic,gamma, tau) for agent_idx in range(self.n_agents)]

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        for agent_idx in range(self.n_agents):
            self.agents[agent_idx].save_models()

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        for agent_idx in range(self.n_agents):
            self.agents[agent_idx].load_models()

    def reset_noise(self):
        for agent_idx in range(self.n_agents):
            self.agents[agent_idx].reset_noise()

    def choose_action(self, task_state,vm_state, edge_index, valid_action, Exploration, noise_l, epsilon_pro):
        """
        function for choosing action
        :param raw_obs: raw observation
        :param exploration: exploration state flag
        :param n_l: limitation for noise processes, default value is 0.2
        :return: agents' actions
        """
        # 使用GNN层计算全局嵌入
        actions = []
        with T.no_grad():
            print('valid_action', valid_action)
            for vm_id, allowed_actions in valid_action.items():
                # 如果当前虚拟机没有合法动作 则直接跳过
                if not allowed_actions:
                    actions.append(-1)
                    continue
                print('vm_id',vm_id)
                print('allowed_actions',allowed_actions)
                task_state = T.tensor(task_state, dtype=T.float32).cuda()
                edge_index = T.tensor(edge_index, dtype=T.long).cuda()
                vm_state = T.tensor(vm_state, dtype=T.float32).cuda()
              
                agent_embedding = self.agents[vm_id].gnn(task_state, vm_state, edge_index)
                # 每个agent 基于自己的观察
                agent_embedding = agent_embedding[vm_id, :]  # 提取agent的嵌入
                action = self.agents[vm_id].choose_action(agent_embedding, allowed_actions, Exploration, noise_l, epsilon_pro)
                actions.append(action)
        return actions


    def learn(self, memory, writer, steps_total, batch_size, vm_num, embedding_num):
        """
        agents would learn after filling the bitch size of memory, and update actor and critic networks
        :param memory: memory state (from buffer file)
        :param writer: writer for saving data, which will be used for TensorBoard
        :param steps_total: total steps(all training episodes)
        :return: results after learning
        """
        vm_embedding = 5
        device = self.agents[0].actor.device
        
        for agent_idx in range(self.n_agents):
            task_states, vm_states, edge_indices, actions, rewards, task_new_states, VM_new_states, edge_new_indices, dones = memory.sample_buffer()
           
            rewards = T.tensor(rewards).to(device)
            dones = T.tensor(dones).to(device)
            # actions according to the target network for the new state

            # 处理动作
            actions_tensor = T.tensor(actions, dtype=T.float).to(device)
            actions_clipped = T.clamp(actions_tensor, min=0).long()  # 将 -1 转换为 0
            one_hot_actions = F.one_hot(actions_clipped, num_classes=self.n_actions).float()  # 转为 one-hot 编码
            one_hot_actions[actions_tensor == -1] = 0.0  # 对 -1 的动作重新设置为全零

            task_states = T.tensor(task_states)
            vm_states = T.tensor(vm_states)
            edge_indices = T.tensor(edge_indices)
          
            # 当前的状态
            # 处理batch个GNN嵌入  因为gnn_layer只能接收一张图
            # 当前时刻的批量数据
            current_task_features = T.cat([T.tensor(ts, dtype=T.float).to(device) for ts in task_states], dim=0)
            current_vm_features = T.cat([T.tensor(vms, dtype=T.float).to(device) for vms in vm_states], dim=0)
            current_edge_indices = T.cat([T.tensor(ei, dtype=T.long).to(device) for ei in edge_indices], dim=1)
        
            # 下一时刻的批量数据
            next_task_features = T.cat([T.tensor(ts, dtype=T.float).to(device) for ts in task_new_states], dim=0)
            next_vm_features = T.cat([T.tensor(vms, dtype=T.float).to(device) for vms in VM_new_states], dim=0)
            next_edge_indices = T.cat([T.tensor(ei, dtype=T.long).to(device) for ei in edge_new_indices], dim=1)

            # 批索引 (区分子图)
            batch_index = T.cat([T.full((len(ts),), i, dtype=T.long) for i, ts in enumerate(task_states)]).to(device)
          
            # 使用当前 agent 的 GNN 处理批量嵌入
            current_embedding_batch = self.agents[agent_idx].gnn(
                current_task_features, current_vm_features, current_edge_indices, batch=batch_index
            )   # [batch_num * vm_num, task_num * embedding + embedding]
            new_embedding_batch = self.agents[agent_idx].gnn(
                next_task_features, next_vm_features, next_edge_indices, batch=batch_index
            )   # [batch_num * vm_num, task_num * embedding + embedding]
            
            # extract actor embedding form 
            target_indices = T.arange(0, batch_size).to(device) * vm_num + agent_idx
            selected_current_embeddings = current_embedding_batch[target_indices] 
            selected_new_embeddings = new_embedding_batch[target_indices]  
          
            # 动作
            new_pi = self.agents[agent_idx].target_actor(selected_new_embeddings)
            mu_pi = self.agents[agent_idx].actor(selected_current_embeddings)

            # 打印 mu_pi，确保它是正确的，并且它的输出用于损失计算
            # Critic input: concatenate all node embeddings for each subgraph
            
            new_embedding_batch_3d = new_embedding_batch.view(batch_size, vm_num, -1)
            current_embedding_batch_3d = current_embedding_batch.view(batch_size, vm_num, -1)
        
            current_embedding_batch_task = current_embedding_batch_3d[:, 0, :-vm_embedding]  # [batch, task_num * embedding]
            current_embedding_batch_vm = current_embedding_batch_3d[:, :, -vm_embedding:]  # [batch, vm_num, embedding]
         
            vm_embedding_concat_current = current_embedding_batch_vm.reshape(current_embedding_batch_vm.size(0), -1) 
            # 拼接task部分和vm_embedding-concat部分
            critic_current_inputs = T.cat((current_embedding_batch_task, vm_embedding_concat_current), dim = -1)   

            new_embedding_batch_task = new_embedding_batch_3d[:, 0, :-vm_embedding]
            new_embedding_batch_vm = new_embedding_batch_3d[:, :, -vm_embedding:]
            vm_embedding_cocncat_new = new_embedding_batch_vm.reshape(new_embedding_batch_vm.size(0), -1)
            # 拼接task部分和vm_embedding-concat部分
            critic_new_inputs = T.cat((new_embedding_batch_task, vm_embedding_cocncat_new), dim = -1) 
            # critic 估计 Q 值
            current_Q1, current_Q2 = self.agents[agent_idx].critic(critic_current_inputs, one_hot_actions[:, agent_idx])
       
            target_Q1, target_Q2 = self.agents[agent_idx].target_critic(critic_new_inputs, new_pi)
            # 使用 target critic 的最小值作为目标 Q 值
            target_Q_min = T.min(target_Q1, target_Q2)
            target_Q = rewards[:, agent_idx] + (self.agents[agent_idx].gamma * target_Q_min) 
            # Critic 更新
            self.agents[agent_idx].critic_loss = F.mse_loss(current_Q1.float(), target_Q.float()) +\
                                                 F.mse_loss(current_Q2.float(), target_Q.float())
            self.agents[agent_idx].critic.optimizer.zero_grad()
            self.agents[agent_idx].critic_loss.backward()
            # self.agents[agent_idx].critic_loss.backward(retain_graph=True)
            self.agents[agent_idx].critic.optimizer.step()
            writer.add_scalar('agent_%s' % agent_idx + '_critic_loss', self.agents[agent_idx].critic_loss, steps_total)

            # Actor 更新：每隔一定频率更新 Actor 和目标网络
            if steps_total % self.freq == 0 and steps_total > 0:
                batch_index = T.cat([T.full((len(ts),), i, dtype=T.long) for i, ts in enumerate(task_states)]).to(
                    device)
                current_embedding_batch = self.agents[agent_idx].gnn(
                    current_task_features, current_vm_features, current_edge_indices, batch=batch_index
                )
                # extract actor embedding form
                target_indices = T.arange(0, batch_size).to(device) * vm_num + agent_idx
                selected_current_embeddings = current_embedding_batch[target_indices]  # [batch_size, embedding_dim * 2]
                mu_pi = self.agents[agent_idx].actor(selected_current_embeddings)

                current_embedding_batch_3dd = current_embedding_batch.view(batch_size,vm_num,-1)
                current_embedding_batch_task = current_embedding_batch_3dd[:,0,:-vm_embedding]
                current_embedding_batch_vm = current_embedding_batch_3dd[:, :, -vm_embedding:]
                vm_embedding_concat_current_1 = current_embedding_batch_vm.reshape(current_embedding_batch_vm.size(0),-1)
                critic_current_inputs_1 = T.cat((current_embedding_batch_task, vm_embedding_concat_current_1),dim=-1)

                # 计算 Actor 损失
                self.agents[agent_idx].actor_loss = -T.mean(self.agents[agent_idx].critic.Q1(critic_current_inputs_1,mu_pi))
                self.agents[agent_idx].actor.optimizer.zero_grad()
                self.agents[agent_idx].actor_loss.backward()
                # self.agents[agent_idx].actor_loss.backward(retain_graph=True)
                self.agents[agent_idx].actor.optimizer.step()

                # 更新目标网络参数
                self.agents[agent_idx].update_network_parameters()
                writer.add_scalar('agent_%s' % agent_idx + '_actor_loss', self.agents[agent_idx].actor_loss, steps_total)

            return self.agents[0].critic_loss
