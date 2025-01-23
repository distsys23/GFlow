"""
Function for building networks

For each part(actor or critic), there are two network.
All four networks have similar structure.
Different parameter values, like learning rates(alpha and beta), could be implemented in networks.
For the critic part, the critic network would have the same parameter values with the critic target network.
The same situation happens in the actor part.

Using:
pytroch: 1.12.0
os: Built-in package of Python
Python: 3.9
"""
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

def init_weights(layer):
    """
    Initialize weights of networks (-0.001, 0.001)
    :param layer:
    :return:
    """
    if type(layer) == nn.Linear:
        nn.init.uniform_(layer.weight, a=-0.001, b=0.001)
        nn.init.constant_(layer.bias, 0)

class HeteroGATLayer(nn.Module):
    def __init__(self, task_num, task_input_dim, vm_input_dim, embedding_dim, num_heads=1, dropout=0.1):
        super(HeteroGATLayer, self).__init__()

        # GAT for task node feature processing
        self.gat1 = GATConv(task_input_dim, embedding_dim, heads=num_heads, dropout=dropout)  # 第一个GAT层
        # 输出是[num_tasks, embedding_dim * num_heads]
        self.gat2 = GATConv(embedding_dim * num_heads, embedding_dim, heads=1, dropout=dropout)  # 第二个GAT层

        # self.sage1 = SAGEConv(task_input_dim, embedding_dim)  # 第一个 SAGE
        # self.sage2 = SAGEConv(embedding_dim, embedding_dim) # 第二个SAGE
        # MLP for mapping VM node features to the same dimension as task embedding
        # self.vm_fc1 = nn.Linear(vm_input_dim, embedding_dim)
        # self.vm_fc2 = nn.Linear(embedding_dim, embedding_dim)
        # 输出是[num_vms, embedding_dim]
        # reduction layer
        self.reduce_fc = nn.Linear(embedding_dim * task_num, 512)
        
    def forward(self, task_features, vm_features, task_edge_index, batch = None):
        # Step 1: Process task nodes with GAT

        # task_embeddings = self.sage1(task_features, task_edge_index)  # 第一层
        # task_embeddings = nn.ReLU()(task_embeddings)  # 激活函数
        # task_embeddings = self.sage2(task_embeddings, task_edge_index)  # 第二层
        # task_embeddings = nn.ReLU()(task_embeddings)  # 激活函数
        task_embeddings = F.relu(self.gat1(task_features, task_edge_index))
        task_embeddings = F.relu(self.gat2(task_embeddings, task_edge_index))

        if batch is None:
            # 单图
            task_embedding_flattened = task_embeddings.view(1, -1)
        else:
            num_graphs = len(T.unique(batch))  # 获取图的数量
            batch_size = num_graphs
            task_embedding_flattened = T.cat(
                [task_embeddings[batch == i].view(1, -1) for i in range(batch_size)],
                dim=0
            )
        # Step 2: Process VM nodes with MLP (simple transformation to match the embedding dimension)
        vm_embedding = vm_features
        # vm_embedding = F.relu(self.vm_fc1(vm_features)) # [num_vms, embedding_dim]
        # vm_embedding = F.relu(self.vm_fc2(vm_embedding))  # 使用 vm_fc2 层
        
        if batch is None:
            # 单图
            task_embedding_repeated = task_embedding_flattened.expand(vm_embedding.size(0), -1)  # [num_vms, embedding_reduced]
        else:
            # 我们现在希望把每个vm的embedding和相同batch_szie_id的任务embedding拼接在一起，所以我们需要把task_embedding_mean扩充为 (num_vms*batch_size, embedding_dim),从而和vm_embedding拼接
            num_vms=int(vm_embedding.shape[0]/len(T.unique(batch)))
            task_embedding_repeated = task_embedding_flattened.unsqueeze(1).repeat(1, num_vms, 1).reshape(vm_embedding.size(0), -1)# (batch_size, embedding_dim)->(batch_size, 1, embedding_dim)->(batch_size, num_vms, embedding_dim)->(batch_size*num_vms, embedding_dim)
        # Step 3: Concatenate task and VM embeddings
        final_embedding = T.cat((task_embedding_repeated, vm_embedding), dim=-1)  
        return final_embedding # 

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_critic,fc2_critic, n_agents, n_actions, name, chkpt_dir):
        """
        :param beta: learning rate of critic network
        :param input_dims: number of dimensions for inputs
        :param fc1_dims: number of dimensions for first layer
        :param fc2_dims: number of dimensions for second layer
        :param n_agents: number of agents
        :param n_actions: number of actions
        :param name: name of network
        :param chkpt_dir: check point directory
        """
        super(CriticNetwork, self).__init__()  # call the superclass(nn.Module) constructor

        self.chkpt_file = os.path.join(chkpt_dir, name)
        # 确保使用正确的路径分隔符，并确保目录存在

        # network architecture of q1
        self.fc1 = nn.Linear(input_dims + n_actions, fc1_critic)
        self.fc2 = nn.Linear(fc1_critic, fc2_critic)
        self.q1 = nn.Linear(fc2_critic, 1)

        # network architecture of q2
        self.fc3 = nn.Linear(input_dims + n_actions, fc1_critic)
        self.fc4 = nn.Linear(fc1_critic, fc2_critic)
        self.q2 = nn.Linear(fc2_critic, 1)
        # optimization method
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        # if possible, use GPU to train
        self.apply(init_weights)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """
        :param state:
        :param action:
        :return: result of the network
        """
        x1 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x1 = F.relu(self.fc2(x1))  
        q1 = self.q1(x1)
    
        x2 = F.relu(self.fc1(T.cat([state, action], dim=1)))     
        x2 = F.relu(self.fc2(x2))    
        q2 = self.q2(x2)

        return q1.squeeze(-1), q2.squeeze(-1)

    def Q1(self, state, action):
        """
        :param state:
        :param action:
        :return: result of the network
        """
        x1 = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x1 = F.relu(self.fc2(x1))
        q1 = self.q1(x1)
        return q1.squeeze(-1)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_actor, fc2_actor, n_actions, name, chkpt_dir):
        """
        :param alpha: learning rate of actor network
        :param input_dims: number of dimensions for inputs
        :param fc1_dims: number of dimensions for first layer
        :param fc2_dims: number of dimensions for second layer
        :param n_actions: number of actions
        :param name: name of network
        :param chkpt_dir: check point directory
        """
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(input_dims, fc1_actor)
        self.fc2 = nn.Linear(fc1_actor, fc2_actor)
        self.pi = nn.Linear(fc2_actor, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.apply(init_weights)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, agent_embedding):
        """
        :param state:
        :return: result of the network
        """
        x = F.leaky_relu(self.fc1(agent_embedding))
        x = F.leaky_relu(self.fc2(x))
        # x = F.relu(self.fc2(state))
        # output range (-1,1)
        pi = nn.Tanh()(self.pi(x))
        # pi = nn.Softsign()(self.pi(x))
        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))