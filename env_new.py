"""
Code for creating a multi-agent environment with VMs and tasks 

定义VM, workflow中的task放入queue进行调度
"""
import queue
from workflow import *
from parameter import args
import random
import copy

class MultiAgentEnv:
    def __init__(self, dag):
        """
        Function for initializing the environment
        VM, dag, task, queue
        """
        self.dag = dag
        # VM
        self.vm_num = args.VM_num # VM number
        self.private_cloud_num = args.private_cloud_num # private cloud VM number
        self.public_cloud_num = args.public_cloud_num # public cloud VM number
        self.vm_cost = args.vm_cost # VM price
        self.vm_speed = args.vm_speed # VM computing speed
        self.vm_type = args.vm_type # VM type: 0 for private cloud VM, 1 for public cloud VM
        self.vm_security_authentication = args.vm_security_authentication
        self.vm_security_confidentiality = args.vm_security_confidentiality
        self.vm_security_integrity = args.vm_security_integrity
        # self.vm_security = args.vm_security

        # queues for vms
        self.vm_queues = {i: queue.Queue() for i in range(self.vm_num)}
        # task
        self.task_num = dag.n_task # task number
        # edge
        self.edges = self.dag.edges
        self.edge_events = np.zeros((self.task_num, self.task_num)) # record edge transfer completion state, 1 for finished, 0 for unfinished

        # record state
        # self.task_events = np.zeros((self.task_num)) # task finish state, 1 for finished, 0 for unfinished
        self.task_events = [False] * self.task_num
        # vm state, including vm idle time and tasks on vm: [vm_idle, task_id]
        # vm state, including [task_num_of_vm, executing_task_time, wating_task_time]
        self.vm_events = np.zeros((2, self.vm_num))  # 0. 等待时间 1. 执行任务的个数
        self.dqn_events = np.zeros((7, self.task_num)) # dqn state, including .... 
        '''
        dqn_events:
        self.dqn_events[0, action] = excution_time
        self.dqn_events[1, action] = reward
        self.dqn_events[2, action] = task_arrival_time   # arrival at Queue
        self.dqn_events[3, action] = task_start_time
        self.dqn_events[4, action] = task_waiting_time    # task_start_time - task_arrival_time
        self.dqn_events[5, action] = task_finish_time
        self.dqn_events[6, action] = tranfer_time
        '''
        # record for makespan
        self.response_time_list = []

        # record for cost
        self.vm_cost_list = []

        # record for success rate
        self.success_event = np.zeros((self.task_num))
        # queue for dag

        self.dag_queue = Queue(dag)

    def reset(self):
        # edge
        self.edge_events = np.zeros((self.task_num, self.task_num))
        # queues for vms
        self.vm_queues = {i: queue.Queue() for i in range(self.vm_num)}
        # self.task_events = np.zeros((self.task_num)) 
        self.task_events = [False] * self.task_num
        self.vm_events = np.zeros((2, self.vm_num)) 
        self.dqn_events = np.zeros((7, self.task_num))
        self.dag_queue = Queue(self.dag)
        # record for makespan
        self.response_time_list = []
        # record for cost
        self.vm_cost_list = []
        # record for success rate
        self.success_event = np.zeros((self.task_num))
          
    def update_arrival_time(self, n):
        """
        Function for updating the arrival time required for each task
        n: task id
        """
        # arrival time of tasks: update in task_load, that is, when task is put into Queue
        # arrival_final = min(current_arrive_time, makespan)
        if len(self.dag.precursor[n]) == 0:  #
            self.dqn_events[2, n] = 0  # 如果没有前继节点  那么任务的到达时间就是 该工作流的到达时间
        else:
            max_time = 0
            for pred_n in self.dag.precursor[n]:
                if(self.dqn_events[5, pred_n] > max_time):
                    max_time = self.dqn_events[5, pred_n]
            self.dqn_events[2, n] = max_time

    def get_task_state(self):
        # [is_done, is_can_execute, running_time, waiting_time]
        task_state = np.zeros((self.task_num, 8))
        task_events_np = np.array(self.task_events)
        task_state[:,0] = task_events_np.astype(int) # 0:未完成, 1:已完成
        # 获取当前可执行任务：通过检查dag_queue任务队列
        can_execute_tasks = set(self.dag_queue.task_queue.queue)  # 当前可执行任务的集合
        # 遍历每个任务
        for task_id in range(self.task_num):
        # 任务是否可以执行：只需要检查该任务是否在可执行队列中
            task_state[task_id, 1] = 1 if task_id in can_execute_tasks else 0  # 1:可执行 0：不可执行
            task_state[task_id, 2] = self.dag.jobs[task_id]['runtime']  # 获取任务的runningtime
            task_state[task_id, 3] = self.dqn_events[2, task_id]
            task_state[task_id, 4] = self.dag.jobs[task_id]['security_authentication']
            task_state[task_id, 5] = self.dag.jobs[task_id]['security_confidentiality']
            task_state[task_id, 6] = self.dag.jobs[task_id]['security_integrity']
            task_state[task_id, 7] = self.dag.jobs[task_id]['privacy_security_level']
        return task_state

    def get_VM_state(self):
        vm_state = np.zeros((self.vm_num, 5))
        # vm available time 
        # print('self.vm_events[5,:]', self.vm_events[4,:])
        vm_state[:,0] = self.vm_events[0,:]
        vm_state[:,1] = self.vm_type
        vm_state[:,2] = self.vm_security_authentication
        vm_state[:,3] = self.vm_security_confidentiality
        vm_state[:,4] = self.vm_security_integrity
        # vm_state[:,2] = self.vm_events[1,:]
        return vm_state

    def get_task_matrix(self):
        # 初始化邻接矩阵
        task_matrix = np.copy(self.dag.dag)
        # 遍历任务完成状态，移除已完成任务的出边
        for task_id, is_finished in enumerate(self.task_events):
            if is_finished:
                task_matrix[task_id, :] = 0  # 将已完成任务的所有出边设置为 0
        return task_matrix

    # 总体queue的操作
    def task_load(self):
        """
        Function for loading tasks into the dag_queue
        """
        unfinished_task = [i for i in range(self.task_num) if self.task_events[i] == False]
        # print('unfinished_task:', unfinished_task)
        # 加载任务：满足所有父任务完成且传输完毕，且不在队列中
        for n in range(self.task_num):
            if(self.dag_queue.check_legal(n, self.task_events) and n not in self.dag_queue.task_queue.queue and n in unfinished_task):
                if all(self.edge_events[pred_n, n] == 1 for pred_n in self.dag.precursor[n]): # 判断所有父任务传输完毕
                    self.dag_queue.push(n)
                    self.update_arrival_time(n) # task arrival time

    def task_pop(self):
        """
        Function for popping a task from the dag_queue
        """
        task = self.dag_queue.pop()
        # print('Queue:', self.dag_queue.task_queue.queue)
        return task
    
    def feedback(self, action, vm_id):
        """
        Function for getting the feedback of environment 
        """
        selected_task = self.dag.jobs[action]
        execution_time = selected_task['runtime'] / self.vm_speed[vm_id]
     
        # VM 可用时间
        Tidle = self.vm_events[0, vm_id]
        self.update_arrival_time(action)
        arrival_time = self.dqn_events[2, action]
        
        # transfer-time
        total_transfer_time = 0
        for succ_n in self.dag.successor[action]:
            transfer_time = self.edges[action, succ_n] / 1000000
            self.edge_events[action, succ_n] = 1
            total_transfer_time += transfer_time

        if Tidle <= (arrival_time + total_transfer_time):
            Twait = 0
            Tstart = (arrival_time + total_transfer_time)
        else:
            Twait = Tidle - (arrival_time + total_transfer_time)
            Tstart = Tidle
        Tduration = Twait + execution_time + total_transfer_time
        self.response_time_list.append(Tduration)

        Tleave = Tstart + execution_time # 任务的离开时间
        Tnew_idle = Tleave # 服务的下一个可用时间

        vm_cost = execution_time * self.vm_cost[vm_id]
        self.vm_cost_list.append(vm_cost)
        reward = - Tleave

        # previous_makespan = self.get_makespan()
        # delta_makespan = Tleave - previous_makespan
        # reward = - delta_makespan
        suc = 0
        # r1 = - Tduration / 10
        # # r1 = - current_makespan / 5
        # r2 = - vm_cost / 10
        # reward = r1 + r2
        # print('r1',r1)

        self.task_events[action] = True
        self.dqn_events[0, action] = execution_time
        self.dqn_events[1, action] = reward
        self.dqn_events[2, action] = arrival_time
        self.dqn_events[3, action] = Tstart
        self.dqn_events[4, action] = Twait
        self.dqn_events[5, action] = Tleave
        self.dqn_events[6, action] = total_transfer_time

        self.vm_events[0, vm_id] = Tnew_idle
        self.vm_events[1, vm_id] += 1

        return reward

    def simulate_feedback(self, action, vm_id):
        """
        模拟分配任务到虚拟机，并计算对应的奖励值
        :param action: 任务ID
        :param vm_id: 虚拟机ID
        :return: 奖励值
        """
        # 临时状态变量
        simulated_task_events = copy.deepcopy(self.task_events)
        simulated_vm_events = copy.deepcopy(self.vm_events)
        simulated_edge_events = copy.deepcopy(self.edge_events)
        simulated_dqn_events = copy.deepcopy(self.dqn_events)

        selected_task = self.dag.jobs[action]
        execution_time = selected_task['runtime'] / self.vm_speed[vm_id]
        Tidle = simulated_vm_events[0, vm_id]  # 当前虚拟机的空闲时间
        # 1. 手动计算到达时间
        if len(self.dag.precursor[action]) == 0:
            arrival_time = 0
        else:
            max_finish_time = 0
            for pred in self.dag.precursor[action]:
                max_finish_time = max(max_finish_time, self.dqn_events[5, pred])  # 前置任务的完成时间
            arrival_time = max_finish_time

         # 3. 计算传输时间
        total_transfer_time = sum(
            self.edges[action, succ_n] / 1000000
            for succ_n in self.dag.successor[action]
        )
         # 4. 计算等待时间
        Twait = max(0, Tidle - (arrival_time + total_transfer_time))
        Tstart = max(Tidle, arrival_time + total_transfer_time)
        Tduration = Twait + execution_time
        Tleave = Tstart + execution_time
        vm_cost = execution_time * self.vm_cost[vm_id]
        # previous_makespan = self.get_makespan()
        # delta_makespan = Tleave - previous_makespan
        # suc = 0
        # r1 = - current_makespan / 5
        r1 = - Tduration / 10
        r2 = - vm_cost / 10
        simulated_reward = r1 + r2
        # simulated_reward = - Tleave
        return simulated_reward
    
    def get_makespan(self):
        makespan = max(self.dqn_events[5,:])
        return makespan

class Queue:
    def __init__(self, dag):
        """
        Function for initializing the task queue
        """
        self.dag = dag
        self.task_queue = queue.Queue(maxsize=dag.n_task)
        self.task_count = 0
        

    def push(self, task):
        """
        Function for pushing a task into the queue
        :param task: the task to be pushed
        :return: the queue after pushing
        """
        self.task_queue.put(task)
        return self.task_queue


    def pop(self):
        """
        Function for popping a task from the queue
        :return: the task popped
        """
        task = self.task_queue.get()
        return task


    def is_empty(self):
        """
        Function for checking whether the queue is empty
        :return: True if the queue is empty, False otherwise
        """
        return self.task_queue.empty()

    
    def size(self):
        """
        Function for checking the size of the queue
        :return: the size of the queue
        """
        return self.task_queue.qsize()


    def check_legal(self, n, task_finish_events):
        """
        Function for checking whether the task put in queue is legal, that is, all the precursor tasks are finished
        :return: True if the queue is legal, False otherwise
        """
        if len(self.dag.precursor[n]) == 0:
            return True
        for pred_n in self.dag.precursor[n]:
            if(task_finish_events[pred_n] == False):
                return False    
        return True


    def check_succ(self, n, finish_vertexs, wait_vertexs):
        """
        Function for 
        返回一个 "n个后续任务合法且不在wait_vertexs中 "的任务的列表
        """
        list = []
        for succ_n in self.dag.successor[n+1]:
            if(self.check_legal(succ_n, finish_vertexs) and succ_n not in wait_vertexs):
                list.append(succ_n)
        return list
    
    def pop_specific(self, task_id):
        '''
        按照任务ID弹出特定任务
        '''
        temp_queue = queue.Queue(maxsize=self.task_queue.maxsize)
        popped_task = None
        while not self.task_queue.empty():
            task = self.task_queue.get()
            if task == task_id:
                popped_task = task
            else:
                temp_queue.put(task)
        self.task_queue = temp_queue
        return popped_task
