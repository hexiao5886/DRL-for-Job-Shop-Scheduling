import os
import gym
import time
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from gym import spaces
from orliberty import load_random, load_instance




class GIN_JsspEnv(gym.Env):
    def __init__(self,
                 name: str = None,
                 num_jobs: int = None,
                 num_machines: int = None):
        super(GIN_JsspEnv, self).__init__()

        # Load instance
        if name[0].isdigit():
            self.name = name
            l = name.split('_')
            num_jobs, num_machines, idx = int(l[0]), int(l[1]), int(l[2])
            i, n, m, processing_time_matrix, machine_matrix = load_random(num_jobs, num_machines)
            processing_time_matrix, machine_matrix = processing_time_matrix[idx], machine_matrix[idx]
        else:
            self.name = name
            n, m, processing_time_matrix, machine_matrix = load_instance(name)
        
        # Initialize
        self.num_jobs, self.num_machines = n, m
        self.num_tasks = n * m
        self.processing_time_matrix = processing_time_matrix
        self.machine_matrix = machine_matrix
        self.pos_base = np.clip(np.random.rand(self.num_jobs), 0.5, 0.8)               # position base for render the graph plot

        # Action space and observation space
        # self.action_space = spaces.Discrete(self.num_jobs)
        # self.observation_space = 



    def step(self, action):
        # check if action is available
        job_id = action
        if self.job_done[job_id]:
            raise Exception("This job already done, Action not available!")

        
        op_id = self.op_to_assign[job_id]
        machine_id = self.machine_matrix[job_id][op_id]
        self.op_to_assign[job_id] += 1
        self.job_done = self.op_to_assign == self.num_jobs
        self.available_actions = np.where(~self.job_done)[0]

        node_id = "{}_{}".format(op_id, job_id)
        op = self.g.nodes[node_id]
        # print(f"Node {node_id} scheduled")

        op['scheduled'] = 1
        self.scheduled[job_id][op_id] = 1

        # determine start time of current op, by finding the earliest feasible time period to allocate it on the required machine.
        # add/remove edges on self.g
        # update ops_on_machine
        # update start time of other ops, only when edges had been removed
        ops_on_machine = self.assigned_jobs[machine_id]
        est = self.compute_est(node_id=node_id)
        if not ops_on_machine:
            ops_on_machine.append(node_id)
            self.update_start_time(node_id)
        else:
            inserted = False                                                        # current op would be inserted behind the op whose start time <= est
            for idx_to_insert_on in reversed(range(len(ops_on_machine))):
                node_id_to_insert_on = ops_on_machine[idx_to_insert_on]
                node_to_insert_on = self.g.nodes[node_id_to_insert_on]
                if node_to_insert_on['start'] <= est:
                    inserted = True
                    inserted_at_last = idx_to_insert_on + 1 == len(ops_on_machine)
                    
                    if inserted_at_last:
                        self.g.add_edge(ops_on_machine[-1], node_id)
                        ops_on_machine.insert(idx_to_insert_on + 1, node_id)
                        self.update_start_time(node_id)
                    else:
                        self.g.remove_edge(ops_on_machine[idx_to_insert_on], ops_on_machine[idx_to_insert_on + 1])
                        self.g.add_edge(ops_on_machine[idx_to_insert_on], node_id)
                        self.g.add_edge(node_id, ops_on_machine[idx_to_insert_on + 1])
                        ops_on_machine.insert(idx_to_insert_on + 1, node_id)
                        self.update_start_time(node_id)
                    break
                    

            if not inserted:                                            # not inserted behind, then inserted at the first
                self.g.add_edge(node_id, ops_on_machine[0])
                ops_on_machine.insert(0, node_id)
                self.update_start_time(node_id)

        # print(f"For job {job_id}, ops on machine: {ops_on_machine}")
        
        # Calculate self.lb
        for j in range(self.num_jobs):
            for i in range(self.num_machines):
                if self.scheduled[j][i]:                            # scheduled, lb = start + duration
                    this_node_id = "{}_{}".format(i, j)
                    this_node = self.g.nodes[this_node_id]
                    self.lb[j][i] = this_node['start'] + this_node['duration']
                else:                                               # not scheduled: lb = lb of prev op + duration
                    if i == 0:
                        self.lb[j][i] = self.processing_time_matrix[j][i]
                    else:
                        self.lb[j][i] = self.lb[j][i-1] + self.processing_time_matrix[j][i]
                
        view = nx.subgraph_view(self.g, filter_node=lambda x: x not in ['s','t'])
        adj = nx.adjacency_matrix(view).todense()

        scheduled_vec = self.scheduled.reshape(-1,1)
        lb_vec = self.lb.reshape((-1,1))
        feature = np.concatenate((scheduled_vec, lb_vec), axis=1)

        cur_max_lb = np.max(self.lb[:,-1])
        reward = self.max_lb - cur_max_lb
        self.max_lb = cur_max_lb

        done = self.job_done.all()
        
        # print("adj shape: ",adj.shape)
        # print("feature shape: ", feature.shape)
        candidate_operation_indexes = []
        for j in range(self.num_jobs):
            idx = self.op_to_assign[j] + j * self.num_machines
            idx = min(self.num_jobs*self.num_machines-1, idx)       
            # for the last operation of jobs, the idx would exceed the scope, 
            # but it only matters when it's the last job, because there is mask.
            candidate_operation_indexes.append(idx)

        mask = self.job_done
        state = np.array([adj, feature, mask, candidate_operation_indexes])
        info = {}
        if done:
            info["makespan"] = self.max_lb

        return state, reward, done, info




    def reset(self):
        # Create one list of assigned tasks per machine, where assigned tasks are actually graph node_ids
        self.assigned_jobs = collections.defaultdict(list)
        self.op_to_assign = np.zeros(self.num_jobs, dtype=np.int)
        self.job_done = self.op_to_assign == self.num_jobs
        self.available_actions = np.where(~self.job_done)[0]
        
        

        self.max_lb = 47             # an initial value, constant

        Times = self.processing_time_matrix
        n_processes = self.num_machines
        ####################### Create a DAG and add the edges that indicates the Process Constraints ##########################
        graph = nx.DiGraph()

        for j in range(self.num_jobs):
            for i in range(n_processes):
                if i == 0:
                    graph.add_edge("s", "{}_{}".format(i, j), weight=0)                                         # s 指向第一个工序
                    graph.add_edge("{}_{}".format(i, j), "{}_{}".format(i+1, j), weight=Times[j][i])
                elif i == n_processes-1:
                    graph.add_edge("{}_{}".format(i, j), "t", weight=Times[j][i])                           # 最后一个工序指向 t
                else:
                    graph.add_edge("{}_{}".format(i, j), "{}_{}".format(i+1, j), weight=Times[j][i])        # 同一个Job，前一个工序指向后一个工序

        ################################# Add node features ###################################
        start_node, end_node = graph.nodes['s'], graph.nodes['t']
        start_node['job'], start_node['op'], start_node['duration'] = None, None, 0
        end_node['job'], end_node['op'], end_node['duration'] = None, None, 0
        start_node['start'] = 0
        start_node['scheduled'] = 1

        for j in range(self.num_jobs):
            for i in range(n_processes):
                node_id = "{}_{}".format(i, j)
                
                node = graph.nodes[node_id]
                node['job'], node['op'], node['duration'] = j, i, Times[j][i]
                node['scheduled'] = 0

        end_node['scheduled'] = 0

        self.lb = np.cumsum(Times, axis=1)
        self.scheduled = np.zeros((self.num_jobs, self.num_machines))

        self.g = graph
        view = nx.subgraph_view(self.g, filter_node=lambda x: x not in ['s','t'])       # print(view.nodes)      # the order is correct
        adj = nx.adjacency_matrix(view).todense()

        scheduled_vec = self.scheduled.reshape(-1,1)
        lb_vec = self.lb.reshape((-1,1))
        feature = np.concatenate((scheduled_vec, lb_vec), axis=1)

        # print(feature.shape)        # (36, 2)
        candidate_operation_indexes = []
        for j in range(self.num_jobs):
            idx = self.op_to_assign[j] + j * self.num_machines
            candidate_operation_indexes.append(idx)
        mask = self.job_done
        state = np.array([adj, feature, mask, candidate_operation_indexes])
        
        return state


    def render(self):
        nodes = self.g.nodes(data=True)
        node_ids = [node[0] for node in nodes]
        positions = {}
        for node_id in node_ids:
            positions[node_id] = self.get_pos_from_id(node_id)

        colors = self.get_colors_from_nodelist(node_ids)
        labels = {}
        for node_id in node_ids:
            node_ = nodes[node_id]
            if node_['scheduled']:
                label = f"{node_['duration']}({node_['start']})"
            else:
                label = f"{node_['duration']}(?)"
            labels[node_id] = label
        
        ax = plt.figure().gca()
        ax.set_axis_off()
        options = {"node_size": 300, "labels": labels}
        nx.draw_networkx(self.g, pos=positions, node_color=colors, with_labels=True, **options)
        plt.show()


    def update_start_time(self, current_node_id):
        """Update start time of current node and its successors repeatedly.
        """
        current_node = self.g.nodes[current_node_id]
        current_node['start'] = self.compute_est(node_id=current_node_id)

        dangerous = set()
        for n in self.g.successors(current_node_id):
            if self.g.nodes[n]['scheduled']:
                dangerous.add(n)
        
        
        while len(dangerous) > 0:
            node_id = dangerous.pop()
            node = self.g.nodes[node_id]
            prev_start_time = node['start']
            node['start'] = self.compute_est(node_id=node_id)
            if prev_start_time != node['start']:
                for n in self.g.successors(node_id):
                    if self.g.nodes[n]['scheduled']:
                        dangerous.add(n)
                


    def compute_est(self, node_id):
        """Compute earliest start time of given node, which is the max end time of all prev nodes
            Hint: only works when edges are already updated"""
        est = 0
        for prev_node_id in self.g.predecessors(node_id):
            prev_node = self.g.nodes[prev_node_id]
            end = prev_node['start'] + prev_node['duration']
            est = max(end, est)

        return est



    def seed(self, seed):
        """Sets the seed for this environment's random number generator(s)."""
        np.random.seed(seed)
        # self.action_space.seed(seed)


    def get_pos_from_id(self, node_id):
        if node_id == 's':
            return [-3, self.num_jobs // 2]
        elif node_id == 't':
            return [self.num_machines + 10, self.num_jobs // 2]
        else:
            i, j = self.get_i_j_from_node_id(node_id)
            i += self.pos_base[j]
            i += i * self.pos_base[j]
            i += 1 if j%2==0 else 0
            j *= 1.2
            return i, j

    def get_colors_from_nodelist(self, nodelist):
        # ops on the same machine would have the same color
        color_map = np.linspace(0, 1, num=self.num_machines + 2)
        colors = []
        for node_id in nodelist:
            if node_id == 's':
                colors.append(color_map[0])
            elif node_id == 't':
                colors.append(color_map[-1])
            else:
                i, j = self.get_i_j_from_node_id(node_id)
                machine_id = self.machine_matrix[j][i]
                colors.append(color_map[machine_id + 1])

        return colors

    
    def get_i_j_from_node_id(self, node_id):
        i, j = node_id[0], node_id[-1]
        i, j = int(i), int(j)
        return i, j


if __name__ == '__main__':

    env  = GIN_JsspEnv("ft06")
    env.seed(0)

    adj, feature, legal_actions = env.reset()
    done = False

    while not done:
        a = np.random.choice(legal_actions)
        state, reward, done, _ = env.step(a)
        adj, feature, legal_actions, candidate_operation_indexes = state

        

    

