import os
import gym
import collections
import numpy as np
import networkx as nx
from gym import spaces
from orliberty import load_random, load_instance




assigned_task_type = collections.namedtuple('assigned_task_type', 'start job op duration')




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
        self.processing_time_matrix = processing_time_matrix
        self.machine_matrix = machine_matrix

        # Create one list of assigned tasks per machine, where assigned tasks are actually graph nodes
        self.assigned_jobs = collections.defaultdict(list)
        self.op_to_assign = np.zeros(self.num_jobs, dtype=np.int)


        # Action space and observation space
        self.action_space = spaces.Discrete(self.num_jobs)
        # self.observation_space = 



    def step(self, action):
        job_id = action
        op_id = self.op_to_assign[job_id]
        machine_id = self.machine_matrix[job_id][op_id]
        duration = self.processing_time_matrix[job_id][op_id]
        self.op_to_assign[job_id] += 1
        node_id = "{}_{}".format(op_id, job_id)
        op = self.g.nodes[node_id]



        # determine start time of current op, by finding the earliest feasible time period to allocate it on the required machine.
        ops_on_machine = self.assigned_jobs[machine_id]
        if not ops_on_machine:
            op['start'] = 0
            ops_on_machine.append(op)
        else:
            # compute earliest start time of current op is maximum end time of all prev nodes
            est = 0
            for prev_node_id in self.g.predecessors(node_id):
                prev_node = self.g.nodes[prev_node_id]
                print(prev_node)
                end = prev_node['start'] + prev_node['duration']
                est = max(end, est)

            inserted = False
            for idx_to_insert_on in reversed(range(len(ops_on_machine))):
                if ops_on_machine[idx_to_insert_on]['start'] <= est:
                    op['start'] = est
                    ops_on_machine.insert(idx_to_insert_on + 1, op)
                    inserted = True
            if not inserted:
                op['start'] = 0
                ops_on_machine.insert(0, op)

        # 更新edges，然后再更新['start']
            


        #self.assigned_jobs[machine_id].append(assigned_task_type(start=, job=job_id, op=op_id, duration=duration))


        #return state, reward, done, info

    def reset(self):
        Times = self.processing_time_matrix
        n_processes = self.num_machines

        ####################### Create a DAG and add the edges that indicates the Process Constraints ##########################
        graph = nx.DiGraph()
        for j in range(self.num_jobs):
            for i in range(n_processes):
                # Add edges
                if i == 0:
                    graph.add_edge("s", "{}_{}".format(i, j), weight=0)                                         # s 指向第一个工序
                    graph.add_edge("{}_{}".format(i, j), "{}_{}".format(i+1, j), weight=Times[j][i])
                elif i == n_processes-1:
                    graph.add_edge("{}_{}".format(i, j), "t", weight=Times[j][i])                           # 最后一个工序指向 t
                else:
                    graph.add_edge("{}_{}".format(i, j), "{}_{}".format(i+1, j), weight=Times[j][i])        # 同一个Job，前一个工序指向后一个工序

        # Add node features:  job, op, duration
        start_node, end_node = graph.nodes['s'], graph.nodes['t']
        start_node['job'], start_node['op'], start_node['duration'] = None, None, 0
        end_node['job'], end_node['op'], end_node['duration'] = None, None, 0

        for j in range(self.num_jobs):
            for i in range(n_processes):
                node = graph.nodes["{}_{}".format(i, j)]
                node['job'], node['op'], node['duration'] = j, i, Times[j][i]
    
        start_node['start'] = 0

        self.g = graph

        return graph

    
    def render(self):
        """
        同步显示，析取图、甘特图随着agent做动作的动画
        """
        pass


    def seed(self, seed):
        """Sets the seed for this environment's random number generator(s)."""
        np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)



if __name__ == '__main__':
    env  = GIN_JsspEnv("ft06")
    g = env.reset()
    env.step(1)
    nodes = g.nodes(data=True)
    n1 = nodes['s']
    n2 = nodes['3_2']

    for k in g.predecessors('1_0'):
        print(k)
    

    l = [1,2,3,4]
    l.insert(2, 'o')
    print(l)