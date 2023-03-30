import os
import numpy as np
import pandas as pd
import torch
import collections
import networkx as nx
from ortools.sat.python import cp_model
from matplotlib import pyplot as plt
from gymjsp.jsspenv import HeuristicJsspEnv
from tianshou.data import Batch
import warnings
warnings.filterwarnings("ignore")
from gymjsp.orliberty import load_random, load_instance


class ORtools_scheduler:
    def __init__(self, instance_name=None, max_time=300, times=None, machines=None) -> None:
        self.instance_name = instance_name
        self.max_time = max_time
        self.n_jobs = None
        self.n_machines = None
        self.times = None
        self.machines = None
        if not instance_name[0].isdigit():
            self.load_instance_from_file(self.instance_name)
        else:
            l = instance_name.split('_')
            num_jobs, num_machines, idx = int(l[0]), int(l[1]), int(l[2])
            i, n, m, processing_time_matrix, machine_matrix = load_random(num_jobs, num_machines)
            processing_time_matrix, machine_matrix = processing_time_matrix[idx], machine_matrix[idx]
            self.load_instance_from_matrix(times=processing_time_matrix, machines=machine_matrix)


        self.status = None
        self.obj_val = None

        self.assigned_jobs = None


    def load_instance_from_file(self, instance_name):
        path = os.path.join('jobshop', 'jobshop.txt')
        time = []
        times = []
        machine = []
        machines = []

        with open(path, "r") as file:
            lines = file.readlines()
            start = -1
            for i, line in enumerate(lines):
                if line.find("instance " + instance_name) != -1:
                    start = i + 4
                    break
            line = lines[start]
            line = line.strip()
            line = list(filter(None, line.split(' ')))
            assert len(line) == 2
            self.n_jobs = int(line[0])
            self.n_machines = int(line[1])
            start += 1
            for i in range(self.n_jobs):
                machine = []
                time = []
                line = lines[start + i]
                line = line.strip()
                line = list(filter(None, line.split(' ')))
                line = [int(x) for x in line]
                for j in range(self.n_machines):
                    machine.append(line[2 * j])
                    time.append(line[2 * j + 1])
                times.append(time)
                machines.append(machine)

        self.times = np.array(times)
        self.machines = np.array(machines)



    def load_instance_from_matrix(self, times, machines):
        self.n_jobs = len(times)
        self.n_machines = len(times[0])
        self.times = np.array(times)
        self.machines = np.array(machines)


    def compute_makespan(self, shifted_time=None):
        """
        shifted_time: shape of (n_jobs, n_ops)
        """
        assigned_jobs = self.assigned_jobs
        Times = shifted_time if shifted_time is not None else self.times.copy()
        ####################### Create a DAG and add the edges that indicates the Process Constraints ##########################
        n_processes = self.n_machines
        graph = nx.DiGraph()
        for j in range(self.n_jobs):
            for i in range(n_processes):
                if i == 0:
                    graph.add_edge("s", "{}_{}".format(i, j), weight=0)                                         # s 指向第一个工序
                    graph.add_edge("{}_{}".format(i, j), "{}_{}".format(i+1, j), weight=Times[j][i])
                elif i == n_processes-1:
                    graph.add_edge("{}_{}".format(i, j), "t", weight=Times[j][i])                           # 最后一个工序指向 t
                else:
                    graph.add_edge("{}_{}".format(i, j), "{}_{}".format(i+1, j), weight=Times[j][i])        # 同一个Job，前一个工序指向后一个工序

        ################################## Add the edges that indicates the schedule scheme ############################
        for machine in range(self.n_machines):
            assigned_tasks = assigned_jobs[machine]
            assigned_jobs[machine].sort()               # sort the task by start time!
            for i, task in enumerate(assigned_tasks):
                next_task = assigned_tasks[i+1] if i<len(assigned_tasks)-1 else None
                if next_task:
                    j, i = task.job, task.index
                    j_, i_ = next_task.job, next_task.index
                    duration = Times[j][i]
                    graph.add_edge("{}_{}".format(i, j), "{}_{}".format(i_, j_), weight=duration)

        longest_path_length = nx.dag_longest_path_length(graph)
        #if shifted_time is None:
        #    assert longest_path_length == self.obj_val, "Without shift time, caculate makespan wrongly!"

        return longest_path_length


    def get_optimal_of_new_time_mat(self, times):
        return self.optimize_(times)

    def optimize_(self, times):
        """
        Get the schedule scheme through OR-tools
        """
        jobs_data = []
        for job_id, ms in enumerate(self.machines):
            l = []
            for op_id, machine_id in enumerate(ms):
                l.append((machine_id, times[job_id][op_id]))
            jobs_data.append(l)

        machines_count = 1 + max(task[0] for job in jobs_data for task in job)
        all_machines = range(machines_count)
        # Computes horizon dynamically as the sum of all durations.
        horizon = sum(task[1] for job in jobs_data for task in job)

        model = cp_model.CpModel()

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple('assigned_task_type',
                                                    'start job index duration')

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(start=start_var,
                                                    end=end_var,
                                                    interval=interval_var)
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in all_machines:
            model.AddNoOverlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)


        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
            for job_id, job in enumerate(jobs_data)
        ])
        model.Minimize(obj_var)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_time
        status = solver.Solve(model)


        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(start=solver.Value(
                            all_tasks[job_id, task_id].start),
                                        job=job_id,
                                        index=task_id,
                                        duration=task[1]))

        #self.obj_val = solver.ObjectiveValue()
        #self.status = status
        #self.assigned_jobs = assigned_jobs

        return (status == cp_model.OPTIMAL), solver.ObjectiveValue()

    def optimize(self):
        """
        Get the schedule scheme through OR-tools
        """
        jobs_data = []
        for job_id, ms in enumerate(self.machines):
            l = []
            for op_id, machine_id in enumerate(ms):
                l.append((machine_id, self.times[job_id][op_id]))
            jobs_data.append(l)

        machines_count = 1 + max(task[0] for job in jobs_data for task in job)
        all_machines = range(machines_count)
        # Computes horizon dynamically as the sum of all durations.
        horizon = sum(task[1] for job in jobs_data for task in job)

        model = cp_model.CpModel()

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple('assigned_task_type',
                                                    'start job index duration')

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                duration = task[1]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                    'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(start=start_var,
                                                    end=end_var,
                                                    interval=interval_var)
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in all_machines:
            model.AddNoOverlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(jobs_data):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id + 1].start >= all_tasks[job_id, task_id].end)


        # Makespan objective.
        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
            for job_id, job in enumerate(jobs_data)
        ])
        model.Minimize(obj_var)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.max_time
        status = solver.Solve(model)


        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Create one list of assigned tasks per machine.
            assigned_jobs = collections.defaultdict(list)
            for job_id, job in enumerate(jobs_data):
                for task_id, task in enumerate(job):
                    machine = task[0]
                    assigned_jobs[machine].append(
                        assigned_task_type(start=solver.Value(
                            all_tasks[job_id, task_id].start),
                                        job=job_id,
                                        index=task_id,
                                        duration=task[1]))

        self.obj_val = solver.ObjectiveValue()
        self.status = status
        self.assigned_jobs = assigned_jobs


    def store_time_mat(self, filename):
        times = np.array(self.times)
        np.save(filename, times)

    def load_time_mat(self, filename):
        self.times = np.load(filename)


    def store_solution(self, filename=None):
        assigned_jobs = self.assigned_jobs
        data = {"machine_id":[], "start":[], "job":[], "index":[], "duration":[]}
        df = pd.DataFrame(data)
        for machine_id in range(len(assigned_jobs)):
            tasks = assigned_jobs[machine_id]
            for task in tasks:
                start = task.start
                job = task.job
                index = task.index
                duration = task.duration
                df = df.append({"machine_id":machine_id, "start":start, "job":job, "index":index, "duration":duration}, ignore_index=True)
        df = df.astype(int)
        if not filename:
            df.to_csv(f"sols/{self.instance_name}.csv")
        else:
            df.to_csv(filename)


    def read_solution(self, filename=None):
        if filename:
            df = pd.read_csv(filename, index_col=0)
        else:
            df = pd.read_csv(f"sols/{self.instance_name}.csv", index_col=0)
        assigned_jobs = collections.defaultdict(list)
        assigned_task_type = collections.namedtuple('assigned_task_type',
                                                            'start job index duration')
        for data in df.values.tolist():
            machine_id, start, job, index, duration = data
            assigned_jobs[machine_id].append(assigned_task_type(start=start, job=job, index=index, duration=duration))

        self.assigned_jobs = assigned_jobs

    def shifted_time(self, prob, scale):
        """
        For each duration t, if p < prob, then set t = round(t+N(0,1)*scale)
        """
        times = self.times.copy()
        for i in range(len(times)):
            for j in range(len(times[0])):
                if np.random.random()  < prob:
                    deviation = np.random.randn() * scale
                    times[i][j] += np.round(deviation)
                    if times[i][j] == 0:
                        times[i][j] = 0

        return times
    

    def shifted_time_(self, random_rate=0.1, cv=0.1):
        
        times = self.times.copy()

        for job_id in range(len(times)):
            for step_id in range(len(times[0])):
                if np.random.random() < random_rate:
                    x = times[job_id][step_id]

                    bias = np.random.normal(loc=0, scale=cv*x)      # cv = standard deviation / mean
                    bias = min(max(-1, bias), 1)
                    
                    x *= 1 + bias
                    x = int(x)
                    x = 1 if x == 0 else x
                    #times[job_id][step_id] *= 1 + bias
                    #times[job_id][step_id] = int(times[job_id][step_id])
                    times[job_id][step_id] = x
        
        return times


    def shift_time(self, shifted_time):
        self.times = shifted_time

    def policy_makespan(self, model_type, model, shifted_time=None):
        test_env = HeuristicJsspEnv(self.instance_name, shifted_time=shifted_time)

        state = test_env.reset(random=False)
        done = False
        score = 0

        while not done:
            if model_type == 'dqn':
                action = model(torch.FloatTensor(state).to("cuda")).argmax()
                action = action.detach().cpu().numpy()
            elif model_type == 'ppo':
                action = model(Batch(obs=state[np.newaxis, :])).act.item()
            
            next_state, reward, done, info = test_env.step(action)

            state = next_state
            score += reward


        return info["makespan"]


    def compare_dqn_ortools(self, model, n):
        times_list = [self.shifted_time_() for _ in range(n)]
        dqn_makespans = []
        ortools_makespans = []
        for time in times_list:
            dqn_makespans.append(self.policy_makespan(model=model, shifted_time=time))
            ortools_makespans.append(self.compute_makespan(shifted_time=time))
        
        dqn_makespans, ortools_makespans = np.array(dqn_makespans), np.array(ortools_makespans)

        return np.mean(dqn_makespans), np.mean(ortools_makespans)


    def fluctuate_makespan(self, n, prob, scale):
        """OR-tools faces the dynamic processing time"""
        makespans = [self.compute_makespan(shifted_time=self.shifted_time(prob, scale)) for _ in range(n)]
        plt.bar(range(n), makespans)
        plt.hlines(self.obj_val, -1, n, color="red")
        plt.show()


    def show_gantt(self):
        pass

    def print_results(self):
        assigned_jobs = self.assigned_jobs
        # Create per machine output lines.
        output = ''
        for machine in range(self.n_machines):
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_task_%i' % (assigned_task.job,
                                        assigned_task.index)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        status = self.status
        print('Solution:')
        if status == cp_model.OPTIMAL:
            print(f'Optimal Schedule Length: {self.obj_val}')
        if status == cp_model.FEASIBLE:
            print(f'Feasible Schedule Length: {self.obj_val}')
        print(output)
