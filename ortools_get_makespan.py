import os
import numpy as np
import pandas as pd
import collections
import plotly.express as px
from ortools.sat.python import cp_model
import warnings
warnings.filterwarnings("ignore")


def min_makespan(instance: str = 'ft06', times=None, machines=None, print_results=False, show_gantt=False):
    """
    Return:
    (status, obj_val)
    status can be 'Feasible' or 'Optimal'
    """
    if not (times and machines):
        times, machines = load_instance(instance)
    jobs_data = []
    for job_id, ms in enumerate(machines):
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


        if print_results:
            # Create per machine output lines.
            output = ''
            for machine in all_machines:
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

            print('Solution:')
            if status == cp_model.OPTIMAL:
                print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
            if status == cp_model.FEASIBLE:
                print(f'Feasible Schedule Length: {solver.ObjectiveValue()}')
            print(output)


        if show_gantt:
            df = []
            for machine in all_machines:
                for task in assigned_jobs[machine]:
                    start, job, op, duration = task.start, task.job, task.index, task.duration
                    finish = start + duration
                    operation = f"Job_{job}_Op{op}"
                    df.append(dict(Machine=machine, Start=start, Finish=finish, Operation=operation, Job=job))
            df = pd.DataFrame(df)

            # 处理时间
            datetime = pd.Timestamp('20230101 00:00:00')
            for i in range(len(df)):
                df['Start'].loc[i] = datetime + pd.Timedelta(minutes = df['Start'].loc[i])
                df['Finish'].loc[i] = datetime + pd.Timedelta(minutes = df['Finish'].loc[i])


            fig = px.timeline(df, x_start="Start", x_end="Finish", y="Machine", color="Job", hover_name="Operation")
            fig.update_yaxes(autorange="reversed") # otherwise tasks are listed from the bottom up
            fig.show()


        if status == cp_model.OPTIMAL:
            return "Optimal", solver.ObjectiveValue()
        elif status == cp_model.FEASIBLE:
            return "Feasible", solver.ObjectiveValue()
    else:
        return False
    


def load_instance(instance: str = 'ft06'):
    """
    Loads the specified JSSP instance and gets the matrix used to describe that instance.

    Args:
        instance: Instance name. Values for abz5-9, ft06, ft10, ft20, la01-la40, orb01-orb10, swv01-swv20, yn1-yn4.

    Returns:
        * N: Number of jobs.
        
        * M: Number of machines
        
        * time_mat: The processing time matrix. Shape is (N, M).
        
        * machine_mat: The machine processing matrix. Shape is (N, M).
    Example:
        >>> N, M, time_mat, machine_mat = load_instance('abz5')
    """
    path = os.path.join('jobshop', 'jobshop.txt')
    
    assert os.path.exists(path)
    assert os.path.isfile(path)

    # time为一个实例的时间表，machine为一个实例的机器表
    # times为该文件包含的所有实例，machines同理
    time = []
    times = []
    machine = []
    machines = []
    with open(path, "r") as file:
        lines = file.readlines()
        start = -1

        for i, line in enumerate(lines):
            if line.find("instance " + instance) != -1:
                start = i + 4
                break
        line = lines[start]
        line = line.strip()
        line = list(filter(None, line.split(' ')))
        assert len(line) == 2
        N = int(line[0])
        M = int(line[1])
        start += 1

        for i in range(N):
            machine = []
            time = []
            line = lines[start + i]
            line = line.strip()
            line = list(filter(None, line.split(' ')))
            line = [int(x) for x in line]

            for j in range(M):
                machine.append(line[2 * j])
                time.append(line[2 * j + 1])

            times.append(time)
            machines.append(machine)

    times = np.array(times)
    machines = np.array(machines)
    return times, machines



def load_random(N, M):
    """
    Load several randomly generated JSSP instances according to certain rules, 
    and obtain the relevant information describing these instances.

    Args:
        * N: number of jobs for the instance to be generated. Optional values: {15, 20, 30, 50, 100}.
        
        * M: Number of machines to generate instances. Optional values: {15, 20}.

    Returns:
        * I: Number of instances.

        * N: Number of jobs.
        
        * M: Number of machines
        
        * time_mat: The processing time matrix. Shape is (I, N, M).
        
        * machine_mat: The machine processing matrix. Shape is (I, N, M).
    Example:
        >>> I, N, M, time_mat, machine_mat = load_random(30, 15)
    """

    path = os.path.join('jobshop','tai{}_{}.txt'.format(N, M))
    # print(path)
    assert os.path.exists(path)
    assert os.path.isfile(path)

    state = "start"

    time = []
    times = []
    machine = []
    machines = []
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) <= 0:
                continue
            if line[0].isalpha():
                state = __next_state(state)
                continue
            nums = list(filter(None, line.split(' ')))

            if "row" == state:
                if machine:
                    machines.append(machine)
                    machine = []
                    times.append(time)
                    time = []
            elif "times" == state:
                time.append(
                    [int(num) for num in nums]
                )
            elif "machines" == state:
                machine.append(
                    [int(num) for num in nums]
                )
            else:
                raise RuntimeError('State error.')

        machines.append(machine)
        times.append(time)

    I = len(times)
    N = len(times[0])
    M = len(times[0][0])
    times = np.array(times)
    machines = np.array(machines)
    return I, N, M, times, machines


def __next_state(state):
    if "start" == state:
        return "row"
    elif "row" == state:
        return 'times'
    elif "times" == state:
        return 'machines'
    elif "machines" == state:
        return "row"
    else:
        return "error"