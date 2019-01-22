import spynnaker8 as p
import sys, os
import time
import numpy as np
from spinn_bandit.python_models.bandit import Bandit
from spinn_arm.python_models.arm import Arm
from copy import deepcopy
from spinn_front_end_common.utilities.globals_variables import get_simulator
import traceback
import csv
import pathos.multiprocessing
from spinn_front_end_common.utilities import globals_variables
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from multiprocessing import Queue
from typing import Callable
from typing import Iterable
from typing import Dict
from typing import Any

max_fail_score = -int(runtime / exposure_time)
setup_retry_time = 60
new_split = 100

# stdp_model = p.STDPMechanism(
#     timing_dependence=p.SpikePairRule(tau_plus=20., tau_minus=20.0, A_plus=0.003, A_minus=0.003),
#     weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=0.1))


class ProcessKillingExecutor():
    """
    The ProcessKillingExecutor works like an `Executor
    <https://docs.python.org/dev/library/concurrent.futures.html#executor-objects>`_
    in that it uses a bunch of processes to execute calls to a function with
    different arguments asynchronously.

    But other than the `ProcessPoolExecutor
    <https://docs.python.org/dev/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`_,
    the ProcessKillingExecutor forks a new Process for each function call that
    terminates after the function returns or if a timeout occurs.

    This means that contrary to the Executors and similar classes provided by
    the Python Standard Library, you can rely on the fact that a process will
    get killed if a timeout occurs and that absolutely no side can occur
    between function calls.

    Note that descendant processes of each process will not be terminated
    they will simply become orphaned.
    """

    def __init__(self, max_workers=None):
        self.processes = max_workers or os.cpu_count()

    def map(self,
            func,
            iterable,
            timeout=None,
            callback_timeout=None,
            daemon=True):
        """
        :param func: the function to execute
        :param iterable: an iterable of function arguments
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called, if the task
                times out. It gets the same arguments as the original function
        :param daemon: define the child process as daemon
        """
        executor = ThreadPoolExecutor(max_workers=self.processes)
        params = ({'func': func, 'fn_args': p_args, "p_kwargs": {},
                   'timeout': timeout, 'callback_timeout': callback_timeout,
                   'daemon': daemon} for p_args in iterable)
        return executor.map(self._submit_unpack_kwargs, params)

    def _submit_unpack_kwargs(self, params):
        """ unpack the kwargs and call submit """

        return self.submit(**params)

    def submit(self,
               func,
               fn_args,
               p_kwargs,
               timeout,
               callback_timeout,
               daemon):
        """
        Submits a callable to be executed with the given arguments.
        Schedules the callable to be executed as func(*args, **kwargs) in a new
         process.
        :param func: the function to execute
        :param fn_args: the arguments to pass to the function. Can be one argument
                or a tuple of multiple args.
        :param p_kwargs: the kwargs to pass to the function
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called with the same
                arguments, if the task times out.
        :param daemon: run the child process as daemon
        :return: the result of the function, or None if the process failed or
                timed out
        """
        p_args = fn_args if isinstance(fn_args, tuple) else (fn_args,)
        queue = Queue()
        p = Process(target=self._process_run,
                    args=(queue, func, fn_args,), kwargs=p_kwargs)

        if daemon:
            p.deamon = True

        p.start()
        p.join(timeout=timeout)
        if not queue.empty():
            return queue.get()
        if callback_timeout:
            callback_timeout(*p_args, **p_kwargs)
        if p.is_alive():
            p.terminate()
            p.join()

    @staticmethod
    def _process_run(queue, func=None,
                     *args, **kwargs):
        """
        Executes the specified function as func(*args, **kwargs).
        The result will be stored in the shared dictionary
        :param func: the function to execute
        :param queue: a Queue
        """
        queue.put(func(*args, **kwargs))


# def some_task(n, *args, **kwargs):
#     import time
#     time.sleep(n/4)
#     return n ** 2
#
#
# if __name__ == "__main__":
#     def fun_timeout(n):
#         print('timeout:', n)
#     executor = ProcessKillingExecutor(max_workers=2)
#     generator = executor.map(some_task, [1, 1, 2, 2, 3, 3, 4, 4], timeout=2,
#                              callback_timeout=fun_timeout)
#     for elem in generator:
#         print(elem)

def split_ex_in(connections):
    excite = []
    inhib = []
    for conn in connections:
        if conn[2] > 0:
            excite.append(conn)
        else:
            inhib.append(conn)
    for conn in inhib:
        conn[2] *= -1
    return excite, inhib

def split_plastic(connections):
    plastic = []
    non_plastic = []
    for conn in connections:
        if conn[4] == 'plastic':
            plastic.append([conn[0], conn[1], conn[2], conn[3]])
        else:
            non_plastic.append([conn[0], conn[1], conn[2], conn[3]])
    return plastic, non_plastic

def get_scores(game_pop, simulator):
    g_vertex = game_pop._vertex
    scores = g_vertex.get_data(
        'score', simulator.no_machine_time_steps, simulator.placements,
        simulator.graph_mapper, simulator.buffer_manager, simulator.machine_time_step)
    return scores.tolist()

def thread_bandit(connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                  reward=0, size_f=False, spike_f=False, top=True):
    def helper(args):
        return bandit_test(*args)

    def func_timeout(n):
        print('timeout:', n)

    step_size = len(connections) / split
    if step_size == 0:
        step_size = 1
    if isinstance(arms[0], list):
        connection_threads = []
        all_configs = [[[connections[x:x + step_size], arm, split, runtime, exposure_time, noise_rate, noise_weight,
                         reward, spike_f, np.random.randint(1000000000)] for x in xrange(0, len(connections), step_size)] for arm in arms]
        for arm in all_configs:
            for config in arm:
                connection_threads.append(config)
    else:
        connection_threads = [[connections[x:x + step_size], arms, split, runtime, exposure_time, noise_rate,
                               noise_weight, reward, spike_f, np.random.randint(1000000000)] for x in xrange(0, len(connections), step_size)]

    executor = ProcessKillingExecutor(max_workers=len(connection_threads))

    generator = executor.map(helper, connection_threads, timeout=300, callback_timeout=func_timeout)

    # pool = pathos.multiprocessing.Pool(processes=len(connection_threads))
    #
    # pool_result = pool.map(func=helper, iterable=connection_threads)
    #
    # pool.close()
    pool_result = []
    i = 0
    for elem in generator:
        if i > len(connection_threads):
            break
        if elem == 'fail' or elem == None and len(connection_threads[i][0]) > 1:
            print "splitting ", len(connection_threads[i][0]), " into ", new_split, " pieces"
            problem_arms = connection_threads[i][1]
            pool_result.append(thread_bandit(connection_threads[i][0], problem_arms, new_split, runtime,
                                                exposure_time, noise_rate, noise_weight, reward, spike_f, top=False))
        else:
            pool_result.append(elem)
        i += 1

    # for i in range(len(pool_result)):
    #     if pool_result[i] == 'fail' and len(connection_threads[i][0]) > 1:
    #         print "splitting ", len(connection_threads[i][0]), " into ", new_split, " pieces"
    #         problem_arms = connection_threads[i][1]
    #         pool_result[i] = thread_bandit(connection_threads[i][0], problem_arms, new_split, runtime,
    #                                             exposure_time, noise_rate, noise_weight, reward, spike_f, top=False)

    agent_fitness = []
    for thread in pool_result:
        if isinstance(thread, list):
            for result in thread:
                agent_fitness.append(result)
        else:
            agent_fitness.append(thread)

    if isinstance(arms[0], list) and top:
        copy_fitness = deepcopy(agent_fitness)
        agent_fitness = []
        for i in range(len(arms)):
            arm_results = []
            for j in range(pop_size):
                arm_results.append(copy_fitness[(i * pop_size) + j])
            agent_fitness.append(arm_results)
        if size_f:
            arm_results = []
            for i in range(pop_size):
                arm_results.append(connections[i][2] + connections[i][5])
            agent_fitness.append(arm_results)
    return agent_fitness

def connect_to_arms(pre_pop, from_list, arms, r_type, plastic, stdp_model):
    arm_conn_list = []
    for i in range(len(arms)):
        arm_conn_list.append([])
    for conn in from_list:
        arm_conn_list[conn[1]].append((conn[0], 0, conn[1], conn[2]))
        # print "out:", conn[1]
        # if conn[1] == 2:
        #     print '\nit is possible\n'
    for i in range(len(arms)):
        if len(arm_conn_list[i]) != 0:
            if plastic:
                p.Projection(pre_pop, arms[i], p.FromListConnector(arm_conn_list[i]),
                             receptor_type=r_type, synapse_type=stdp_model)
            else:
                p.Projection(pre_pop, arms[i], p.FromListConnector(arm_conn_list[i]),
                             receptor_type=r_type)

def bandit_test(connections, arms, split=4, runtime=2000, exposure_time=200, noise_rate=100, noise_weight=0.01,
                reward=0, spike_f=False, seed=0):
    np.random.seed(seed)
    sleep = 10 * np.random.random()
    # time.sleep(sleep)
    max_attempts = 2
    try_except = 0
    while try_except < max_attempts:
        bandit = []
        bandit_count = -1
        bandit_arms = []
        excite = []
        excite_count = -1
        excite_marker = []
        inhib = []
        inhib_count = -1
        inhib_marker = []
        failures = []
        start = time.time()
        try_count = 0
        while time.time() - start < setup_retry_time:
            try:
                p.setup(timestep=1.0, min_delay=1, max_delay=127)
                p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
                print "\nfinished setup seed = ", seed, "\n"
                break
            except:
                traceback.print_exc()
                sleep = 1 * np.random.random()
                time.sleep(sleep)
            print "\nsetup", try_count, " seed = ", seed, "\n", "\n"
            try_count += 1
        print "\nfinished setup seed = ", seed, "\n"
        print config
        for i in range(len(connections)):
            [in2e, in2i, in2in, in2out, e2in, i2in, e_size, e2e, e2i, i_size,
             i2e, i2i, e2out, i2out, out2e, out2i, out2in, out2out] = connections[i]
            if len(in2e) == 0 and len(in2i) == 0 and len(in2out) == 0:
                failures.append(i)
                print "agent {} was not properly connected to the game".format(i)
            else:
                bandit_count += 1
                # removed_prob_arms = [1 for i in range(len(arms))]
                bandit.append(
                    p.Population(len(arms), Bandit(arms, exposure_time, reward_based=reward,
                                                   rand_seed=[np.random.randint(0xffff) for k in range(4)],
                                                   label='bandit_pop_{}-{}'.format(bandit_count, i))))
                # added to ensure that the arms and bandit are connected to and from something
                null_pop = p.Population(1, p.IF_cond_exp(), label='null{}'.format(i))
                p.Projection(bandit[bandit_count], null_pop, p.AllToAllConnector())
                arm_collection = []
                for j in range(len(arms)):
                    arm_collection.append(p.Population(int(np.ceil(np.log2(len(arms)))),
                                                       Arm(arm_id=j, reward_delay=exposure_time,
                                                           rand_seed=[np.random.randint(0xffff) for k in range(4)],
                                                           no_arms=len(arms), arm_prob=1),
                                                       label='arm_pop{}:{}:{}'.format(bandit_count, i, j)))
                    p.Projection(arm_collection[j], bandit[bandit_count], p.AllToAllConnector(), p.StaticSynapse())
                    p.Projection(null_pop, arm_collection[j], p.AllToAllConnector())
                bandit_arms.append(arm_collection)
                if e_size > 0:
                    excite_count += 1
                    excite.append(
                        p.Population(e_size, p.IF_cond_exp(), label='excite_pop_{}-{}'.format(excite_count, i)))
                    if noise_rate:
                        excite_noise = p.Population(e_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(excite_noise, excite[excite_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if spike_f:
                        excite[excite_count].record('spikes')
                    excite_marker.append(i)
                if i_size > 0:
                    inhib_count += 1
                    inhib.append(p.Population(i_size, p.IF_cond_exp(), label='inhib_pop_{}-{}'.format(inhib_count, i)))
                    if noise_rate:
                        inhib_noise = p.Population(i_size, p.SpikeSourcePoisson(rate=noise_rate))
                        p.Projection(inhib_noise, inhib[inhib_count], p.OneToOneConnector(),
                                     p.StaticSynapse(weight=noise_weight), receptor_type='excitatory')
                    if spike_f:
                        inhib[inhib_count].record('spikes')
                    inhib_marker.append(i)
                stdp_model = p.STDPMechanism(
                    timing_dependence=p.SpikePairRule(
                        tau_plus=20., tau_minus=20.0, A_plus=0.02, A_minus=0.02),
                    weight_dependence=p.AdditiveWeightDependence(w_min=0, w_max=0.1))
                if len(in2e) != 0:
                    [in_ex, in_in] = split_ex_in(in2e)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(bandit[bandit_count], excite[excite_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(bandit[bandit_count], excite[excite_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(bandit[bandit_count], excite[excite_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(bandit[bandit_count], excite[excite_count], p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(in2i) != 0:
                    [in_ex, in_in] = split_ex_in(in2i)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            p.Projection(bandit[bandit_count], inhib[inhib_count], p.FromListConnector(plastic),
                                         receptor_type='excitatory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(bandit[bandit_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                         receptor_type='excitatory')
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            p.Projection(bandit[bandit_count], inhib[inhib_count], p.FromListConnector(plastic),
                                         receptor_type='inhibitory', synapse_type=stdp_model)
                        if len(non_plastic) != 0:
                            p.Projection(bandit[bandit_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                         receptor_type='inhibitory')
                if len(in2out) != 0:
                    [in_ex, in_in] = split_ex_in(in2out)
                    if len(in_ex) != 0:
                        [plastic, non_plastic] = split_plastic(in_ex)
                        if len(plastic) != 0:
                            connect_to_arms(bandit[bandit_count], plastic, bandit_arms[bandit_count], 'excitatory',
                                            True, stdp_model=stdp_model)
                        if len(non_plastic) != 0:
                            connect_to_arms(bandit[bandit_count], non_plastic, bandit_arms[bandit_count], 'excitatory',
                                            True, stdp_model=stdp_model)
                    if len(in_in) != 0:
                        [plastic, non_plastic] = split_plastic(in_in)
                        if len(plastic) != 0:
                            connect_to_arms(bandit[bandit_count], plastic, bandit_arms[bandit_count], 'inhibitory',
                                            True, stdp_model=stdp_model)
                        if len(non_plastic) != 0:
                            connect_to_arms(bandit[bandit_count], non_plastic, bandit_arms[bandit_count], 'inhibitory',
                                            True, stdp_model=stdp_model)
                if len(e2e) != 0:
                    [plastic, non_plastic] = split_plastic(e2e)
                    if len(plastic) != 0:
                        p.Projection(excite[excite_count], excite[excite_count], p.FromListConnector(plastic),
                                     receptor_type='excitatory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], excite[excite_count], p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(e2i) != 0:
                    [plastic, non_plastic] = split_plastic(e2i)
                    if len(plastic) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count], p.FromListConnector(plastic),
                                     receptor_type='excitatory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(excite[excite_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                     receptor_type='excitatory')
                if len(i2e) != 0:
                    [plastic, non_plastic] = split_plastic(i2e)
                    if len(plastic) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count], p.FromListConnector(plastic),
                                     receptor_type='inhibitory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], excite[excite_count], p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(i2i) != 0:
                    [plastic, non_plastic] = split_plastic(i2i)
                    if len(plastic) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count], p.FromListConnector(plastic),
                                     receptor_type='inhibitory', synapse_type=stdp_model)
                    if len(non_plastic) != 0:
                        p.Projection(inhib[inhib_count], inhib[inhib_count], p.FromListConnector(non_plastic),
                                     receptor_type='inhibitory')
                if len(e2out) != 0:
                    [plastic, non_plastic] = split_plastic(e2out)
                    if len(plastic) != 0:
                        connect_to_arms(excite[excite_count], plastic, bandit_arms[bandit_count], 'excitatory',
                                            True, stdp_model=stdp_model)
                    if len(non_plastic) != 0:
                        connect_to_arms(excite[excite_count], non_plastic, bandit_arms[bandit_count], 'excitatory',
                                            True, stdp_model=stdp_model)
                if len(i2out) != 0:
                    [plastic, non_plastic] = split_plastic(i2out)
                    if len(plastic) != 0:
                        connect_to_arms(inhib[inhib_count], plastic, bandit_arms[bandit_count], 'inhibitory',
                                            True, stdp_model=stdp_model)
                    if len(non_plastic) != 0:
                        connect_to_arms(inhib[inhib_count], non_plastic, bandit_arms[bandit_count], 'inhibitory',
                                            True, stdp_model=stdp_model)

        print "\nfinished connections seed = ", seed, "\n"
        simulator = get_simulator()
        try:
            print "\nrun seed = ", seed, "\n"
            if len(connections) == len(failures):
                p.end()
                print "nothing to run so ending and returning fail"
                return 'fail'
            p.run(runtime)
            try_except = max_attempts
            break
        except:
            traceback.print_exc()
            try:
                print "\nrun 2 seed = ", seed, "\n"
                globals_variables.unset_simulator()
                print "end was necessary"
            except:
                traceback.print_exc()
                print "end wasn't necessary"
            try_except += 1
            print "failed to run on attempt ", try_except, "\n"  # . total fails: ", all_fails, "\n"
            if try_except >= max_attempts:
                print "calling it a failed population, splitting and rerunning"
                return 'fail'
        print "\nfinished run seed = ", seed, "\n"

    scores = []
    agent_fitness = []
    fails = 0
    excite_spike_count = [0 for i in range(len(connections))]
    excite_fail = 0
    inhib_spike_count = [0 for i in range(len(connections))]
    inhib_fail = 0
    print "reading the spikes of ", config, '\n', seed
    for i in range(len(connections)):
        print "started processing fitness of: ", i, '/', len(connections)
        if i in failures:
            print "worst score for the failure"
            fails += 1
            scores.append([[max_fail_score], [max_fail_score], [max_fail_score], [max_fail_score]])
            # agent_fitness.append(scores[i])
            excite_spike_count[i] -= max_fail_score
            inhib_spike_count[i] -= max_fail_score
        else:
            if spike_f:
                if i in excite_marker:
                    print "counting excite spikes"
                    spikes = excite[i - excite_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            excite_spike_count[i] += 1
                else:
                    excite_fail += 1
                    print "had an excite failure"
                if i in inhib_marker:
                    print "counting inhib spikes"
                    spikes = inhib[i - inhib_fail - fails].get_data('spikes').segments[0].spiketrains
                    for neuron in spikes:
                        for spike in neuron:
                            inhib_spike_count[i] += 1
                else:
                    inhib_fail += 1
                    print "had an inhib failure"
            scores.append(get_scores(game_pop=bandit[i - fails], simulator=simulator))
            # pop[i].stats = {'fitness': scores[i][len(scores[i]) - 1][0]}  # , 'steps': 0}
        print "\nfinished spikes", seed, "\n"
        if spike_f:
            agent_fitness.append([scores[i][len(scores[i]) - 1][0], excite_spike_count[i] + inhib_spike_count[i]])
        else:
            agent_fitness.append(scores[i][len(scores[i]) - 1][0])
        # print i, "| e:", excite_spike_count[i], "-i:", inhib_spike_count[i], "|\t", scores[i]
    print seed, "\nThe scores for this run of {} agents are:".format(len(connections))
    for i in range(len(connections)):
        print "c:{}, s:{}, si:{}, si0:{}".format(len(connections), len(scores), len(scores[i]), len(scores[i][0]))
        e_string = "e: {}".format(excite_spike_count[i])
        i_string = "i: {}".format(inhib_spike_count[i])
        score_string = ""
        for j in range(len(scores[i])):
            score_string += "{:4},".format(scores[i][j][0])
        print "{:3} | {:8} {:8} - ".format(i, e_string, i_string), score_string
    print "before end = ", seed
    p.end()
    print "\nafter end = ", seed, "\n"
    print config
    return agent_fitness

def print_fitnesses(fitnesses):
    with open('fitnesses {}.csv'.format(config), 'w') as file:
        writer = csv.writer(file, delimiter=',', lineterminator='\n')
        for fitness in fitnesses:
            writer.writerow(fitness)
        file.close()
    # with open('done {}.csv'.format(config), 'w') as file:
    #     writer = csv.writer(file, delimiter=',', lineterminator='\n')
    #     writer.writerow('', '')
    #     file.close()



fitnesses = thread_bandit(connections, arms, split, runtime, exposure_time, noise_rate, noise_weight, reward, size_f, spike_f, True)

print_fitnesses(fitnesses)