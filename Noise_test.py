import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from bandit.spinn_bandit.python_models.bandit import Bandit
import numpy as np

def test_levels(rates=(1, 0.2, 20), weights=(4.7, 4.75, 4.8, 22.5, 23, 23.5), pop_size=1):
    counter = 0
    receive_pop = []
    spike_input = []
    p.setup(timestep=1, min_delay=1, max_delay=127)
    thing = p.IF_curr_exp
    p.set_number_of_neurons_per_core(thing, 10)
    for rate in rates:
        for weight in weights:
            receive_pop.append(p.Population(pop_size, thing()))#, label="receive_pop{}-{}".format(rate, weight)))

            receive_pop[counter].record(['spikes', 'v'])#["spikes"])

            # Connect key spike injector to input population
            spike_input.append(p.Population(pop_size, p.SpikeSourcePoisson(rate=rate), label="input_connect{}-{}".format(rate, weight)))
            p.Projection(
                spike_input[counter], receive_pop[counter], p.OneToOneConnector(), p.StaticSynapse(weight=weight))

            print "reached here 1"
            runtime = 11000

            counter += 1

    p.run(runtime)
    print "reached here 2"

    for i in range(counter):
        weight_index = i % len(weights)
        param_index = (i - weight_index) / len(weights)
        print weight_index
        print param_index
        # for j in range(receive_pop_size):
        spikes = receive_pop[i].get_data('spikes').segments[0].spiketrains
        for neuron in spikes:
            for spike_time in neuron:
                a = float(spike_time)
                print spike_time
                print a
        v = receive_pop[i].get_data('v').segments[0].filter(name='v')[0]
        plt.figure("alpha rate = {} - weight = {} {}".format(rates[param_index], weights[weight_index], thing))
        Figure(
            Panel(spikes, xlabel="Time (ms)", ylabel="nID", xticks=True),
            Panel(v, ylabel="Membrane potential (mV)", yticks=True)
        )
        plt.show()

    # End simulation
    p.end()

def test_params(params=(0.1, 0.2, 0.3, 0.4), weights=(2, 5), pop_size=1):
    counter = 0
    receive_pop = []
    spike_input = []
    p.setup(timestep=1, min_delay=1, max_delay=127)
    thing = p.extra_models.IFCurrExpCa2Adaptive
    p.set_number_of_neurons_per_core(thing, 10)
    rate = 2
    for param in params:
        for weight in weights:
            # thing = p.IF_curr_alpha
            receive_pop.append(p.Population(pop_size, thing(i_alpha=param+0.5)))#, label="receive_pop{}-{}".format(rate, weight)))

            receive_pop[counter].record(['spikes', 'v'])#["spikes"])

            # Connect key spike injector to input population
            spike_input.append(p.Population(pop_size, p.SpikeSourcePoisson(rate=rate), label="input_connect{}-{}".format(param, weight)))
            spike_input[counter].record('spikes')
            p.Projection(
                spike_input[counter], receive_pop[counter], p.OneToOneConnector(), p.StaticSynapse(weight=weight))

            print "reached here 1"
            runtime = 11000

            counter += 1

    p.run(runtime)
    print "reached here 2"

    # for i in range(counter):
    #     weight_index = i % len(weights)
    #     param_index = (i - weight_index) / len(weights)
    #     print weight_index
    #     print param_index
    #     # for j in range(receive_pop_size):
    #     spikes = receive_pop[i].get_data('spikes').segments[0].spiketrains
    #     for neuron in spikes:
    #         for spike_time in neuron:
    #             a = float(spike_time)
    #             print spike_time
    #             print a
    #     v = receive_pop[i].get_data('v').segments[0].filter(name='v')[0]
    #     plt.figure("param = {} - weight = {} {}".format(params[param_index], weights[weight_index], thing))
    #     Figure(
    #         Panel(spikes, xlabel="Time (ms)", ylabel="nID", xticks=True),
    #         Panel(v, ylabel="Membrane potential (mV)", yticks=True)
    #     )
    #     plt.show()
    v = []
    spikes = []
    in_spikes = []
    for i in range(counter):
        weight_index = i % len(weights)
        param_index = (i - weight_index) / len(weights)
        # for j in range(receive_pop_size):
        in_spikes.append(spike_input[i].get_data('spikes').segments[0].spiketrains)
        spikes.append(receive_pop[i].get_data('spikes').segments[0].spiketrains)
        v.append(receive_pop[i].get_data('v').segments[0].filter(name='v')[0])
        plt.figure("rate = {} - param = {} - weight = {} {}".format(rate, params, weights, thing))
    Figure(
        Panel(in_spikes[0], xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(spikes[0], xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(v[0], ylabel="Membrane potential (mV)", yticks=True),
        Panel(in_spikes[1], xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(spikes[1], xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(v[1], ylabel="Membrane potential (mV)", yticks=True),
        Panel(in_spikes[2], xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(spikes[2], xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(v[2], ylabel="Membrane potential (mV)", yticks=True),
        Panel(in_spikes[3], xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(spikes[3], xlabel="Time (ms)", ylabel="nID", xticks=True),
        Panel(v[3], ylabel="Membrane potential (mV)", yticks=True)#,
        # Panel(in_spikes[4], xlabel="Time (ms)", ylabel="nID", xticks=True),
        # Panel(spikes[4], xlabel="Time (ms)", ylabel="nID", xticks=True),
        # Panel(v[4], ylabel="Membrane potential (mV)", yticks=True),
        # Panel(in_spikes[5], xlabel="Time (ms)", ylabel="nID", xticks=True),
        # Panel(spikes[5], xlabel="Time (ms)", ylabel="nID", xticks=True),
        # Panel(v[5], ylabel="Membrane potential (mV)", yticks=True),
        # Panel(in_spikes[6], xlabel="Time (ms)", ylabel="nID", xticks=True),
        # Panel(spikes[6], xlabel="Time (ms)", ylabel="nID", xticks=True),
        # Panel(v[6], ylabel="Membrane potential (mV)", yticks=True),
        # Panel(in_spikes[7], xlabel="Time (ms)", ylabel="nID", xticks=True),
        # Panel(spikes[7], xlabel="Time (ms)", ylabel="nID", xticks=True),
        # Panel(v[7], ylabel="Membrane potential (mV)", yticks=True)
    )
    plt.show()

    # End simulation
    p.end()

def test_packets(rate=100, weight=0.01, probability=0.7, seed=27, pop_size=2, count=200, with_bandit=False):
    counter = 0
    receive_pop = []
    output_pop = []
    spike_input = []
    bandit_pops = []
    p.setup(timestep=1, min_delay=1, max_delay=127)
    p.set_number_of_neurons_per_core(p.IF_cond_exp, 10)
    for i in range(count):
        # pop_size = 2

        receive_pop.append(p.Population(pop_size, p.IF_cond_exp(), label="receive_pop{}".format(i)))
        output_pop.append(p.Population(2, p.IF_cond_exp(), label="output_pop{}".format(i)))
        p.Projection(receive_pop[counter], output_pop[counter], p.AllToAllConnector(), p.StaticSynapse(weight=0.1))
        np.random.seed(seed)
        p.Projection(receive_pop[counter], receive_pop[counter], p.FixedProbabilityConnector(probability), p.StaticSynapse(weight=0.1))
        p.Projection(output_pop[counter], output_pop[counter], p.FixedProbabilityConnector(probability), p.StaticSynapse(weight=0.1))

        if with_bandit:
            random_seed = []
            for j in range(4):
                random_seed.append(np.random.randint(0xffff))
            band = Bandit([0.9, 0.1], reward_delay=200, rand_seed=random_seed, label="bandit {}".format(i))
            bandit_pops.append(p.Population(band.neurons(), band, label="bandit {}".format(i)))
            p.Projection(bandit_pops[counter], receive_pop[counter], p.OneToOneConnector(), p.StaticSynapse(weight=0.1))
            p.Projection(output_pop[counter], bandit_pops[counter], p.OneToOneConnector(), p.StaticSynapse(weight=weight))

        receive_pop[counter].record(['spikes'])#, 'v'])#["spikes"])

        # Connect key spike injector to input population
        spike_input.append(p.Population(pop_size, p.SpikeSourcePoisson(rate=rate), label="input_connect{}-{}".format(rate, weight)))
        p.Projection(
            spike_input[counter], receive_pop[counter], p.OneToOneConnector(), p.StaticSynapse(weight=weight))
        # spike_output.append(p.Population(pop_size, p.SpikeSourcePoisson(rate=rate), label="input_connect{}-{}".format(rate, weight)))
        p.Projection(
            spike_input[counter], receive_pop[counter], p.OneToOneConnector(), p.StaticSynapse(weight=weight))

        runtime = 21000

        counter += 1

    p.run(runtime)

    spikes = []
    for i in range(counter):
        spikes.append(receive_pop[i].get_data('spikes').segments[0].spiketrains)

    # End simulation
    p.end()
    print "ended"

def from_list_test(list_size):
    # number_of_chips = 1000
    p.setup(timestep=1, min_delay=1, max_delay=127)#, n_chips_required=number_of_chips)
    thing = IFCurrDeltaGrazAdaptive
    # p.set_number_of_neurons_per_core(thing, 15)
    from_list = []
    for i in range(list_size):
        from_list.append((0, 0, 0.01, 12))

    receive_pop = p.Population(1, thing())  # , label="receive_pop{}-{}".format(rate, weight)))
    receive_pop.record('spikes')
    # machine = p.get_machine()
    # for i, chip in enumerate(machine.ethernet_connected_chips):
    #     print "i:", i, "- chip:", chip.x, "/", chip.y
    # if not machine.is_chip_at(0, 0):
    #     receive_pop.add_placement_constraint(x=1, y=1)
    # else:
    #     receive_pop.add_placement_constraint(x=0, y=0)

    # Connect key spike injector to input population
    spike_input = p.Population(100, p.SpikeSourcePoisson(rate=4),
                               label="input_connect")
    # p.Projection(spike_input, receive_pop, p.FromListConnector(from_list))
    p.Projection(spike_input, receive_pop, p.FromListConnector(from_list))
    # p.Projection(spike_input, receive_pop, p.AllToAllConnector())
    # p.Projection(spike_input, receive_pop, p.AllToAllConnector())

    print "reached here 1"
    runtime = 11000

    p.run(runtime)

    spikes = receive_pop.get_data('spikes').segments[0].spiketrains

    Figure(
        Panel(spikes, xlabel="Time (ms)", ylabel="nID", xticks=True)
    )

    print "done"

# from_list_test(1)
# test_levels()
test_params()
# for prob in np.linspace(0.2,1,100):
#     seed = np.random.randint(0,1000)
#     print "seed:", seed, "prob:", prob
#     test_packets(probability=prob, seed=seed, with_bandit=True)

print "done all"