# main_simulation_script.py

import numpy as np
import matplotlib.pyplot as plt
from IFneuron import IFneuron 

def run_neuron_network_simulation():
    neuron_A = IFneuron(id="Neuron_A")
    neuron_B = IFneuron(id="Neuron_B")
    neuron_C = IFneuron(id="Neuron_C")
    neuron_D = IFneuron(id="Neuron_D")
    neuron_E = IFneuron(id="Neuron_E")

    neuron_A.FIFO = None
    neuron_B.FIFO = None
    neuron_C.FIFO = None
    neuron_D.FIFO = None
    neuron_E.FIFO = None

    print(f"Created neurons: {neuron_A.id}, {neuron_B.id}, {neuron_C.id}, {neuron_D.id}", {neuron_E.id})


    # Neuron A and B connects to C
    neuron_C.receptors.append((neuron_A, -0.1)) # Neuron C receives input from Neuron A
    neuron_C.receptors.append((neuron_B, -0.5)) # Neuron C receives input from Neuron B
    print(f"{neuron_C.id} is set to receive input from {neuron_A.id} and {neuron_B.id}")

    # Neurons A and B connect to D
    neuron_D.receptors.append((neuron_A, 0.4)) # Neuron D receives input from Neuron A
    neuron_D.receptors.append((neuron_B, 0.5)) # Neuron D receives input from Neuron B
    print(f"{neuron_D.id} is set to receive input from {neuron_A.id} and {neuron_B.id}")

    # Neurons C and D connect to E
    neuron_E.receptors.append((neuron_C, 1.5)) # Neuron E receives input from Neuron C
    neuron_E.receptors.append((neuron_D, 1.0)) # Neuron E receives input from Neuron D
    print(f"{neuron_E.id} is set to receive input from {neuron_C.id} and {neuron_D.id}")


    # Stimulate Neuron A directly at specific times
    neuron_A.attach_direct_stim(t_ms=10)
    neuron_A.attach_direct_stim(t_ms=50)
    neuron_A.attach_direct_stim(t_ms=90)
    print(f"Direct stimulation times set for {neuron_A.id}: {neuron_A.t_directstim_ms}")

    # Stimulate Neuron B directly at specific times (optional, added for more activity)
    neuron_B.attach_direct_stim(t_ms=11)
    neuron_B.attach_direct_stim(t_ms=60)
    print(f"Direct stimulation times set for {neuron_B.id}: {neuron_B.t_directstim_ms}")

    # set spontaneous activity for one neuron
    # neuron_C.set_spontaneous_activity(mean_stdev=(200, 20))
    # print(f"Spontaneous activity set for {neuron_C.id}")


    simulation_duration_ms = 200
    dt_ms = 1
    all_neurons = [neuron_A, neuron_B, neuron_C, neuron_D, neuron_E] 


    print("\nStarting simulation...")
    for t in np.arange(0, simulation_duration_ms + dt_ms, dt_ms):
        for neuron in all_neurons:
            neuron.update(t_ms=t, recording=True)
    print("Simulation finished.")

    print("\nSimulation Results:")

    fig, axs = plt.subplots(len(all_neurons), 1, figsize=(12, 2 * len(all_neurons)), sharex=True)
    fig.suptitle('Neuron Membrane Potentials and Spike Times')

    for i, neuron in enumerate(all_neurons):
        python_int_spike_times = [int(t) for t in neuron.t_act_ms]
        print(f"\n{neuron.id} Spike Times (ms): {python_int_spike_times}")

        if neuron.t_recorded_ms and neuron.Vm_recorded:
            axs[i].plot(neuron.t_recorded_ms, neuron.Vm_recorded, label=f'{neuron.id} Vm (mV)')
            axs[i].axhline(y=neuron.Vact_mV, color='r', linestyle='--', label='Threshold')
            if neuron.t_act_ms:
                spike_y = [neuron.Vact_mV + 5] * len(neuron.t_act_ms) 
                axs[i].plot(neuron.t_act_ms, spike_y, 'o', color='black', markersize=4, label='Spike')

            axs[i].set_ylabel('Vm (mV)')
            axs[i].set_title(f'{neuron.id}')
            axs[i].legend(loc='upper right')
            axs[i].grid(True)
        else:
            axs[i].text(0.5, 0.5, f'No recordings for {neuron.id}', horizontalalignment='center', verticalalignment='center', transform=axs[i].transAxes)

    axs[-1].set_xlabel('Time (ms)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()


if __name__ == "__main__":
    run_neuron_network_simulation()