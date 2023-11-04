

from pylab import * # general plotting and computing
from matplotlib.collections import PolyCollection # extra fancy plotting
from ipywidgets import widgets, fixed # sliders
from IPython.display import display, clear_output
import LFPy # neural simulation
from neuron import h # neural simulation

# colors for plotting of electrodes
el_clrs = [matplotlib.cm.get_cmap('Set3')(i) for i in np.linspace(0,2,num=10)]

def plot_morph(ax, cell, input_locs=[0], rec_locs=[1000], color='k', title = 'Morphology'):
    input_idcs = [cell.get_closest_idx(x=xloc, y=0, z=0) for xloc in input_locs]
    rec_idcs = [cell.get_closest_idx(x=xloc, y=0, z=0) for xloc in rec_locs]
    zips = []
    for x, z in cell.get_idx_polygons():
        zips.append(list(zip(x, z)))
    polycol = PolyCollection(zips,
                             edgecolors=color,
                             facecolors=color,
                             linewidths = cell.d)

    ax.add_collection(polycol)

    for el_num, idx in enumerate(input_idcs):
        ax.plot(cell.x.mean(axis=-1)[idx], cell.z.mean(axis=-1)[idx]+1, 'v', color=el_clrs[el_num], ms=15, \
                label = 'Input electrode '+str(el_num+1))

    for el_num, idx in enumerate(rec_idcs):
        ax.plot(cell.x.mean(axis=-1)[idx], cell.z.mean(axis=-1)[idx]-1, '^', color='r', ms=15, \
                label = 'Recording electrode')#+str(el_num+1))
    ax.set_ylim([-10, 12])

    #ax.spines['left'].set_visible(False)
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel(r'x ($\mu$m)')
    ax.set_title(title)
    ax.set_xlim([-100,4100])
    return ax


def plot_electrode_current(ax, cell, stim_electrodes):
    for el_num, electrode in enumerate(stim_electrodes):
        ax.plot(cell.tvec, electrode.i, color=el_clrs[el_num], ls='--', label='electrode ' + str(el_num+1))
    ax.set_xlabel('t (ms)')
    ax.set_ylabel(r'I(t) (nA)')
    ax.set_title('Injected current (t)')
    ax.set_xticks(np.linspace(0, 100, num=11))

def plot_membrane_potential_t(ax_v, ax_morph, cell, xloc, label = None):
    idx = cell.get_closest_idx(x=xloc, y=0, z=0)
    #ax_morph.plot(cell.x.mean(axis=-1)[idx], cell.z.mean(axis=-1)[idx]-1, '^', color='r', ms=15, label = 'Recording electrode')
    ax_v.plot(cell.tvec, cell.vmem[idx,:], label = label)
    ax_v.set_xlabel('t (ms)')
    ax_v.set_ylabel(r'$V$(t) (mV)')
    ax_v.set_title('potential (t)')
    ax_v.set_ylim([-100, 50])
    ax_v.set_yticks(np.arange(-100, 50, 20))
    ax_v.set_xlim([0,100])

def plot_membrane_potential_x(ax, cell, t_point, delay=20, label = None):
    '''plot potential as function of x'''
    t = int(t_point/cell.dt) # convert from time in (ms) to timepoint
    ax.plot(cell.x.mean(axis=-1), cell.vmem[:,t], label = label)
    ax.set_ylabel('V(x) (mV)')
    ax.set_xlabel(r'x ($\mu$m)')
    ax.set_title('Membrane potential across length at\ntime '+str(t_point-delay)+' ms after stimulation')
    ax.set_ylim([-100, 50])
    ax.set_yticks(np.arange(-100, 50, 20))
    ax.set_xlim([0,4000])

def plotting_cosmetics(fig):
    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout(h_pad=-2)
    #fig.set_size_inches(10., 10.)

def custom_fun(self, **custom_fun_args):
    secnames = ''.join([sec.name() for sec in h.allsec()])
    if 'axon' in secnames:
        h.axon[0].insert('hh')
        h.axon[1].insert('hh')
    if 'soma' in secnames:
        h.soma[0].insert('hh')


def simulate_cell(morphology, input_locs = [0], delays=[20], active=False, i_amp=1, i_dur = 5, temperature = 6.3):
    num_electrodes = len(input_locs)
    if len(delays) == 0:
        delays = 20*np.ones(len(input_locs))
    if len(delays) != num_electrodes:
        raise AssertionError('You need the same number of input locations and delays! Try again.')

    # set synapse parameters
    r_m = 30000. # membrane resistance
    cell_parameters = {
        'morphology' : morphology,
        'cm' : 1.0,         # membrane capacitance
        'Ra' : 150.,        # axial resistance
        'v_init' : -70.,    # initial membrane potential
        'passive' : True,   # turn on NEURONs passive mechanism for all sections
        'passive_parameters' : {'g_pas' : 1./r_m, 'e_pas' : -70}, # e_pas = resting potential
        'nsegs_method': 'lambda_f',
        'dt' : 2.**-3,      # simulation time step size
        'tstart' : -100.,      # start time of simulation, recorders start at t=0
        'tstop' : 100.,     # stop simulation at 100 ms.
        'custom_fun': None,
        'celsius': temperature,
    }

    if active:
        cell_parameters['custom_fun'] = [custom_fun]
        cell_parameters['custom_fun_args'] = [{}]

    # create cell object
    cell = LFPy.Cell(**cell_parameters)
    cell.set_pos(x=-1000+cell.x[0,0])


    if morphology == './cells/ball_n_stick.hoc':
        cell.set_pos(x = 500)
        cell.set_rotation(z = pi)
    # create electrode for stimulating cell
    pointprocess = {
        'record_current' : True,
        'pptype' : 'IClamp',
        'amp' : i_amp,
        'dur' : i_dur,
    }

    stim_electrodes = []
    for i in range(len(input_locs)):
        pointprocess['idx'] = cell.get_closest_idx(x=input_locs[i], y=0, z=0)
        pointprocess['delay'] = delays[i]
        stim_electrodes.append(LFPy.StimIntElectrode(cell, **pointprocess))

    # simulate cell
    cell.simulate(rec_imem=True, rec_vmem=True)

    return cell, stim_electrodes

def run_sim_and_plot(morphology, input_locs, active, timepoint, delays = []):
    cell, electrode = simulate_cell(morphology, input_locs, delays = delays, active=active)

    fig = figure()
    ax_morph = subplot2grid((2,1),(0,0))
    ax_v_x = subplot2grid((2,1),(1,0))

    # plot morph
    idcs = [cell.get_closest_idx(x=x_loc, y=0, z=0) for x_loc in input_locs]
    plot_morph(ax_morph, cell, input_idcs = idcs)
    # plot V(x)
    plot_membrane_potential_x(ax_v_x, cell, timepoint)
    plotting_cosmetics(fig)