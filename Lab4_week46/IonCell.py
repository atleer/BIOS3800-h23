"""
Effort to model the cell membrane through molecular dynamics. The goal has been to
develop some intuition for how the membrane (resting and equilibrium) potential is 
set up through molecular dynamics simulation of a semi-permeabel membrane. The resulting
simulations are tuned (dimensionless) in an effort to provide said intuition. Lastly, the
code and its structure is not made for ease of use nor to easily adapt (especially these 
classes). This is, again, because the simulations are tuned to provide a concrete set
of intuitions, rather than making the code easily adaptable. If you want to change parameters,
the easiest way to do so is directly here, in the code.

High-level description of contents:
    Laws: 
        1. Columbs law gives the electrostatic forces
        2. Newtons second law F=ma gives the acceleration from force
        3. (attempt) at friction (drag/viscosity) across membrane - gave up
            -> became exponential decay of speed and force when in and crossing membrane
    Integration scheme:
        Velocity Verlet with periodic square glia, and velocity and accleration
        modulation across membrane.
    Otherwise:
        Potential modelled as inverse distance
        
Extensions to these simulations may consider the following:
    1. Including concentration diffusion, perhaps modelled using Gibbs free energy.
    2. Model the membrane permeability correctly.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt2d
from scipy.ndimage import gaussian_filter

from methods import *
from utils import *


class Template:
    def __init__(self):
        # Free parameters
        self.membrane_width = 0.5
        self.cell_radius = 5
        self.tau = 5e-2
        box_range = [-10, 10]
        self.box_range = box_range
        self.L = box_range[1] - box_range[0]
        # Initialise conditions
        in_cell = lambda x: in_circle(x, radius=self.cell_radius)
        in_membrane = lambda x: in_circle(
            x, radius=self.cell_radius + self.membrane_width
        ) & ~in_cell(x)
        in_glia = lambda x: ~in_cell(x) & ~in_membrane(x)
        # Initialise ions
        self.permeability_fn = lambda r, r_update: permeability_mask(
            r, r_update, in_cell, in_membrane, in_glia
        )
        start_n = 100
        self.rs = [sample_circle(self.cell_radius, (start_n, 2))]
        self.vs = [np.zeros_like(self.rs[-1])]
        self.forces = [np.zeros_like(self.rs[-1])]
        self.names = ["Na+"] * start_n
        self.charges = [np.ones(start_n)]
        self.permeability_decay_outward = [np.ones(start_n)]
        self.permeability_decay_inward = [np.zeros(start_n)]
        # self.add_ions(100,1,1,0,"Na+",sampler=lambda n: sample_circle(self.cell_radius,(n,2)))
        self.add_ions(
            100,
            -1,
            0,
            1,
            "K-",
            sampler=lambda n: sample_circle(self.cell_radius, (n, 2)),
        )

        self.res = 50
        self.space = np.stack(
            np.meshgrid(
                np.linspace(*box_range, self.res), np.linspace(*box_range, self.res)
            ),
            axis=-1,
        )
        self.electrode_inside = np.array([-2, -2])
        self.electrode_outside = np.array([-7, -7])
        self.electrode_inside_area = (
            np.linalg.norm((self.space - self.electrode_inside), axis=-1) < 1.5
        )
        self.electrode_outside_area = (
            np.linalg.norm((self.space - self.electrode_outside), axis=-1) < 1.5
        )

    def __call__(self, t):
        """
        Iterate the system forward to time t according to the laws of physics
        """
        for i in range(len(self.rs), t + 1):
            r_prev, v_prev, a = velocity_verlet(
                self.rs[-1],
                self.vs[-1],
                self.forces[-1],
                self.charges[-1],
                self.tau,
                L=self.L,
                permeability_fn=self.permeability_fn,
                periodic_fn=lambda x: periodic_square(x, self.box_range),
                outward_decay=self.permeability_decay_outward[-1],
                inward_decay=self.permeability_decay_inward[-1],
            )
            self.rs.append(r_prev)
            self.vs.append(v_prev)
            self.forces.append(a)
            self.charges.append(self.charges[-1])
            self.permeability_decay_outward.append(self.permeability_decay_outward[-1])
            self.permeability_decay_inward.append(self.permeability_decay_inward[-1])

            if t > 1000 and t < 1200:
                injection_electrode_fn = lambda n: np.array(
                    [[8, -8]]
                ) + np.random.uniform(size=2)
                self.add_ions(1, 1, 1, 1, "Na+", sampler=injection_electrode_fn)

        return self.rs[t], self.vs[t], self.forces[t]

    def add_ions(
        self, n, charge, permeability_outward, permeability_inward, name, sampler=None
    ):
        """"
        Add more ions to the system with a given charge and membrane permeability
        according to some spatial sampler.
        """
        self.charges[-1] = np.concatenate([self.charges[-1], np.ones(n) * charge])
        self.permeability_decay_outward[-1] = np.concatenate(
            [self.permeability_decay_outward[-1], np.ones(n) * permeability_outward]
        )
        self.permeability_decay_inward[-1] = np.concatenate(
            [self.permeability_decay_inward[-1], np.ones(n) * permeability_inward]
        )
        self.names += [name] * n
        self.rs[-1] = (
            np.concatenate([self.rs[-1], sampler(n)]) if self.rs else sampler(n)
        )
        self.vs[-1] = (
            np.concatenate([self.vs[-1], np.zeros((n, 2))])
            if self.vs
            else np.zeros((n, 2))
        )
        self.forces[-1] = (
            np.concatenate([self.forces[-1], np.zeros((n, 2))])
            if self.forces
            else np.zeros((n, 2))
        )

    def energy(self, t):
        """
        Calculate the total energy of the system at time t, i.e.
        the kinetic energy + the potential energy
        """
        if t + 1 > len(self.rs):
            self(t)
        return total_energy(self.rs[t], self.vs[t])

    def plot(self, timepoint):
        """
        plot the cell, the particle positions and potential at a given timepoint
        """
        r, v, a = self(timepoint)
        fig, axs = plt.subplots(figsize=(12, 6), ncols=2)
        # plot membrane
        plot_circle(radius=self.cell_radius, ax=axs[0], color="green", alpha=0.6)
        plot_circle(
            radius=self.cell_radius + self.membrane_width, ax=axs[0], color="green", alpha=0.6
        )
        names = self.names[: len(r)]
        for name in np.unique(names):
            mask = np.array(names) == name
            marker = name[-1] if name[-1] == "+" else "_"
            axs[0].scatter(*r[mask].T, alpha=1, label=name, marker=marker)
        axs[0].set_xlim(self.box_range)
        axs[0].set_ylim(self.box_range)
        axs[0].axis("off")
        axs[0].set_title("Molecular Dynamics")
        axs[0].scatter(
            *self.electrode_inside, marker="v", s=200, color="purple", label="electrode"
        )
        axs[0].scatter(*self.electrode_outside, marker="v", s=200, color="purple")
        axs[0].legend(loc="upper right")

        ppotential_fn = V_E(r[self.charges[timepoint] > 0], epsilon=1e-4)
        npotential_fn = V_E(r[self.charges[timepoint] < 0], epsilon=1e-4)
        potential = ppotential_fn(self.space.reshape(-1, 2)).reshape(
            self.res, self.res
        ) - npotential_fn(self.space.reshape(-1, 2)).reshape(self.res, self.res)
        axs[1].imshow(medfilt2d(potential), origin="lower")  # MEDIAN FILTERING
        # axs[1].imshow(gaussian_filter(potential,2))#,vmin=0,vmax=0.5) # GAUSSIAN FILTERING
        axs[1].axis("off")
        axs[1].set_title("Potential")
        membrane_potential = np.mean(
            potential[self.electrode_inside_area]
            - potential[self.electrode_outside_area]
        )
        membrane_potential = -membrane_potential # by convention? To follow nernst and goldman potentials
        print("DIMENSIONLESS Membrane Potential =", membrane_potential)

        # return fig, axs
        
        
        
class Scenario1:
    """
    1 ion type. impermeable membrane. many ions inside.
    """
    def __init__(self):
        # Free parameters
        self.membrane_width = 0.5
        self.cell_radius = 5
        self.tau = 5e-2
        box_range = [-10, 10]
        self.box_range = box_range
        self.L = box_range[1] - box_range[0]
        # Initialise conditions
        in_cell = lambda x: in_circle(x, radius=self.cell_radius)
        in_membrane = lambda x: in_circle(
            x, radius=self.cell_radius + self.membrane_width
        ) & ~in_cell(x)
        in_glia = lambda x: ~in_cell(x) & ~in_membrane(x)
        # Initialise ions
        self.permeability_fn = lambda r, r_update: permeability_mask(
            r, r_update, in_cell, in_membrane, in_glia
        )
        start_n = 60
        self.rs = [sample_circle(self.cell_radius, (start_n, 2))]
        self.rs[-1] = np.concatenate([self.rs[-1], np.random.uniform(*box_range, (40, 2))])
        self.vs = [np.zeros_like(self.rs[-1])]
        self.forces = [np.zeros_like(self.rs[-1])]
        self.names = ["K+"] * len(self.rs[-1])
        self.charges = [np.ones(len(self.rs[-1]))]
        self.permeability_decay_outward = [np.ones(len(self.rs[-1]))*0]
        self.permeability_decay_inward = [np.ones(len(self.rs[-1]))*1]

        self.res = 50
        self.space = np.stack(
            np.meshgrid(
                np.linspace(*box_range, self.res), np.linspace(*box_range, self.res)
            ),
            axis=-1,
        )
        self.electrode_inside = np.array([-2, -2])
        self.electrode_outside = np.array([-7, -7])
        self.electrode_inside_area = (
            np.linalg.norm((self.space - self.electrode_inside), axis=-1) < 1.5
        )
        self.electrode_outside_area = (
            np.linalg.norm((self.space - self.electrode_outside), axis=-1) < 1.5
        )

    def __call__(self, t):
        """
        Iterate the system forward to time t according to the laws of physics
        """
        for i in range(len(self.rs), t + 1):
            r_prev, v_prev, a = velocity_verlet(
                self.rs[-1],
                self.vs[-1],
                self.forces[-1],
                self.charges[-1],
                self.tau,
                L=self.L,
                permeability_fn=self.permeability_fn,
                periodic_fn=lambda x: periodic_square(x, self.box_range),
                outward_decay=self.permeability_decay_outward[-1],
                inward_decay=self.permeability_decay_inward[-1],
            )
            self.rs.append(r_prev)
            self.vs.append(v_prev)
            self.forces.append(a)
            self.charges.append(self.charges[-1])
            self.permeability_decay_outward.append(self.permeability_decay_outward[-1])
            self.permeability_decay_inward.append(self.permeability_decay_inward[-1])

        return self.rs[t], self.vs[t], self.forces[t]

    def plot(self, timepoint):
        """
        plot the cell, the particle positions and potential at a given timepoint
        """
        r, v, a = self(timepoint)
        fig, axs = plt.subplots(figsize=(12, 6), ncols=2)
        # plot membrane
        plot_circle(radius=self.cell_radius, ax=axs[0], color="green", alpha=0.6)
        plot_circle(
            radius=self.cell_radius + self.membrane_width, ax=axs[0], color="green", alpha=0.6, label="Membrane"
        )
        names = self.names[: len(r)]
        for name in np.unique(names):
            mask = np.array(names) == name
            marker = name[-1] if name[-1] == "+" else "_"
            axs[0].scatter(*r[mask].T, alpha=1, label=name, marker=marker)
        axs[0].set_xlim(self.box_range)
        axs[0].set_ylim(self.box_range)
        axs[0].axis("off")
        axs[0].set_title("Molecular Dynamics")
        axs[0].scatter(
            *self.electrode_inside, marker="v", s=200, color="purple", label="electrode"
        )
        axs[0].scatter(*self.electrode_outside, marker="v", s=200, color="purple")
        axs[0].legend(loc="upper right")

        ppotential_fn = V_E(r[self.charges[timepoint] > 0], epsilon=1e-4)
        npotential_fn = V_E(r[self.charges[timepoint] < 0], epsilon=1e-4)
        potential = ppotential_fn(self.space.reshape(-1, 2)).reshape(
            self.res, self.res
        ) - npotential_fn(self.space.reshape(-1, 2)).reshape(self.res, self.res)
        axs[1].imshow(medfilt2d(potential), origin="lower")  # MEDIAN FILTERING
        # axs[1].imshow(gaussian_filter(potential,2))#,vmin=0,vmax=0.5) # GAUSSIAN FILTERING
        axs[1].axis("off")
        axs[1].set_title("Potential")
        membrane_potential = np.mean(
            potential[self.electrode_inside_area]
            - potential[self.electrode_outside_area]
        )
        membrane_potential = -membrane_potential # by convention? To follow nernst and goldman potentials
        print("(dimensionlesss) Membrane Potential =", membrane_potential)

        # return fig, axs
  
class Scenario2:
    def __init__(self):
        # Free parameters
        self.membrane_width = 0.5
        self.cell_radius = 5
        self.tau = 5e-2
        box_range = [-10, 10]
        self.box_range = box_range
        self.L = box_range[1] - box_range[0]
        # Initialise conditions
        in_cell = lambda x: in_circle(x, radius=self.cell_radius)
        in_membrane = lambda x: in_circle(
            x, radius=self.cell_radius + self.membrane_width
        ) & ~in_cell(x)
        in_glia = lambda x: ~in_cell(x) & ~in_membrane(x)
        # Initialise ions
        self.permeability_fn = lambda r, r_update: permeability_mask(
            r, r_update, in_cell, in_membrane, in_glia
        )
        start_n = 100
        self.rs = [sample_circle(self.cell_radius, (start_n, 2))]
        self.vs = [np.zeros_like(self.rs[-1])]
        self.forces = [np.zeros_like(self.rs[-1])]
        self.names = ["K+"] * start_n
        self.charges = [np.ones(start_n)]
        self.permeability_decay_outward = [np.ones(start_n)*0]
        self.permeability_decay_inward = [np.ones(start_n)*1]
        # self.add_ions(100,1,1,0,"Na+",sampler=lambda n: sample_circle(self.cell_radius,(n,2)))
        self.add_ions(
            100,
            -1,
            1,
            0,
            "Cl-",
            sampler=lambda n: sample_circle(self.cell_radius, (n, 2)),
        )

        self.res = 50
        self.space = np.stack(
            np.meshgrid(
                np.linspace(*box_range, self.res), np.linspace(*box_range, self.res)
            ),
            axis=-1,
        )
        self.electrode_inside = np.array([-2, -2])
        self.electrode_outside = np.array([-7, -7])
        self.electrode_inside_area = (
            np.linalg.norm((self.space - self.electrode_inside), axis=-1) < 1.5
        )
        self.electrode_outside_area = (
            np.linalg.norm((self.space - self.electrode_outside), axis=-1) < 1.5
        )

    def __call__(self, t):
        """
        Iterate the system forward to time t according to the laws of physics
        """
        for i in range(len(self.rs), t + 1):
            r_prev, v_prev, a = velocity_verlet(
                self.rs[-1],
                self.vs[-1],
                self.forces[-1],
                self.charges[-1],
                self.tau,
                L=self.L,
                permeability_fn=self.permeability_fn,
                periodic_fn=lambda x: periodic_square(x, self.box_range),
                outward_decay=self.permeability_decay_outward[-1],
                inward_decay=self.permeability_decay_inward[-1],
            )
            self.rs.append(r_prev)
            self.vs.append(v_prev)
            self.forces.append(a)
            self.charges.append(self.charges[-1])
            self.permeability_decay_outward.append(self.permeability_decay_outward[-1])
            self.permeability_decay_inward.append(self.permeability_decay_inward[-1])

        return self.rs[t], self.vs[t], self.forces[t]

    def add_ions(
        self, n, charge, permeability_outward, permeability_inward, name, sampler=None
    ):
        """"
        Add more ions to the system with a given charge and membrane permeability
        according to some spatial sampler.
        """
        self.charges[-1] = np.concatenate([self.charges[-1], np.ones(n) * charge])
        self.permeability_decay_outward[-1] = np.concatenate(
            [self.permeability_decay_outward[-1], np.ones(n) * permeability_outward]
        )
        self.permeability_decay_inward[-1] = np.concatenate(
            [self.permeability_decay_inward[-1], np.ones(n) * permeability_inward]
        )
        self.names += [name] * n
        self.rs[-1] = (
            np.concatenate([self.rs[-1], sampler(n)]) if self.rs else sampler(n)
        )
        self.vs[-1] = (
            np.concatenate([self.vs[-1], np.zeros((n, 2))])
            if self.vs
            else np.zeros((n, 2))
        )
        self.forces[-1] = (
            np.concatenate([self.forces[-1], np.zeros((n, 2))])
            if self.forces
            else np.zeros((n, 2))
        )

    def energy(self, t):
        """
        Calculate the total energy of the system at time t, i.e.
        the kinetic energy + the potential energy
        """
        if t + 1 > len(self.rs):
            self(t)
        return total_energy(self.rs[t], self.vs[t])

    def plot(self, timepoint):
        """
        plot the cell, the particle positions and potential at a given timepoint
        """
        r, v, a = self(timepoint)
        fig, axs = plt.subplots(figsize=(12, 6), ncols=2)
        # plot membrane
        plot_circle(radius=self.cell_radius, ax=axs[0], color="green", alpha=0.6)
        plot_circle(
            radius=self.cell_radius + self.membrane_width, ax=axs[0], color="green", alpha=0.6
        )
        names = self.names[: len(r)]
        for name in np.unique(names):
            mask = np.array(names) == name
            marker = name[-1] if name[-1] == "+" else "_"
            axs[0].scatter(*r[mask].T, alpha=1, label=name, marker=marker)
        axs[0].set_xlim(self.box_range)
        axs[0].set_ylim(self.box_range)
        axs[0].axis("off")
        axs[0].set_title("Molecular Dynamics")
        axs[0].scatter(
            *self.electrode_inside, marker="v", s=200, color="purple", label="electrode"
        )
        axs[0].scatter(*self.electrode_outside, marker="v", s=200, color="purple")
        axs[0].legend(loc="upper right")

        ppotential_fn = V_E(r[self.charges[timepoint] > 0], epsilon=1e-4)
        npotential_fn = V_E(r[self.charges[timepoint] < 0], epsilon=1e-4)
        potential = ppotential_fn(self.space.reshape(-1, 2)).reshape(
            self.res, self.res
        ) - npotential_fn(self.space.reshape(-1, 2)).reshape(self.res, self.res)
        axs[1].imshow(medfilt2d(potential), origin="lower")  # MEDIAN FILTERING
        # axs[1].imshow(gaussian_filter(potential,2))#,vmin=0,vmax=0.5) # GAUSSIAN FILTERING
        axs[1].axis("off")
        axs[1].set_title("Potential")
        membrane_potential = np.mean(
            potential[self.electrode_inside_area]
            - potential[self.electrode_outside_area]
        )
        membrane_potential = -membrane_potential # by convention? To follow nernst and goldman potentials
        print("DIMENSIONLESS Membrane Potential =", membrane_potential)

        # return fig, axs
        
        

class Scenario3:
    def __init__(self):
        # Free parameters
        self.membrane_width = 0.5
        self.cell_radius = 5
        self.tau = 5e-2
        box_range = [-10, 10]
        self.box_range = box_range
        self.L = box_range[1] - box_range[0]
        # Initialise conditions
        in_cell = lambda x: in_circle(x, radius=self.cell_radius)
        in_membrane = lambda x: in_circle(
            x, radius=self.cell_radius + self.membrane_width
        ) & ~in_cell(x)
        in_glia = lambda x: ~in_cell(x) & ~in_membrane(x)
        # Initialise ions
        self.permeability_fn = lambda r, r_update: permeability_mask(
            r, r_update, in_cell, in_membrane, in_glia
        )
        start_n = 100
        self.rs = [sample_circle(self.cell_radius, (start_n, 2))]
        self.vs = [np.zeros_like(self.rs[-1])]
        self.forces = [np.zeros_like(self.rs[-1])]
        self.names = ["K+"] * start_n
        self.charges = [np.ones(start_n)]
        self.permeability_decay_outward = [np.ones(start_n)*0]
        self.permeability_decay_inward = [np.ones(start_n)*1]
        self.add_ions(
            100,
            -1,
            1,
            0,
            "Cl-",
            sampler=lambda n: sample_circle(self.cell_radius, (n, 2)),
        )

        self.res = 50
        self.space = np.stack(
            np.meshgrid(
                np.linspace(*box_range, self.res), np.linspace(*box_range, self.res)
            ),
            axis=-1,
        )
        self.electrode_inside = np.array([-2, -2])
        self.electrode_outside = np.array([-7, -7])
        self.electrode_inside_area = (
            np.linalg.norm((self.space - self.electrode_inside), axis=-1) < 1.5
        )
        self.electrode_outside_area = (
            np.linalg.norm((self.space - self.electrode_outside), axis=-1) < 1.5
        )
        
        self.membrane_potential = np.zeros(3001)
        self.membrane_potential[:] = np.nan

    def __call__(self, t):
        """
        Iterate the system forward to time t according to the laws of physics
        """
        for i in range(len(self.rs), t + 1):
            r_prev, v_prev, a = velocity_verlet(
                self.rs[-1],
                self.vs[-1],
                self.forces[-1],
                self.charges[-1],
                self.tau,
                L=self.L,
                permeability_fn=self.permeability_fn,
                periodic_fn=lambda x: periodic_square(x, self.box_range),
                outward_decay=self.permeability_decay_outward[-1],
                inward_decay=self.permeability_decay_inward[-1],
            )
            self.rs.append(r_prev)
            self.vs.append(v_prev)
            self.forces.append(a)
            self.charges.append(self.charges[-1])
            self.permeability_decay_outward.append(self.permeability_decay_outward[-1])
            self.permeability_decay_inward.append(self.permeability_decay_inward[-1])

            if t > 1000 and t < 1050:
                injection_electrode_fn = lambda n: np.array(
                    [[-7, -7]]
                ) + np.random.uniform(size=2)
                self.add_ions(1, 1, 0, 0, "K+", sampler=injection_electrode_fn)

        return self.rs[t], self.vs[t], self.forces[t]

    def add_ions(
        self, n, charge, permeability_outward, permeability_inward, name, sampler=None
    ):
        """"
        Add more ions to the system with a given charge and membrane permeability
        according to some spatial sampler.
        """
        self.charges[-1] = np.concatenate([self.charges[-1], np.ones(n) * charge])
        self.permeability_decay_outward[-1] = np.concatenate(
            [self.permeability_decay_outward[-1], np.ones(n) * permeability_outward]
        )
        self.permeability_decay_inward[-1] = np.concatenate(
            [self.permeability_decay_inward[-1], np.ones(n) * permeability_inward]
        )
        self.names += [name] * n
        self.rs[-1] = (
            np.concatenate([self.rs[-1], sampler(n)]) if self.rs else sampler(n)
        )
        self.vs[-1] = (
            np.concatenate([self.vs[-1], np.zeros((n, 2))])
            if self.vs
            else np.zeros((n, 2))
        )
        self.forces[-1] = (
            np.concatenate([self.forces[-1], np.zeros((n, 2))])
            if self.forces
            else np.zeros((n, 2))
        )

    def energy(self, t):
        """
        Calculate the total energy of the system at time t, i.e.
        the kinetic energy + the potential energy
        """
        if t + 1 > len(self.rs):
            self(t)
        return total_energy(self.rs[t], self.vs[t])

    def plot(self, timepoint):
        """
        plot the cell, the particle positions and potential at a given timepoint
        """
        r, v, a = self(timepoint)
        fig, axs = plt.subplots(figsize=(12, 4), ncols=3)
        # plot membrane
        plot_circle(radius=self.cell_radius, ax=axs[0], color="green", alpha=0.6)
        plot_circle(
            radius=self.cell_radius + self.membrane_width, ax=axs[0], color="green", alpha=0.6
        )
        names = self.names[: len(r)]
        for name in np.unique(names):
            mask = np.array(names) == name
            marker = name[-1] if name[-1] == "+" else "_"
            axs[0].scatter(*r[mask].T, alpha=1, label=name, marker=marker)
        axs[0].set_xlim(self.box_range)
        axs[0].set_ylim(self.box_range)
        axs[0].axis("off")
        axs[0].set_title("Molecular Dynamics")
        axs[0].scatter(
            *self.electrode_inside, marker="v", s=200, color="purple", label="electrode"
        )
        axs[0].scatter(*self.electrode_outside, marker="v", s=200, color="purple")
        axs[0].legend(loc="upper right")

        ppotential_fn = V_E(r[self.charges[timepoint] > 0], epsilon=1e-4)
        npotential_fn = V_E(r[self.charges[timepoint] < 0], epsilon=1e-4)
        potential = ppotential_fn(self.space.reshape(-1, 2)).reshape(
            self.res, self.res
        ) - npotential_fn(self.space.reshape(-1, 2)).reshape(self.res, self.res)
        axs[1].imshow(medfilt2d(potential), origin="lower")  # MEDIAN FILTERING
        # axs[1].imshow(gaussian_filter(potential,2))#,vmin=0,vmax=0.5) # GAUSSIAN FILTERING
        axs[1].axis("off")
        axs[1].set_title("Potential")
        
        membrane_potential = np.mean(
            potential[self.electrode_inside_area]
            - potential[self.electrode_outside_area]
        )
        membrane_potential = -membrane_potential # by convention? To follow nernst and goldman potentials
        print("DIMENSIONLESS Membrane Potential =", membrane_potential)
        self.membrane_potential[timepoint] = membrane_potential
        axs[2].plot(self.membrane_potential, "-o")
        axs[2].set_ylabel("Potential")
        axs[2].set_xlabel("Time")
        # return fig, axs
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


class Scenario_notused:
    """
    A single ion type starting in a small square in the middle of the 
    surrounding box. Eventually the system uniformly distributes the ions.
    """
    def __init__(self):
        # Free parameters
        self.tau = 5e-2
        box_range = [-10, 10]
        self.box_range = box_range
        self.L = box_range[1] - box_range[0]
        start_n = 100
        sampler = lambda n: np.random.uniform(-2,2,size=(n,2))
        self.rs = [sampler(start_n)]
        self.vs = [np.zeros_like(self.rs[-1])]
        self.forces = [np.zeros_like(self.rs[-1])]
        self.charges = [np.ones(start_n)]
        self.names = ["Na+"] * start_n

    def __call__(self, t):
        """
        Iterate the system forward to time t according to the laws of physics
        """
        for i in range(len(self.rs), t + 1):
            r_prev, v_prev, a = velocity_verlet(
                self.rs[-1],
                self.vs[-1],
                self.forces[-1],
                self.charges[-1],
                self.tau,
                L=self.L,
                periodic_fn=lambda x: periodic_square(x, self.box_range),
            )
            self.rs.append(r_prev)
            self.vs.append(v_prev)
            self.forces.append(a)
            self.charges.append(self.charges[-1])

        return self.rs[t], self.vs[t], self.forces[t]

    def plot(self, timepoint):
        """
        plot the cell, the particle positions and potential at a given timepoint
        """
        r, v, a = self(timepoint)
        fig, ax = plt.subplots(figsize=(6, 6), ncols=1)
        names = self.names[: len(r)]
        for name in np.unique(names):
            mask = np.array(names) == name
            marker = name[-1] if name[-1] == "+" else "_"
            ax.scatter(*r[mask].T, alpha=1, label=name, marker=marker)
        ax.set_xlim(self.box_range)
        ax.set_ylim(self.box_range)
        ax.axis("off")
        ax.set_title("Molecular Dynamics")
        ax.legend(loc="upper right")


      
        
class Scenario2_broken:
    def __init__(self):
        # Free parameters
        self.membrane_width = 0.5
        self.cell_radius = 5
        self.tau = 5e-2
        box_range = [-10, 10]
        self.box_range = box_range
        self.L = box_range[1] - box_range[0]
        # Initialise conditions
        in_cell = lambda x: in_circle(x, radius=self.cell_radius)
        in_membrane = lambda x: in_circle(
            x, radius=self.cell_radius + self.membrane_width
        ) & ~in_cell(x)
        in_glia = lambda x: ~in_cell(x) & ~in_membrane(x)
        # Initialise ions
        self.permeability_fn = lambda r, r_update: permeability_mask(
            r, r_update, in_cell, in_membrane, in_glia
        )
        start_n = 100
        self.rs = [sample_circle(self.cell_radius, (start_n, 2))]
        self.vs = [np.zeros_like(self.rs[-1])]
        self.forces = [np.zeros_like(self.rs[-1])]
        self.names = ["K+"] * start_n
        self.charges = [np.ones(start_n)]
        self.permeability_decay_outward = [np.ones(start_n)*0]
        self.permeability_decay_inward = [np.ones(start_n)*1]
        # self.add_ions(100,1,1,0,"Na+",sampler=lambda n: sample_circle(self.cell_radius,(n,2)))
        self.add_ions(
            100,
            -1,
            0,
            0,
            "Cl-",
            sampler=lambda n: sample_circle(self.cell_radius, (n, 2)),
        )

        self.res = 50
        self.space = np.stack(
            np.meshgrid(
                np.linspace(*box_range, self.res), np.linspace(*box_range, self.res)
            ),
            axis=-1,
        )
        self.electrode_inside = np.array([-2, -2])
        self.electrode_outside = np.array([-7, -7])
        self.electrode_inside_area = (
            np.linalg.norm((self.space - self.electrode_inside), axis=-1) < 1.5
        )
        self.electrode_outside_area = (
            np.linalg.norm((self.space - self.electrode_outside), axis=-1) < 1.5
        )

    def __call__(self, t):
        """
        Iterate the system forward to time t according to the laws of physics
        """
        for i in range(len(self.rs), t + 1):
            r_prev, v_prev, a = velocity_verlet(
                self.rs[-1],
                self.vs[-1],
                self.forces[-1],
                self.charges[-1],
                self.tau,
                L=self.L,
                permeability_fn=self.permeability_fn,
                periodic_fn=lambda x: periodic_square(x, self.box_range),
                outward_decay=self.permeability_decay_outward[-1],
                inward_decay=self.permeability_decay_inward[-1],
            )
            self.rs.append(r_prev)
            self.vs.append(v_prev)
            self.forces.append(a)
            self.charges.append(self.charges[-1])
            self.permeability_decay_outward.append(self.permeability_decay_outward[-1])
            self.permeability_decay_inward.append(self.permeability_decay_inward[-1])

        return self.rs[t], self.vs[t], self.forces[t]

    def add_ions(
        self, n, charge, permeability_outward, permeability_inward, name, sampler=None
    ):
        """"
        Add more ions to the system with a given charge and membrane permeability
        according to some spatial sampler.
        """
        self.charges[-1] = np.concatenate([self.charges[-1], np.ones(n) * charge])
        self.permeability_decay_outward[-1] = np.concatenate(
            [self.permeability_decay_outward[-1], np.ones(n) * permeability_outward]
        )
        self.permeability_decay_inward[-1] = np.concatenate(
            [self.permeability_decay_inward[-1], np.ones(n) * permeability_inward]
        )
        self.names += [name] * n
        self.rs[-1] = (
            np.concatenate([self.rs[-1], sampler(n)]) if self.rs else sampler(n)
        )
        self.vs[-1] = (
            np.concatenate([self.vs[-1], np.zeros((n, 2))])
            if self.vs
            else np.zeros((n, 2))
        )
        self.forces[-1] = (
            np.concatenate([self.forces[-1], np.zeros((n, 2))])
            if self.forces
            else np.zeros((n, 2))
        )

    def energy(self, t):
        """
        Calculate the total energy of the system at time t, i.e.
        the kinetic energy + the potential energy
        """
        if t + 1 > len(self.rs):
            self(t)
        return total_energy(self.rs[t], self.vs[t])

    def plot(self, timepoint):
        """
        plot the cell, the particle positions and potential at a given timepoint
        """
        r, v, a = self(timepoint)
        fig, axs = plt.subplots(figsize=(12, 6), ncols=2)
        # plot membrane
        plot_circle(radius=self.cell_radius, ax=axs[0], color="green", alpha=0.6)
        plot_circle(
            radius=self.cell_radius + self.membrane_width, ax=axs[0], color="green", alpha=0.6
        )
        names = self.names[: len(r)]
        for name in np.unique(names):
            mask = np.array(names) == name
            marker = name[-1] if name[-1] == "+" else "_"
            axs[0].scatter(*r[mask].T, alpha=1, label=name, marker=marker)
        axs[0].set_xlim(self.box_range)
        axs[0].set_ylim(self.box_range)
        axs[0].axis("off")
        axs[0].set_title("Molecular Dynamics")
        axs[0].scatter(
            *self.electrode_inside, marker="v", s=200, color="purple", label="electrode"
        )
        axs[0].scatter(*self.electrode_outside, marker="v", s=200, color="purple")
        axs[0].legend(loc="upper right")

        ppotential_fn = V_E(r[self.charges[timepoint] > 0], epsilon=1e-4)
        npotential_fn = V_E(r[self.charges[timepoint] < 0], epsilon=1e-4)
        potential = ppotential_fn(self.space.reshape(-1, 2)).reshape(
            self.res, self.res
        ) - npotential_fn(self.space.reshape(-1, 2)).reshape(self.res, self.res)
        axs[1].imshow(medfilt2d(potential), origin="lower")  # MEDIAN FILTERING
        # axs[1].imshow(gaussian_filter(potential,2))#,vmin=0,vmax=0.5) # GAUSSIAN FILTERING
        axs[1].axis("off")
        axs[1].set_title("Potential")
        membrane_potential = np.mean(
            potential[self.electrode_inside_area]
            - potential[self.electrode_outside_area]
        )
        membrane_potential = -membrane_potential # by convention? To follow nernst and goldman potentials
        print("DIMENSIONLESS Membrane Potential =", membrane_potential)

        # return fig, axs
    
