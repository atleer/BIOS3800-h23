import numpy as np


def rejection_sampling(sampler, condition, size):
    """
    Parameters:
        sampler: function that can take in size argument
        condition: function to check whether condition is met
        size: int or (nsamples,) or (nsamples,nfeatures)
    """
    if type(size) == int:
        size = np.ones((1, 1))
    elif len(size) == 1:
        size = np.array(size)[:, None]

    nsamples = np.prod(size[:-1])
    samples = sampler(size)
    samples = samples[condition(samples)]
    while len(samples) < size[0]:
        new_samples = sampler((size[0] - samples.shape[0], size[-1]))
        samples = np.append(samples, new_samples[condition(new_samples)], axis=0)
    return samples


def in_square(x, box_range=[-1, 1]):
    """
    check if some positions x are in a square inferred by box_range
    """
    element_wise = (x <= 1) & (x >= -1)
    total = element_wise.all(axis=-1)  # along last (feature/vector) dimension
    return total


def in_circle(x, radius=1):
    """
    check wether elements (with n-dims in last tensor-axes)
    of a tensor is within or outside of a circle with radius 'radius'.
    assumes center at the origin
    """
    return np.linalg.norm(x, axis=-1) < radius


def sample_circle(radius, size):
    """
    Parameters:
        radius float
        size (nsamples,ndims)
    """
    sampler = lambda size: np.random.uniform(-radius, radius, size=size)
    condition = lambda x: in_circle(x, radius=radius)
    return rejection_sampling(sampler, condition, size=size)


def sample_annulus(r1, r2, nsamples):
    """
    rejection sample annulus based on a square encapsulating the annulus
    obs! this is slow/inefficient for wide and thin annuli
    """
    if r1 > r2:
        # make r2 always the bigger radius
        # i.e. user-independence on which is bigger of r1 and r2
        tmp = r2
        r2 = r1
        r1 = tmp
    sampler = lambda size: np.random.uniform(-r2, r2, size)
    condition = lambda x: in_circle(x, radius=r2) & ~in_circle(x, radius=r1)
    samples = rejection_sampling(sampler, condition, size=(nsamples, 2))
    return samples


def periodic_square(pos, box_range=[-1, 1]):
    """
    Make positions periodic with respect to a square
    inferred by box_range.
    """
    pos = pos - box_range[0]
    return (pos % (box_range[1] - box_range[0])) + box_range[0]


def permeability_mask(r, r_update, in_cell_fn, in_membrane_fn, in_glia_fn):
    """
    Check if current or next position are in membrane, or if current
    and next positions are on opposite sides of the membrane
    """
    r_next = r + r_update
    # if current or next position are in membrane
    mask = in_membrane_fn(r) | in_membrane_fn(r_next)
    # if current and next position are on opposite sides of the membrane
    mask = mask | (in_cell_fn(r) & in_glia_fn(r_next))
    mask = mask | (in_cell_fn(r_next) & in_glia_fn(r))
    return mask


def outward(r, r_update):
    """
    if next position is further from origin that current position,
    then we are pointing outward, otherwise inward.
    """
    return np.linalg.norm(r, axis=-1) < np.linalg.norm(r + r_update, axis=-1)


def V_E(x, scale=1, epsilon=1e-6):
    """
    Electric potential (up to scale) using a KDE-like approach

    Parameters:
        x (nparticles,ndims): positions of particles
        scale int: scaling the "kde" - this could method could be correct by choosing
                   this scaling correctly - i.e. including the permittivity of the space.
    """

    def evaluate(x_i):
        """
        Parameters:
            x_i (npositions,ndims): a 1D mesh (could be ravelled from a 2D mesh) of positions
        """
        return np.sum(
            scale / (np.linalg.norm(x[None] - x_i[..., None, :], axis=-1) + epsilon),
            axis=-1,
        )

    return evaluate


def coloumbs_law(particle_positions, charges, L=None):
    """
    Do silly physics

    Parameters:
        particle_positions (nsamples,2): particle positions in space
        charges (nsamples): ints describing each particle ionic-charge
        L float: length of box
    """
    force = particle_positions[:, None] - particle_positions[None]
    force -= (
        np.round(particle_positions / L) * L if L else 0
    )  # lur Even - pac-man geodesic
    dists = np.linalg.norm(force, axis=-1)
    dists += np.eye(dists.shape[0])  # make zero-diagonal ones to avoid zero-divsion
    force_magnitude = 1 / dists ** 2
    force_magnitude -= np.eye(
        dists.shape[0]
    )  # subtract ones to set diagonal zero again
    unit_force = (
        force / dists[..., None]
    )  # force direction before applying charge direction

    force_direction = charges[:, None] * charges[None]
    force_direction = (
        unit_force * force_direction[..., None]
    )  # the actual force direction

    forces = force_direction * force_magnitude[..., None]
    # use mean to more easily scale the time constant tau,
    # because it becomes independent of the number of cells.
    # because: more cells => more force => should use smaller integration constant
    # return np.sum(forces, axis=1)
    return np.mean(forces, axis=1)


def velocity_verlet(
    r_prev,
    v_prev,
    a,
    charges,
    tau=5e-2,
    L=None,
    permeability_fn=None,
    periodic_fn=None,
    outward_decay=1,
    inward_decay=1,
):
    """
    Calculates the next position and velocity from the previous
    position, velocity, acceleration and current acceleration using
    the velocity verlet integration scheme.

    Parameters:
        r_prev (nparticles,ndims): previous particle positions
        v_prev (nparticles,ndims): previous particle velocities
        a (nparticles,ndims): current particle accelerations
        charges (nparticles,): charge of particles
        tau int: integration constant
    """
    r_update = v_prev * tau + 0.5 * a * tau ** 2
    r = r_prev + r_update

    if periodic_fn:
        # pac-man
        r = periodic_fn(r)

    electrostatic_forces = coloumbs_law(r, charges, L)
    electrostatic_forces = np.clip(electrostatic_forces, -L / tau, L / tau)
    forces = electrostatic_forces  # total forces
    a_next = forces  # F = ma with m=1.

    v = v_prev + 0.5 * (a_next + a) * tau

    if permeability_fn:
        # exponential decay of velocity and acceleration while crossing membrane
        permeability_mask = permeability_fn(r, r_update)
        outward_mask = outward(r, r_update)
        opm = permeability_mask * outward_mask
        ipm = permeability_mask * ~outward_mask
        v[opm] = outward_decay[opm, None] * v[opm]
        v[ipm] = inward_decay[ipm, None] * v[ipm]
        a_next[opm] = outward_decay[opm, None] * a_next[opm]
        a_next[ipm] = inward_decay[ipm, None] * a_next[ipm]

    return r, v, a_next


def kinetic_energy(v):
    """
    Calculate the kinetic energy (cleverly - ty Even)!
    """
    return 0.5 * np.sum(v ** 2)


def total_energy(r, v):
    # potential energy to and from ions
    potential_energy = np.sum(
        1 / (np.linalg.norm(r[None] - r[:, None], axis=-1) + 1e-6)
    )
    return kinetic_energy(v) + potential_energy


def _truncate_magnitude(vector, truncate_value=0.1):
    """
    Truncates the magnitude of a vector

    Parameters:
        vector (nsamples,ndims): vector to be magnitude truncated
        truncate_value float: value to truncate to
    """
    magnitude = np.linalg.norm(vector, axis=-1)
    mask = magnitude > truncate_value
    vector[mask] = (vector[mask] / magnitude[mask, None]) * truncate_value
    return vector
