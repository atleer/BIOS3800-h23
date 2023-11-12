import matplotlib.pyplot as plt
import numpy as np


def add_colorbars(fig, axs, imgs):
    # add colorbars to all inputted imgs
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    for ax, img in zip(axs, imgs):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)


def plot_circle(radius, ax=None, res=100, **kwargs):
    if ax == None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.plot(
        radius * np.cos(np.linspace(-np.pi, np.pi, res)),
        radius * np.sin(np.linspace(-np.pi, np.pi, res)),
        **kwargs
    )
    return fig, ax


def plot_ions(negatives, positives, fig=None, ax=None):
    if fig == None or ax == None:
        fig, ax = plt.subplots()

    ax.scatter(*negatives.T, label="Negative Ions", marker="_")
    ax.scatter(*positives.T, label="Positive Ions", marker="+")

    # cell membrane
    ax.plot(
        0.5 * np.cos(np.linspace(-np.pi, np.pi, 100)),
        0.5 * np.sin(np.linspace(-np.pi, np.pi, 100)),
        linewidth=2.5,
        color="green",
        label="Cell Membrane",
    )

    # electrodes
    ax.scatter(0.25, 0.25, marker="v", s=200, color="purple")
    ax.scatter(0.8, 0.25, marker="v", s=200, color="purple", label="Electrodes")

    ax.legend()
    ax.axis("off")
    return fig, ax
