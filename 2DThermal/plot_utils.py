import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri
import colorcet as cc

plt.rcParams["text.usetex"] = True


def contour_mpl(xyz_nodeCoords, z, fname="contour.jpg", flag_save=False):
    """
    Create a contour plot of the solution.
    """
    min_level = min(z)
    max_level = max(z)
    levels = np.linspace(min_level, max_level, 30)
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # create a Delaunay triangultion
    tri = mtri.Triangulation(x, y)

    # Defin colormap
    # cmap = "coolwarm"
    # cmap = "hot"
    cmap = cc.cm["fire"]
    # Plot solution
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    plot = ax.tricontourf(tri, z, levels=levels, cmap=cmap)

    norm = mpl.colors.Normalize(vmin=min_level, vmax=max_level)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="right"
    )

    cbar.ax.set_ylabel(r" $\mathbf{T ^\circ C}$", rotation=0, labelpad=15, ha="center")
    cbar.ax.yaxis.set_label_coords(0.70, 1.05)  # Adjust x and y as needed
    cbar.set_ticks([min(z), (min(z) + max(z)) / 2.0, max(z)])

    if flag_save == True:
        dir = "/Users/seiyonarulampalam/git/FEA/2DThermal/Figures/"
        plt.savefig(dir + fname, dpi=800, edgecolor="none")
    else:
        plt.show()
    return
