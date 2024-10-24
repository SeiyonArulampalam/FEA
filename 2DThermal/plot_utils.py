import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.tri as mtri


def contour_mpl(xyz_nodeCoords, z, fname="contour.jpg", flag=False):
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
    cmap = "coolwarm"

    # Plot solution
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    plot = ax.tricontourf(tri, z, levels=levels, cmap=cmap)

    norm = mpl.colors.Normalize(vmin=min_level, vmax=max_level)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="right"
    )

    cbar.set_ticks([min(z), (min(z) + max(z)) / 2.0, max(z)])

    if flag == True:
        plt.savefig(fname, dpi=800, edgecolor="none")

    plt.show()
    return
