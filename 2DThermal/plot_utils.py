import numpy as np
import matplotlib.animation as animation
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
    T_ambient = 20  # Define the ambient temperature
    min_level = T_ambient
    max_level = max(z)
    levels = np.linspace(min_level, max_level, 30)
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # create a Delaunay triangultion
    tri = mtri.Triangulation(x, y)

    # Define colormap
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
    cbar.set_ticks([T_ambient, (max(z) + T_ambient) * 0.5, max(z)])

    if flag_save == True:
        dir = "/Users/seiyonarulampalam/git/FEA/2DThermal/Figures/"
        plt.savefig(dir + fname, dpi=800, edgecolor="none")
    else:
        plt.show()
    return


def contour_mpl_animate(
    xyz_nodeCoords, u, dt, max_rel_err, fname="contour_animation.gif", flag_save=False
):
    """
    Create an animated contour plot using rows of u for each frame.
    """
    n_steps, n_nodes = u.shape
    T_ambient = 20  # Define the ambient temperature
    x = xyz_nodeCoords[:, 0]
    y = xyz_nodeCoords[:, 1]

    # Create a Delaunay triangulation
    tri = mtri.Triangulation(x, y)

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create the initial contour plot
    min_u = np.min(u)
    max_u = np.max(u)
    levels = np.linspace(min_u, max_u, 30)
    cmap = cc.cm["fire"]
    plot = ax.tricontourf(tri, u[0, :], levels=levels, cmap=cmap)
    plt.axis("equal")
    ax.set_title(f"Max Percent Error at Final Time Step: {max_rel_err:.2e}")
    ax.set_axis_off()
    norm = mpl.colors.Normalize(vmin=min_u, vmax=max_u)
    cbar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, location="right"
    )

    cbar.ax.set_ylabel(r" $\mathbf{T ^\circ C}$", rotation=0, labelpad=15, ha="center")
    cbar.ax.yaxis.set_label_coords(0.70, 1.05)  # Adjust x and y as needed
    cbar.set_ticks([min_u, (min_u + max_u) / 2.0, max_u])

    # Function to update the plot for each frame
    def update(frame):
        nonlocal plot  # Ensure we're modifying the outer 'plot' variable
        for collection in plot.collections:
            collection.remove()  # Remove old contours
        plot = ax.tricontourf(
            tri,
            u[frame, :],
            levels=levels,
            cmap=cmap,
        )  # Redraw the new contours
        ax.set_title(
            f"Max Percent Error at Final Time Step: {max_rel_err:.2e}, time = {dt*frame:.1e}s"
        )
        ax.set_axis_off()
        plt.axis("equal")

    # Create the animation
    n_frames = 20  # Specify the number of frames to animate
    ani = animation.FuncAnimation(fig, update, frames=n_frames, repeat=True)

    # Save or show the animation
    if flag_save:
        dir = "/Users/seiyonarulampalam/git/FEA/2DThermal/Figures/"
        ani.save(dir + fname, writer="pillow")
        print("\nGIF animation has sucessfully been created.")
    else:
        plt.show()
