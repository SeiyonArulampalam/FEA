import numpy as np
import matplotlib.pyplot as plt

"""
PDE: c1*∂u/∂t - ∂/∂x(a*∂u/∂x) + c0*u = g(x,t)

Insulated BC: k*A*∂u/∂t = 0 

Convection BC: k*A*∂u/∂t + β*A*u = 0

Solution: u = T - T_ambient

c1 = rho*cp*A
a = k*A
c0 = P*β

rho = mass density
β = heat transfer coeff
k = thermal conductivity
A = cross-section area
P = perimeter of rod end (curcumprence of rod)
cp =specific heat at constant pressure
g = internal excitation


The FEM discretization of a beam leads to:

Element (M)      (0)         (1)         (2)
            o-----------o-----------o-----------o-----> (x-axis)
Node (N)    0           1           2           3
"""


def Order1ShapeFunc(xi) -> tuple[np.ndarray, np.ndarray[np.any]]:
    """
    Definition of the shape functions for a linear element

    Parameters
    ----------
    xi : float
        quadrature point in [0,1] frame

    Returns
    -------
    arrays
        f : the value of the shape functions at xi
        df: derivative of shape functions at xi
    """
    # shape functions
    f1 = 1 - xi
    f2 = xi
    f = np.array([f1, f2])

    # shape function derivatives
    df1 = -1
    df2 = 1
    df = np.array([df1, df2])

    return f, df


def Order1Map(xi, x_vec) -> float:
    """
    Map the local reference frame to the global reference fram

    Parameters
    ----------
    xi : float
        quadrature point of interest [0,1]
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element

    Returns
    -------
    float
        the mapped global coordinate of the quadrature point
    """
    x1 = x_vec[0]
    x2 = x_vec[1]

    f, _ = Order1ShapeFunc(xi)
    N1 = f[0]
    N2 = f[1]

    return x1 * N1 + x2 * N2


def Order1Jacobian(x_vec) -> float:
    """
    Compute the jacobian of the element

    Parameters
    ----------
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element

    Returns
    -------
    float
        Jacobian is the scaling factor for a change of frame for integration.
        In our case we want to change our integration fram from the global coordiantes
        to the [0,1] reference frame. The jacobian acts as a scaling term.

    """
    x1 = x_vec[0]
    x2 = x_vec[1]
    return x2 - x1


def Order1_dNdx(xi, x_vec) -> np.ndarray:
    """
    Compute the shape function derivatives w.r.t the global frame

    Parameters
    ----------
    xi : float
        quadrature point of interest [0,1]
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element


    Returns
    -------
    array
        First element is the derivative of the first shape function w.r.t. x.
        Second element is the derivative of the second shape function w.r.t. x.
    """
    jac = Order1Jacobian(x_vec)

    _, df = Order1ShapeFunc(xi)
    dN1_dxi = df[0]
    dN2_dxi = df[1]

    dN1_dx = dN1_dxi / jac
    dN2_dx = dN2_dxi / jac

    dNdx = np.array([dN1_dx, dN2_dx])
    return dNdx


def GaussOrder4():
    """
    Define the Gaussian Quadrature points and weights exact up to order 4. The
    weights are multiplied by 0.5 to shift the weights from wiki that are from the [-1,1] frame.
    Similarly, the quadrature points are shifted as well.

    var_hat -> [0,1] frame. While w and xi are in the [-1,1] frame
    w_hat = 0.5*w
    xi_hat = 0.5*xi + 0.5

    Returns
    -------
    arrays
        wts: quadrature weight in [0,1] frame
        xi : quadrature pts in the [0,1] frame
    """
    # Weights and quad points shifted s.t. xi in [0,1] coordinates
    wts = np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927]) * 0.5
    xi = np.array([0.0, 0.538469, -0.538469, 0.90618, -0.90618]) * 0.5 + 0.5
    return wts, xi


def Integrate_K(
    wts,
    xi_pts,
    jac_func,
    shape_func_derivatives,
    shape_func_vals,
    x_vec,
    a,
    c0,
    i,
    j,
) -> float:
    """
    Evaluate the integral that defines the elemental stiffness matrix input.

    Parameters
    ----------
    wts : array
        quadrature weights
    xi_pts : array
        quadrature points
    jac_func : function
        jacobian function handle
    shape_func_derivatives : function
        shape function derivatives handle that returns the derivative at quad point
    shape_func_vals : function
        shape function handle that returns the derivative at quad point
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element
    a : float
        elemental coeff
    c0 : float
        elemental coeff
    i : int
        index for the local stiffness matrix (row)
    j : int
        index for the local stiffness matrix (col)

    Returns
    -------
    float
        Gaussian quadrature integration result
    """
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]
        xi = xi_pts[k]
        jac = jac_func(x_vec)
        dN_dx = shape_func_derivatives(xi, x_vec)
        N, _ = shape_func_vals(xi)

        comp1 = a * dN_dx[i] * dN_dx[j]
        comp2 = c0 * N[i] * N[j]

        I += weight * (comp1 + comp2) * jac

    return I


def Integrate_f(
    wts,
    xi_pts,
    jac_func,
    shape_func_vals,
    x_vec,
    g,
    i,
) -> float:
    """
    Evaluate the elemental f values to build the global F vector.

    Parameters
    ----------
    wts : array
        quadrature weights
    xi_pts : array
        quadrature points
    jac_func : function
        jacobian function handle
    shape_func_vals : function
        shape function handle that returns the derivative at quad point
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element
    g : float
        excitation value for the element
    i : int
        local node number

    Returns
    -------
    float
        Gaussian quadrature integration result
    """
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]
        xi = xi_pts[k]
        jac = jac_func(x_vec)
        N, _ = shape_func_vals(xi)

        comp1 = N[i] * g

        I += weight * comp1 * jac

    return I


def Integrate_C(
    wts,
    xi_pts,
    jac_func,
    x_vec,
    shape_func_vals,
    c1,
    i,
    j,
) -> float:
    """
    Compute the matrix responsible for the time dependence.

    Parameters
    ----------
    wts : array
        quadrature weights
    xi_pts : array
        quadrature points
    jac_func : function
        jacobian function handle
    shape_func_vals : function
        shape function handle that returns the derivative at quad point
    x_vec : array
        first entry is the left node position,
        and the second entry is the 2nd node of the element
    c1 : float
        physics coeff for the element
    i : int
        local index of matrix (row)
    j : int
        local index of matrix (col)

    Returns
    -------
    float
        Gaussian quadrature integration result
    """
    I = 0.0
    for k in range(len(wts)):
        weight = wts[k]
        xi = xi_pts[k]
        jac = jac_func(x_vec)
        N, _ = shape_func_vals(xi)

        comp1 = c1 * N[i] * N[j]

        I += weight * comp1 * jac
    return I


def plot_steady_state_result(xloc, steady_state_soln) -> None:
    """
    Plot the steady state FEA solution

    Parameters
    ----------
    xloc : array
        Node coordinates
    steady_state_soln : array
        Temperature at each node
    """
    # Define the exact solution from the textbook
    xloc_nelems_2 = np.array([0.0, 0.025, 0.05])
    xloc_nelems_4 = np.array([0.0, 0.0125, 0.025, 0.0375, 0.05])
    soln_nelems_2 = np.array([300.0, 217.98, 192.83])
    soln_nelems_4 = np.array([300.0, 251.52, 218.92, 200.16, 194.03])

    num_elems = len(xloc) - 1  # Compute the total number of elements

    # Plot Results
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(xloc_nelems_2, soln_nelems_2, "D", label="2 Elems. Txtbook")
    ax.plot(xloc_nelems_4, soln_nelems_4, "x", label="4 Elems. Txtbook")
    ax.plot(xloc, steady_state_soln, "k-", label="FEA")
    ax.set_xlabel("Location (m)")
    ax.set_ylabel("Temperature (C)")
    ax.set_title(f"Steady-State Heat transfer simulation with {num_elems} elements")
    plt.grid()
    plt.show()
    # plt.savefig("steady.jpg", dpi=800)
