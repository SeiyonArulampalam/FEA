import numpy as np

# Specify the precision of the print statement
np.set_printoptions(precision=4)

"""
Helper Functions that define the shape functions for a first order 
traingular elements. Includes the functions that determine the required 
derivatives for the FE analysis. Note: xi, eta are in the [0,1] ref. frame
"""


def GaussOrder3():
    """Return weights and quadrature points for order 2 triangle"""
    wts = [-27.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0]
    xi_eta = [[1.0 / 3.0, 1.0 / 3.0], [0.2, 0.6], [0.2, 0.2], [0.6, 0.2]]
    return wts, xi_eta


def order_1_N1(xi, eta):
    """Shape Function 1 for 1st Order Elements"""
    f = 1 - xi - eta
    df_dxi = -1
    df_deta = -1
    return f, df_dxi, df_deta


def order_1_N2(xi, eta):
    """Shape Function 2 for 1st Order Elements"""
    f = xi
    df_dxi = 1
    df_deta = 0
    return f, df_dxi, df_deta


def order_1_N3(xi, eta):
    """Shape Function 3 for 1st Order Elements"""
    f = eta
    df_dxi = 0
    df_deta = 1
    return f, df_dxi, df_deta


def x_mapping(xi, eta, x_vec):
    """Mapping from xi, eta plane to x-y frame"""
    x1 = x_vec[0]  # node 1 x-coord
    x2 = x_vec[1]  # node 2 x-coord
    x3 = x_vec[2]  # node 3 x-coord
    comp1, _, _ = order_1_N1(xi, eta) * x1
    comp2, _, _ = order_1_N2(xi, eta) * x2
    comp3, _, _ = order_1_N3(xi, eta) * x3
    return comp1 + comp2 + comp3


def y_mapping(xi, eta, y_vec):
    """Mapping from xi, eta plane to x-y frame"""
    y1 = y_vec[0]  # node 1 y-coord
    y2 = y_vec[1]  # node 2 y-coord
    y3 = y_vec[2]  # node 3 y-coord
    comp1, _, _ = order_1_N1(xi, eta) * y1
    comp2, _, _ = order_1_N2(xi, eta) * y2
    comp3, _, _ = order_1_N3(xi, eta) * y3
    return comp1 + comp2 + comp3


def jac(x_vec, y_vec):
    """Jacobian matrix"""
    x1 = x_vec[0]
    x2 = x_vec[1]
    x3 = x_vec[2]

    y1 = y_vec[0]
    y2 = y_vec[1]
    y3 = y_vec[2]

    jacobian = [[-x1 + x2, -y1 + y2], [-x1 + x3, -y1 + y3]]

    return np.array(jacobian)


def shape_func_derivatives(jacobian, xi, eta):
    """Compute the derivatives of the shape function wrt
    the x and y reference frame. Inputs are the jacobian, xi, and eta."""
    J_inv = np.linalg.inv(jacobian)

    _, dN1_dxi, dN1_deta = order_1_N1(xi, eta)
    deriv_vec_N1 = np.array([dN1_dxi, dN1_deta])
    dN1_vec = np.matmul(J_inv, deriv_vec_N1)

    _, dN2_dxi, dN2_deta = order_1_N2(xi, eta)
    deriv_vec_N2 = np.array([dN2_dxi, dN2_deta])
    dN2_vec = np.matmul(J_inv, deriv_vec_N2)

    _, dN3_dxi, dN3_deta = order_1_N3(xi, eta)
    deriv_vec_N3 = np.array([dN3_dxi, dN3_deta])
    dN3_vec = np.matmul(J_inv, deriv_vec_N3)

    dNi_dx = [dN1_vec[0], dN2_vec[0], dN3_vec[0]]
    dNi_dy = [dN1_vec[1], dN2_vec[1], dN3_vec[1]]

    return np.array(dNi_dx), np.array(dNi_dy)


def shape_func_vals(xi, eta):
    """Return a vector of shape function values given the xi and eta coords"""
    N1, _, _ = order_1_N1(xi, eta)
    N2, _, _ = order_1_N2(xi, eta)
    N3, _, _ = order_1_N3(xi, eta)
    Ni = [N1, N2, N3]
    return np.array(Ni)
