import numpy as np


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
    # print(f"{wts=}")
    # print(f"{xi=}")
    return wts, xi


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


def shape_func_vals(xi):
    """Return the vector of shape function values evaluate at xi"""
    N, _ = Order1ShapeFunc(xi)
    return N


def jac(x_vec, y_vec):
    x1 = x_vec[0]
    x2 = x_vec[1]
    y1 = y_vec[0]
    y2 = y_vec[1]
    return np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))


# def Order1Map(xi, x_vec) -> float:
#     """
#     Map the local reference frame to the global reference fram

#     Parameters
#     ----------
#     xi : float
#         quadrature point of interest [0,1]
#     x_vec : array
#         first entry is the left node position,
#         and the second entry is the 2nd node of the element

#     Returns
#     -------
#     float
#         the mapped global coordinate of the quadrature point
#     """
#     x1 = x_vec[0]
#     x2 = x_vec[1]

#     f, _ = Order1ShapeFunc(xi)
#     N1 = f[0]
#     N2 = f[1]

#     return x1 * N1 + x2 * N2


# def jac(x_vec) -> float:
#     """
#     Compute the jacobian of the element

#     Parameters
#     ----------
#     x_vec : array
#         first entry is the left node position,
#         and the second entry is the 2nd node of the element

#     Returns
#     -------
#     float
#         Jacobian is the scaling factor for a change of frame for integration.
#         In our case we want to change our integration fram from the global coordiantes
#         to the [0,1] reference frame. The jacobian acts as a scaling term.

#     """
#     x1 = x_vec[0]
#     x2 = x_vec[1]
#     return x2 - x1


# def shape_func_derivatives(xi, x_vec) -> np.ndarray:
#     """
#     Compute the shape function derivatives w.r.t the global frame

#     Parameters
#     ----------
#     xi : float
#         quadrature point of interest [0,1]
#     x_vec : array
#         first entry is the left node position,
#         and the second entry is the 2nd node of the element


#     Returns
#     -------
#     array
#         First element is the derivative of the first shape function w.r.t. x.
#         Second element is the derivative of the second shape function w.r.t. x.
#     """
#     jac = jac(x_vec)

#     _, df = Order1ShapeFunc(xi)
#     dN1_dxi = df[0]
#     dN2_dxi = df[1]

#     dN1_dx = dN1_dxi / jac
#     dN2_dx = dN2_dxi / jac

#     dNdx = np.array([dN1_dx, dN2_dx])
#     return dNdx
