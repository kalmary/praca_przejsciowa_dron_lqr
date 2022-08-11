import math
from numpy import array
from numpy import linalg
from numpy import arange
import numpy as np
from control import lqr
import matplotlib.pyplot as plt

from fd_rk45 import *
from lqr2 import *

global K_gain
az_turbulence = 0.
ax_wind = 0.
az_wind = 0.

rad2deg = 180 / np.pi
deg2rad = np.pi / 180


#
# --------------------------------------------------------------
#
def declare_matrix(n, m):
    #
    return array([[0.0 for j in range(0, m)] for i in range(0, n)])


#
# --------------------------------------------------------------
#
def declare_vector(n):
    #
    return array([0.0 for i in range(0, n)])


#
# --------------------------------------------------------------
#
def Jacob_AB(RHS, y, t, u_control, n, m):
    #
    A = declare_matrix(n, n)
    B = declare_matrix(n, m)
    dy = 1.0e-6;
    f0 = RHS(y, t, u_control)
    for i in range(0, n):
        yp = array(y)
        yp[i] += dy
        f = RHS(yp, t, u_control)
        for j in range(0, n):
            A[j, i] = (f[j] - f0[j]) / dy

    for i in range(0, m):
        up = array(u_control)
        up[i] += dy
        f = RHS(y, t, up)
        for j in range(0, n):
            B[j, i] = (f[j] - f0[j]) / dy
    return A, B


#
# --------------------------------------------------------------
#
def RHS(x, t, u_control):  ########BLAD
    n = len(x)
    print(ax_wind)
    x_prim = np.zeros((n, 1))

    g = 9.81
    S = 1.

    mass = 25.
    Iy = 100.

    vx = x[0] + ax_wind
    vz = x[1] + az_wind

    alpha = np.arctan2(vz, vx)
    V = np.sqrt(vz * vz + vx * vx)

    CD_0 = 0.30
    CD = CD_0

    rho_0 = 1.225

    #exit()
    Q_dyn = 0.5 * rho_0 * V * V
    L = 0.
    D = Q_dyn * S * CD
    G = mass * g
    Th = 1.
    # Thrust_1 = Th * u[0]
    # Thrust_2 = Th * u[1]
    Thrust_1 = 0.5 * G + u_control[0]
    Thrust_2 = 0.5 * G + u_control[1]
    # Thrust_1 = 0.5 * G + x[6]
    # Thrust_2 = 0.5 * G + x[7]
    if n == 8:
        Thrust_1 = x[6]
        Thrust_2 = x[7]
    cm_q = -0.01

    Tau = 0.05

    beta = 0.0 * deg2rad
    cb = np.cos(beta)
    sb = np.sin(beta)
    x_prim[0] = (-D * np.cos(alpha) + L * np.sin(alpha) - G * np.sin(x[5]) - Thrust_1 * sb + Thrust_2 * sb) / mass - x[
        2] * vz
    x_prim[1] = (-D * np.sin(alpha) - L * np.cos(alpha) + G * np.cos(x[5]) - Thrust_1 * cb - Thrust_2 * cb) / mass + x[
        2] * vx + az_turbulence
    x_prim[2] = (0.5 * (Thrust_2 * cb - Thrust_1 * cb) + cm_q * x[2]) / Iy
    x_prim[3] = np.cos(x[5]) * vx + np.sin(x[5]) * vz
    x_prim[4] = -np.sin(x[5]) * vx + np.cos(x[5]) * vz
    x_prim[5] = x[2]

    if n == 8:
        x_prim[6] = (1.0 / Tau) * (-x[6] + Th * u_control[0])
        x_prim[7] = (1.0 / Tau) * (-x[7] + Th * u_control[1])

    return x_prim


#
# --------------------------------------------------------------
#
def trajectory(X, Vel, dt):
    # Z = 10.0 * np.sin(1.5 * X) + 3.0 * np.sin(2.11 * X) + 1.0 * np.sin(3.43 * X) + 1.0 * np.sin(4.7 * X)
    # dx = Vel * dt
    # X1 = X + dx
    # Z1 = 10.0 * np.sin(1.5 * X1) + 3.0 * np.sin(2.11 * X1) + 1.0 * np.sin(3.43 * X1) + 1.0 * np.sin(4.7 * X1)
    # alpha = math.atan2(Z1 - Z, dx)

    Z = 1.
    if X <= 1.:
        Z = 1.
    elif 1. < X < 1.5:
        Z = 1. + (X - 1.) * 10.
    elif 1.5 <= X <= 2.:
        Z = 6.
    elif 2. <= X <= 2.5:
        Z = 6. - (X - 2.) * 10.
    else:
        Z = 1.

    Z1 = Z

    dx = Vel * dt
    X = X + dx

    if X <= 1.:
        Z = 1.
    elif 1.0 < X < 1.5:
        Z = 1. + (X - 1.) * 10.
    elif 1.5 <= X <= 2.:
        Z = 6.
    elif 2. <= X <= 2.5:
        Z = 6. - (X - 2.) * 10.
    else:
        Z = 1.0

    alpha = math.atan2(Z - Z1, dx)

    return Z, alpha


def getQR(n, m):
    Q = np.diag(np.diag(np.ones((n, n))))
    R = np.diag(np.diag(np.ones((m, m))))

    Q[0, 0] = 1000.
    Q[1, 1] = 1000.
    Q[2, 2] = 0.1
    Q[3, 3] = 10.
    Q[4, 4] = 100.
    Q[5, 5] = 1.e+03

    if n == 8:
        Q = 10000 * Q

    return Q, R


#
# --------------------------------------------------------------
#
def main():
    # n = 6  # dimension of x
    n = 8  # width engines
    m = 2  # dimension of u

    z0 = 2.
    h_flight = 1.  # over the terrain
    c_turb = 1000.
    X_turb_1 = 1500.
    X_turb_2 = 2000.

    x = declare_vector(n)
    x[3] = -z0
    u_control = declare_vector(m)

    Vel = 0.1  # /3.6 to kmph

    t = 0.0
    t_end = 100.0
    dt = 0.01

    t_pom = int((t_end + dt) / dt)
    tp = []
    yp = np.zeros((len(x), t_pom))
    up = np.zeros((len(u_control), t_pom))
    gp = []
    zp = []

    #with np.printoptions(threshold=np.inf):
    #    print(yp)
    i = 0
    while t < t_end + dt:
        X = x[3]

        Z0 = 5.
        z_terr, alpha = trajectory(X, Vel, dt)
        Vx = Vel * np.cos(alpha)
        Vz = Vel * np.sin(alpha)

        z_ref = z_terr + h_flight

        tp.append(t)
        yp[:, i]=np.transpose(x)
        up[:, i]=np.transpose(u_control)
        gp.append(z_terr)
        zp.append(z_ref)

        x_ref = X

        Q, R = getQR(n, m)

        # get e
        e = np.zeros((n, 1))
        e[0] = x[0] - (np.cos(x[5]) * Vx + np.sin(x[5]) * Vz)
        e[1] = x[1] - (np.sin(x[5]) * Vx - np.cos(x[5]) * Vz)
        e[2] = x[2] - 0
        e[3] = x[3] - (x_ref)
        e[3] = 0.
        e[4] = x[4] - (-z_ref)
        e[5] = x[5] - 0.

        A, B = Jacob_AB(RHS, x, t, u_control, n, m)

        K_gain = lqr2(A, B, Q, R)[0]

        u_control = -K_gain * e

        u_max = 10000.
        u_control[0] = np.maximum(-u_max, np.minimum(u_max, u_control[0]))
        u_control[1] = np.maximum(-u_max, np.minimum(u_max, u_control[1]))

        az_turbulence=0.
        ax_wind=0.
        az_wind=0.

        if X_turb_1 < X < X_turb_2:
            az_turbulence = c_turb * (1.0 - 2.0 * np.random.rand())
            ax_wind = 0.0
            az_wind = 0.0; #15.5 + 3.0 * (1.0 - 2.0 * rand()) ????

        x = fd_rk45(RHS, x, t, dt, u_control)


        i += 1
        t += dt


if __name__ == '__main__':
    main()
