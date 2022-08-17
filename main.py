import math
from numpy import array
from numpy import linalg
from numpy import arange
import numpy as np
from control import lqr
from matplotlib.pyplot import *
import matplotlib.pyplot as plt

from fd_rk45 import *
from lqr2 import *

global K_gain


# n = 6  # dimension of x
n = 8  # width engines
m = 2  # dimension of u

rad2deg = 180 / np.pi
deg2rad = np.pi / 180


#
# --------------------------------------------------------------
#
def declare_matrix(n, m):  # dziala
    #
    return array([[0.0 for j in range(0, m)] for i in range(0, n)])


#
# --------------------------------------------------------------
#
def declare_vector(n):  # dziala
    #
    return array([0.0 for i in range(0, n)])


#
# --------------------------------------------------------------
#
def Jacob_AB(RHS, y, t, u_control, n, m, az_turbulence, ax_wind, az_wind):  # raczej dziala
    A = declare_matrix(n, n)
    B = declare_matrix(n, m)
    dy = 1.0e-6
    f0 = RHS(y, t, u_control, az_turbulence, ax_wind, az_wind)
    for i in range(0, n):
        yp = array(y)
        yp[i] += dy
        f = RHS(yp, t, u_control, az_turbulence, ax_wind, az_wind)
        for j in range(0, n):
            A[j, i] = (f[j] - f0[j]) / dy

    for i in range(0, m):
        up = array(u_control)
        up[i] += dy
        f = RHS(y, t, up, az_turbulence, ax_wind, az_wind)
        for j in range(0, n):
            B[j, i] = (f[j] - f0[j]) / dy
    return A, B


#
# --------------------------------------------------------------
#
def RHS(x, t, u_control, az_turbulence, az_wind, ax_wind):
    n = len(x)
    dx_dt = np.zeros((n, 1))

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
    rho = rho_0 * math.pow((1.0 - math.fabs(x[4]) / 44300.0), 4.256)

    # exit()
    Q_dyn = 0.5 * rho * V * V
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

    beta = 0. * deg2rad
    cb = np.cos(beta)
    sb = np.sin(beta)
    dx_dt[0] = (-D * math.cos(alpha) + L * math.sin(alpha) - G * math.sin(
        x[5]) - Thrust_1 * sb + Thrust_2 * sb) / mass - x[
                   2] * vz
    dx_dt[1] = (-D * math.sin(alpha) - L * math.cos(alpha) + G * math.cos(
        x[5]) - Thrust_1 * cb - Thrust_2 * cb) / mass + x[
                   2] * vx + az_turbulence
    dx_dt[2] = (0.5 * (Thrust_2 * cb - Thrust_1 * cb) + cm_q * x[2]) / Iy
    dx_dt[3] = np.cos(x[5]) * vx + np.sin(x[5]) * vz
    dx_dt[4] = -np.sin(x[5]) * vx + np.cos(x[5]) * vz
    dx_dt[5] = x[2]

    if n == 8:
        dx_dt[6] = (1.0 / Tau) * (-x[6] + Th * u_control[0])
        dx_dt[7] = (1.0 / Tau) * (-x[7] + Th * u_control[1])
    return dx_dt


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


#
# ----------------------------------------------------------
#
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
# ----------------------------------------------------------
#
def keep_my_size(vector):
    vector2 = declare_vector(len(vector))

    for i in range(0, len(vector2)):  # troche skomplikowane ale tylko tak upewniam sie, ze to macierz 1D
        vector2[i] = vector[i]

    return vector2


#
# ----------------------------------------------------------
#
def mdl(X, Y, theta, c):
    xs0 = np.array([-0.1, 0.1, 0.1, -0.1, -0.1]) * c
    ys0 = np.array([-0.1, 0.1, 0.1, -0.1, -0.1]) * c

    xs = declare_vector(5)
    ys = declare_vector(5)

    for i in range(0, 5):
        xs[i] = X + xs0[i] * math.cos(theta) - ys0[i] * math.sin(theta)
        ys[i] = Y + xs0[i] * math.sin(theta) - ys0[i] * math.cos(theta)

    return xs, ys


#
# --------------------------------------------------------------
#
def main():
    z0 = 2.
    h_flight = 1.  # over the terrain
    c_turb = 1000.
    X_turb_1 = 1500.
    X_turb_2 = 2000.

    az_turbulence = 0.
    ax_wind = 0.
    az_wind = 0.

    x = declare_vector(n)
    x[4] = -z0

    u_control = declare_vector(m)

    Vel = 0.4  # /3.6 to [m/s]

    t = 0.0
    t_end = 100.0
    dt = 0.005



    # -------------------PLOT----------------------
    plt.figure(figsize=(15, 7))
    plt.ion()
    ax = plt.gca()
    plt.axis([0., 10., 0, 8])

    # lines
    line0, = ax.plot([], [], 'r')
    line1, = ax.plot([], [], 'b')
    line2, = ax.plot([], [], 'g')

    # points
    #pnt0, = ax.plot([], [], 'or')
    pnt1, = ax.plot([], [], 'ob')
    #pnt2, = ax.plot([], [], 'og')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    # -------------------PLOT----------------------

    tp = []

    # yp = np.zeros((len(x), t_pom))
    #yp0 = []
    #yp1 = []
    #yp2 = []
    yp3 = []
    yp4 = []
    yp4_2 = []
    #yp5 = []
    #yp6 = []
    #yp7 = []

    # up = np.zeros((len(u_control), t_pom))
    up0 = []
    up1 = []

    gp = []
    zp = []

    # with np.printoptions(threshold=np.inf):
    #    print(yp)

    i = 0
    while t < t_end + dt:
        X = x[3]

        Z0 = 5.
        z_terr, alpha = trajectory(x[3], Vel, dt)

        Vx = Vel * np.cos(alpha)
        Vz = Vel * np.sin(alpha)

        z_ref = z_terr + h_flight

        tp.append(t)

        # yp[:, i] = np.transpose(x)
        #yp0.append(x[0])
        #yp1.append(x[1])
        #yp2.append(x[2])
        yp3.append(x[3])
        #yp4.append(x[4])
        yp4_2.append(-x[4])
        #yp5.append(x[5])
        #yp6.append(x[6])
        #yp7.append(x[7])

        up0.append(u_control[0])
        up1.append(u_control[1])

        gp.append(z_terr)
        zp.append(z_ref)

        #print(f"\nlen yp3={len(yp3)}\nlen yp4_2={len(yp4_2)}")

        x_ref = X

        Q, R = getQR(n, m)
        A, B = Jacob_AB(RHS, x, t, u_control, n, m, az_turbulence, ax_wind, az_wind)

        # get e
        e = np.zeros((n, 1))
        e[0] = x[0] - (np.cos(x[5]) * Vx + np.sin(x[5]) * Vz)
        e[1] = x[1] - (np.sin(x[5]) * Vx - np.cos(x[5]) * Vz)
        e[2] = x[2] - 0
        e[3] = x[3] - (x_ref)
        e[3] = 0.
        e[4] = x[4] - (-z_ref)
        e[5] = x[5] - 0.


        K_gain, _, _ = lqr2(A, B, Q, R)

        u_control = keep_my_size(-K_gain @ e)

        u_max = 10000.
        u_control[0] = np.maximum(-u_max, np.minimum(u_max, u_control[0]))
        u_control[1] = np.maximum(-u_max, np.minimum(u_max, u_control[1]))


        if X_turb_1 < X < X_turb_2:
            az_turbulence = c_turb * (1.0 - 2.0 * np.random.rand())
            ax_wind = 0.0
            az_wind = 0.0  # 15.5 + 3.0 * (1.0 - 2.0 * rand()) ????

        # -------------------PLOT----------------------
        v_x = Vx  # sin(x[5]) * x[0] -cos(x[5]) * x[1]
        v_z = Vz  # sin(x[5]) * x[0] -cos(x[5]) * x[1]
        V = math.sqrt(x[0] * x[0] + x[1] * x[1])
        e_v = Vel - V
        e = e.flatten()
        theta = x[5] * rad2deg
        alpha_deg = alpha * rad2deg

        #alt = -x[4]
        #gamma = math.atan(((math.sin(x[5]) * x[0]) - (math.cos(x[5]) * x[1])) / (
        #        (math.cos(x[5]) * x[0]) + (math.sin(x[5]) * x[2]))) * rad2deg


        #
        txt = 't={:8.4f}    V={:9f} [m/s]   v_x={:9f}   v_z={:9f}   theta={:9f}     alpha={:9f}' \
              '\nu1={:9f}   u2={:9f}' \
              '\ne_v={:9f} e(z)={:9f}'.format(t, V, v_x, v_z, theta, alpha_deg, u_control[0], u_control[1], e_v, e[4])
        #
        line0.set_data(yp3, zp) #over the ground
        line1.set_data(yp3, yp4_2)
        line2.set_data(yp3, gp) #ground


        pnt1.set_data(x[3], -x[4])

        # -------------------PLOT----------------------

        i+=1

        if i % 20 == 0:
            ax.relim()
            ax.autoscale_view(False, True, False)
            ax.grid(True)
            plt.legend([line0, line1, line2], ["reference height(t)", "y(t)", "ground height(t)"], loc=2)
            plt.subplots_adjust(left=0.07, right=0.95, bottom=0.1, top=0.85)
            plt.xlabel('position x(t)', size=15)
            plt.ylabel('height y(t)', size=15)
            plt.axis([yp3[0] - 0.25, yp3[-1]+0.25, 0, 8])
            plt.title("Calkowanie ruchu drona\n\n" + txt)
            plt.draw()
            plt.pause(0.001)


        if yp3[-1] > 1.:
            tp.pop(0)
            yp3.pop(0)
            yp4_2.pop(0)
            zp.pop(0)
            gp.pop(0)


        x = fd_rk45(RHS, x, t, dt, u_control, az_turbulence, ax_wind, az_wind)

        t += dt
    plt.pause(1.5)

if __name__ == '__main__':
    main()
