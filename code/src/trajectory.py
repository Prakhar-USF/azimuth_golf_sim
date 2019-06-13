import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


"""
About imagemagick:
1) local:
$ conda install -c conda-forge imagemagick
$ python
>>> import matplotlib
>>> print(matplotlib.matplotlib_fname()) --> path to matplotlibrc file

$ vi matplotlibrc file
delete comment sign before animation.convert_path:covert  (last part of the file)

2) on EC2:
$ ssh...
$ sudo su - (go to root)
$ yum install php70-pecl-imagick
"""

# helper function
g = 9.806  # (m/sec^2)
h = 0.001  # (sec)
PI = math.pi


def cross_prod(a1, a2, a3, b1, b2, b3):
    vec_i = a2 * b3 - b2 * a3
    vec_j = b1 * a3 - a1 * b3
    vec_k = a1 * b2 - b1 * a2
    return vec_i, vec_j, vec_k


def unit_vec(c1, c2, c3):
    tmp_length = math.sqrt(c1 * c1 + c2 * c2 + c3 * c3)
    unit_i = c1 / tmp_length
    unit_j = c2 / tmp_length
    unit_k = c3 / tmp_length
    return unit_i, unit_j, unit_k


def vec_length(d1, d2, d3):
    return math.sqrt(d1 * d1 + d2 * d2 + d3 * d3)


def minus_vec(e1, e2, e3):
    return -1 * e1, -1 * e2, -1 * e3


def ax(v_ball_i, v_ball_j, v_ball_k, \
       w_unit_i, w_unit_j, w_unit_k, \
       C_D, C_L, rho_air, m, D):
    A = (D / 2) ** 2 * PI

    U_i = -1 * v_ball_i
    U_j = -1 * v_ball_j
    U_k = -1 * v_ball_k

    abs_FD = C_D * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2
    abs_FL = C_L * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2

    U_unit_i, U_unit_j, U_unit_k = unit_vec(U_i, U_j, U_k)
    FL_unit_i, FL_unit_j, FL_unit_k = cross_prod(U_unit_i, U_unit_j, U_unit_k, \
                                                 w_unit_i, w_unit_j, w_unit_k)
    ans = abs_FD * U_unit_i + abs_FL * FL_unit_i
    return ans / m


def ay(v_ball_i, v_ball_j, v_ball_k, \
       w_unit_i, w_unit_j, w_unit_k, \
       C_D, C_L, rho_air, m, D):
    A = (D / 2) ** 2 * PI

    U_i = -1 * v_ball_i
    U_j = -1 * v_ball_j
    U_k = -1 * v_ball_k

    abs_FD = C_D * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2
    abs_FL = C_L * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2

    U_unit_i, U_unit_j, U_unit_k = unit_vec(U_i, U_j, U_k)
    FL_unit_i, FL_unit_j, FL_unit_k = cross_prod(U_unit_i, U_unit_j, U_unit_k, \
                                                 w_unit_i, w_unit_j, w_unit_k)
    ans = abs_FD * U_unit_j + abs_FL * FL_unit_j
    return ans / m


def az(v_ball_i, v_ball_j, v_ball_k, \
       w_unit_i, w_unit_j, w_unit_k, \
       C_D, C_L, rho_air, m, D):
    A = (D / 2) ** 2 * PI

    U_i = -1 * v_ball_i
    U_j = -1 * v_ball_j
    U_k = -1 * v_ball_k

    abs_FD = C_D * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2
    abs_FL = C_L * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2

    U_unit_i, U_unit_j, U_unit_k = unit_vec(U_i, U_j, U_k)
    FL_unit_i, FL_unit_j, FL_unit_k = cross_prod(U_unit_i, U_unit_j, U_unit_k, \
                                                 w_unit_i, w_unit_j, w_unit_k)
    ans = abs_FD * U_unit_k + abs_FL * FL_unit_k - m * g
    return ans / m


def RK4(tn, \
        xn, yn, zn, \
        v_xn, v_yn, v_zn, \
        w_unit_i, w_unit_j, w_unit_k, \
        C_D, C_L, rho_air, m, D):
    # For step-by-step calculations of variables changed with time,
    # we integrate with fourth-order Runge-Kutta method,
    # and then integrate Euler method.
    # ---------------------------------------
    kx1 = ax(v_xn, v_yn, v_zn, w_unit_i, w_unit_j, w_unit_k, C_D, C_L, rho_air, m, D)
    ky1 = ay(v_xn, v_yn, v_zn, w_unit_i, w_unit_j, w_unit_k, C_D, C_L, rho_air, m, D)
    kz1 = az(v_xn, v_yn, v_zn, w_unit_i, w_unit_j, w_unit_k, C_D, C_L, rho_air, m, D)
    # ---------------------------------------
    kx2 = ax(v_xn + kx1 * h / 2, v_yn + ky1 * h / 2, v_zn + kz1 * h / 2, w_unit_i, w_unit_j, w_unit_k, C_D, C_L,
             rho_air, m, D)
    ky2 = ay(v_xn + kx1 * h / 2, v_yn + ky1 * h / 2, v_zn + kz1 * h / 2, w_unit_i, w_unit_j, w_unit_k, C_D, C_L,
             rho_air, m, D)
    kz2 = az(v_xn + kx1 * h / 2, v_yn + ky1 * h / 2, v_zn + kz1 * h / 2, w_unit_i, w_unit_j, w_unit_k, C_D, C_L,
             rho_air, m, D)
    # ---------------------------------------
    kx3 = ax(v_xn + kx2 * h / 2, v_yn + ky2 * h / 2, v_zn + kz2 * h / 2, w_unit_i, w_unit_j, w_unit_k, C_D, C_L,
             rho_air, m, D)
    ky3 = ay(v_xn + kx2 * h / 2, v_yn + ky2 * h / 2, v_zn + kz2 * h / 2, w_unit_i, w_unit_j, w_unit_k, C_D, C_L,
             rho_air, m, D)
    kz3 = az(v_xn + kx2 * h / 2, v_yn + ky2 * h / 2, v_zn + kz2 * h / 2, w_unit_i, w_unit_j, w_unit_k, C_D, C_L,
             rho_air, m, D)
    # ---------------------------------------
    kx4 = ax(v_xn + kx3 * h, v_yn + ky3 * h, v_zn + kz3 * h, w_unit_i, w_unit_j, w_unit_k, C_D, C_L, rho_air, m, D)
    ky4 = ay(v_xn + kx3 * h, v_yn + ky3 * h, v_zn + kz3 * h, w_unit_i, w_unit_j, w_unit_k, C_D, C_L, rho_air, m, D)
    kz4 = az(v_xn + kx3 * h, v_yn + ky3 * h, v_zn + kz3 * h, w_unit_i, w_unit_j, w_unit_k, C_D, C_L, rho_air, m, D)
    # ---------------------------------------
    v_xn1 = v_xn + h * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
    v_yn1 = v_yn + h * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6
    v_zn1 = v_zn + h * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6
    # ---------------------------------------
    tn1 = tn + h  # h = 0.001s
    # ---------------------------------------
    xn1 = xn + (v_xn + v_xn1) * h / 2
    yn1 = yn + (v_yn + v_yn1) * h / 2
    zn1 = zn + (v_zn + v_zn1) * h / 2
    # ---------------------------------------
    return tn1, v_xn1, v_yn1, v_zn1, xn1, yn1, zn1


# track function
def TRACK(m, D, rho_air, C_D, C_L, \
          v_ball, theta, phi, \
          w_theta, w_phi, \
          altitude):
    # set initial conditions
    t0 = 0.0
    x0 = 0.0
    y0 = 0.0
    z0 = 0.0
    v_x0 = v_ball * math.cos(theta) * math.cos(phi)
    v_y0 = v_ball * math.cos(theta) * math.sin(phi)
    v_z0 = v_ball * math.sin(theta)
    w_unit_i = math.cos(w_theta) * math.cos(w_phi)  # angular unit vector in i
    w_unit_j = math.cos(w_theta) * math.sin(w_phi)  # angular unit vector in j
    w_unit_k = math.sin(w_theta)  # angular unit vector in k

    # set input initial conditions
    elements = 100000  # simulate within 100 seconds
    show_t = np.zeros(elements)
    show_x = np.zeros(elements)
    show_y = np.zeros(elements)
    show_z = np.zeros(elements)
    show_vx = np.zeros(elements)
    show_vy = np.zeros(elements)
    show_vz = np.zeros(elements)
    show_t[0] = t0
    show_x[0] = x0
    show_y[0] = y0
    show_z[0] = z0
    show_vx[0] = v_x0
    show_vy[0] = v_y0
    show_vz[0] = v_z0

    # while-loop
    j = 0
    tmp_z = z0
    tmp_vz = 0
    step = 0
    while (tmp_z >= altitude or tmp_vz >= 0 and j + 1 < elements):  # altitude use 0 here ??
        show_t[j + 1], show_vx[j + 1], show_vy[j + 1], show_vz[j + 1], \
        show_x[j + 1], show_y[j + 1], show_z[j + 1] = \
            RK4(show_t[j], \
                show_x[j], show_y[j], show_z[j], \
                show_vx[j], show_vy[j], show_vz[j], \
                w_unit_i, w_unit_j, w_unit_k, \
                C_D, C_L, \
                rho_air, m, D)
        tmp_z = show_z[j + 1]
        tmp_vz = show_vz[j + 1]
        j = j + 1
        step = j
    print_show_x = ("%5.3f" % show_x[step]).strip()
    print_show_y = ("%5.3f" % show_y[step]).strip()
    print_show_t = ("%5.3f" % show_t[step]).strip()
    tmp_distance = math.sqrt(show_x[step] ** 2 + show_y[step] ** 2)
    print_distance = ("%5.3f" % tmp_distance).strip()
    """
    print('    Fly time:           ', print_show_t, '(sec)')
    print('    Fly distance:       ', print_distance, '(m)')
    print('    Drop location in X: ', print_show_x, '(m)')
    print('    Drop location in Y: ', print_show_y, '(m)')
    print('--------------------------------------------')
    """

    return show_x[:step + 1], show_y[:step + 1], show_z[:step + 1], \
           show_x[step], show_y[step], math.sqrt(show_x[step]**2 + show_y[step]**2)   # get final value
    # 0,1,2 --> track
    # 3,4,5  --> final position


def calculation(data):
    ball = data[["ball_speed", "launch_angle", "azimuth", "side_spin", "back_spin"]]
    ball.columns = ["launch_speed", "theta", "phi", "side_spin", "back_spin"]
    ball["theta"] = ball.theta.map(lambda x: math.radians(x))
    ball["phi"] = ball.phi.map(lambda x: math.radians(x))
    ball["launch_speed"] = ball["launch_speed"] * 0.44704  # mile/h --> m/s

    w_theta = np.zeros(ball.shape[0])
    w_phi = np.zeros(ball.shape[0])
    for i in range(ball.shape[0]):
        row = ball.iloc[i]
        w_theta[i] = math.atan(row[3]**2 * (math.cos(row[1]**2))**2 /
                               math.sqrt((row[3]*math.sin(row[1])*math.cos(row[2]) -
                                          row[4]*math.sin(row[2]))**2 + (row[3]*math.sin(row[1])*math.sin(row[2]) +
                                                                         row[4]*math.cos(row[2]))**2))
        w_phi[i] = math.atan((-row[3]*math.sin(row[1])*math.sin(row[2]) - row[4]*math.cos(row[2])) /
                             (-row[3]*math.sin(row[1])*math.cos(row[2]) + row[4]*math.sin(row[2])))
    ball["w_theta"] = w_theta
    ball["w_phi"] = w_phi
    return ball


def update_lines(num, data, line):
    line.set_data(data[0:2, :num])
    line.set_3d_properties(data[2, :num])
    return line,


def plot_animation(x_track, y_track, z_track):
    track = pd.DataFrame({"x":x_track, "y":y_track, "z":z_track})
    track = np.array(track.T)
    track_small = track[:, np.arange(0, len(x_track), 50)]
    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(10, 4))
    ax = p3.Axes3D(fig)
    # starting point
    line = ax.plot(track_small[0, 0:1], track_small[1, 0:1], track_small[2, 0:1])[0]
    # Setting the axes properties
    ax.set_xlim3d([0.0, 150.0])
    ax.set_xlabel('X')
    ax.set_ylim3d([-10.0, 10.0])
    ax.set_ylabel('Y')
    ax.set_zlim3d([0.0, 30.0])
    ax.set_zlabel('Z')
    ax.set_title('3D Test')
    ax.view_init(azim=-88, elev=10)
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, update_lines, track_small.shape[1]+5,
                                       fargs=(track_small, line),
                                       interval=2, blit=True, repeat=True)
    # save gif file
    line_ani.save('sample_animation6.gif', writer='imagemagick', fps=30)


def main():
    # load data
    data = pd.read_csv("all_sessions_refined.csv")
    ball = calculation(data)

    # prediction for 1 observation
    row1 = ball.iloc[1]
    result = TRACK(0.04593, 0.04267, 1.2, 0.16, 0.1,
                   row1[0], row1[1], row1[2], row1[5], row1[6], 0)
    x_track = result[0]
    y_track = result[1]
    z_track = result[2]
    coordinate_x = result[3]
    coordinate_y = result[4]
    coordinate_z = result[5]
    plot_animation(x_track, y_track, z_track)


if __name__ == '__main__':
    main()
