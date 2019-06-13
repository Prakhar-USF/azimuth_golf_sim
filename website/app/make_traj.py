import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as pyoff


class TrackSimulation:
    def __init__(self):
        self.g = 9.806  # (m/sec^2)
        self.h = 0.001  # (sec)
        self.PI = math.pi

    def cross_prod(self, a1, a2, a3, b1, b2, b3):
        """
        Helper function: calculate cross production for two vectors

        :param a1: float, x-axis coordinate of vector 1
        :param a2: float, y-axis coordinate of vector 1
        :param a3: float, z-axis coordinate of vector 1
        :param b1: float, x-axis coordinate of vector 2
        :param b2: float, y-axis coordinate of vector 2
        :param b3: float, z-axis coordinate of vector 2

        :return: float, coordinates of the cross production
        """
        vec_i = a2 * b3 - b2 * a3
        vec_j = b1 * a3 - a1 * b3
        vec_k = a1 * b2 - b1 * a2
        return vec_i, vec_j, vec_k

    def unit_vec(self, c1, c2, c3):
        """
        Helper function: calculate unit vector

        :param c1: float, x-axis coordinate of vector
        :param c2: float, y-axis coordinate of vector
        :param c3: float, z-axis coordinate of vector

        :return: float, coordinates of the unit vector
        """
        tmp_length = math.sqrt(c1 * c1 + c2 * c2 + c3 * c3)
        unit_i = c1 / tmp_length
        unit_j = c2 / tmp_length
        unit_k = c3 / tmp_length
        return unit_i, unit_j, unit_k

    def vec_length(self, d1, d2, d3):
        """
        Helper function: calculate the length of vector

        :param d1: float, x-axis coordinate of vector
        :param d2: float, y-axis coordinate of vector
        :param d3: float, z-axis coordinate of vector

        :return: float, Euclidean length of the vector
        """
        vec_length = math.sqrt(d1 * d1 + d2 * d2 + d3 * d3)
        return vec_length

    def minus_vec(self, e1, e2, e3):
        """
        Helper function: get a vector equal to given vector but in opposite
        direction
        :param e1: float, x-axis coordinate of vector
        :param e2: float, y-axis coordinate of vector
        :param e3: float, z-axis coordinate of vector

        :return: coordinates of vector which equals to the given vector but in
        the opposite direction
        """
        minus_vec = -1 * e1, -1 * e2, -1 * e3
        return minus_vec

    def ax(self, v_ball_i, v_ball_j, v_ball_k, w_unit_i, w_unit_j, w_unit_k,
           C_D, C_L, rho_air, m, D):
        """
        Helper function: calculate accelerated velocity on x-axis

        :param v_ball_i: float, x-axis coordinate of ball's velocity
        :param v_ball_j: float, y-axis coordinate of ball's velocity
        :param v_ball_k: float, z-axis coordinate of ball's velocity
        :param w_unit_i: float, x-axis coordinate of ball's rotation velocity
        :param w_unit_j: float, y-axis coordinate of ball's rotation velocity
        :param w_unit_k: float, z-axis coordinate of ball's rotation velocity
        :param C_D: float, drag force coefficient
        :param C_L: float, lift force coefficient
        :param rho_air: float, air density, constant
        :param m: float, mass of golf ball, constant
        :param D: float, diameter of golf ball, constant

        :return: x-axis value of accelerated velocity
        """
        A = (D / 2) ** 2 * self.PI

        U_i = -1 * v_ball_i
        U_j = -1 * v_ball_j
        U_k = -1 * v_ball_k

        abs_FD = C_D * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2
        abs_FL = C_L * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2

        U_unit_i, U_unit_j, U_unit_k = self.unit_vec(U_i, U_j, U_k)
        FL_unit_i, FL_unit_j, FL_unit_k = self.cross_prod(
            U_unit_i, U_unit_j, U_unit_k, w_unit_i, w_unit_j, w_unit_k)
        ans = abs_FD * U_unit_i + abs_FL * FL_unit_i
        a_x = ans / m

        return a_x

    def ay(self, v_ball_i, v_ball_j, v_ball_k, w_unit_i, w_unit_j, w_unit_k,
           C_D, C_L, rho_air, m, D):
        """
        Helper function: calculate accelerated velocity on y-axis

        :param v_ball_i: float, x-axis coordinate of ball's velocity
        :param v_ball_j: float, y-axis coordinate of ball's velocity
        :param v_ball_k: float, z-axis coordinate of ball's velocity
        :param w_unit_i: float, x-axis coordinate of ball's rotation velocity
        :param w_unit_j: float, y-axis coordinate of ball's rotation velocity
        :param w_unit_k: float, z-axis coordinate of ball's rotation velocity
        :param C_D: float, drag force coefficient
        :param C_L: float, lifg force coefficient
        :param rho_air: float, air density, constant
        :param m: float, mass of golf ball, constant
        :param D: float, diameter of golf ball, constant

        :return: y-axis value of accelerated velocity
        """
        A = (D / 2) ** 2 * self.PI

        U_i = -1 * v_ball_i
        U_j = -1 * v_ball_j
        U_k = -1 * v_ball_k

        abs_FD = C_D * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2
        abs_FL = C_L * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2

        U_unit_i, U_unit_j, U_unit_k = self.unit_vec(U_i, U_j, U_k)
        FL_unit_i, FL_unit_j, FL_unit_k = self.cross_prod(
            U_unit_i, U_unit_j, U_unit_k, w_unit_i, w_unit_j, w_unit_k)
        ans = abs_FD * U_unit_j + abs_FL * FL_unit_j
        a_y = ans / m

        return a_y

    def az(self, v_ball_i, v_ball_j, v_ball_k, w_unit_i, w_unit_j, w_unit_k,
           C_D, C_L, rho_air, m, D):
        """
        Helper function: calculate accelerated velocity on z-axis

        :param v_ball_i: float, x-axis coordinate of ball's velocity
        :param v_ball_j: float, y-axis coordinate of ball's velocity
        :param v_ball_k: float, z-axis coordinate of ball's velocity
        :param w_unit_i: float, x-axis coordinate of ball's rotation velocity
        :param w_unit_j: float, y-axis coordinate of ball's rotation velocity
        :param w_unit_k: float, z-axis coordinate of ball's rotation velocity
        :param C_D: float, drag force coefficient
        :param C_L: float, lifg force coefficient
        :param rho_air: float, air density, constant
        :param m: float, mass of golf ball, constant
        :param D: float, diameter of golf ball, constant

        :return: z-axis value of accelerated velocity
        """
        A = (D / 2) ** 2 * self.PI

        U_i = -1 * v_ball_i
        U_j = -1 * v_ball_j
        U_k = -1 * v_ball_k

        abs_FD = C_D * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2
        abs_FL = C_L * A * rho_air * (U_i ** 2 + U_j ** 2 + U_k ** 2) / 2

        U_unit_i, U_unit_j, U_unit_k = self.unit_vec(U_i, U_j, U_k)
        FL_unit_i, FL_unit_j, FL_unit_k = self.cross_prod(
            U_unit_i, U_unit_j, U_unit_k, w_unit_i, w_unit_j, w_unit_k)
        ans = abs_FD * U_unit_k + abs_FL * FL_unit_k - m * self.g
        a_z = ans / m

        return a_z

    def RK4(self, tn, xn, yn, zn, v_xn, v_yn, v_zn,
            w_unit_i, w_unit_j, w_unit_k, C_D, C_L, rho_air, m, D):
        """
        For movement of each time step, calculate total flying from beginning,
        velocity on x, y, z axis, and distance on x, y, z axis.

        :param tn: float, initial time
        :param xn: float, initial x-axis coordinate of ball position
        :param yn: float, initial y-axis coordinate of ball position
        :param zn: float, initail z-axis coordinate of ball position
        :param v_xn: float, initial x-axis coordinate of ball speed
        :param v_yn: float, initial y-axis coordinate of ball speed
        :param v_zn: float, initial z-axis coordinate of ball speed
        :param w_unit_i:float, initial x-axis coordinate of ball rotation speed
        :param w_unit_j:float, initial y-axis coordinate of ball rotation speed
        :param w_unit_k:float, initial z-axis coordinate of ball rotation speed
        :param C_D: float, drag force coefficient
        :param C_L: float, lifg force coefficient
        :param rho_air: float, air density, constant
        :param m: float, mass of golf ball, constant
        :param D: float, diameter of golf ball, constant

        :return: final time, coordinates of final speed,
        coordinates of final position

        Note
        ----
        For step-by-step calculations of variables changed with time,
        we integrate with fourth-order Runge-Kutta method,
        and then integrate Euler method.
        """
        # For step-by-step calculations of variables changed with time,
        # we integrate with fourth-order Runge-Kutta method,
        # and then integrate Euler method.
        # ---------------------------------------
        kx1 = self.ax(v_xn, v_yn, v_zn, w_unit_i, w_unit_j, w_unit_k,
                      C_D, C_L, rho_air, m, D)
        ky1 = self.ay(v_xn, v_yn, v_zn, w_unit_i, w_unit_j, w_unit_k,
                      C_D, C_L, rho_air, m, D)
        kz1 = self.az(v_xn, v_yn, v_zn, w_unit_i, w_unit_j, w_unit_k,
                      C_D, C_L, rho_air, m, D)
        # ---------------------------------------
        kx2 = self.ax(v_xn + kx1 * self.h / 2, v_yn + ky1 * self.h / 2,
                      v_zn + kz1 * self.h / 2, w_unit_i, w_unit_j,
                      w_unit_k, C_D, C_L,
                      rho_air, m, D)
        ky2 = self.ay(v_xn + kx1 * self.h / 2, v_yn + ky1 * self.h / 2,
                      v_zn + kz1 * self.h / 2, w_unit_i, w_unit_j,
                      w_unit_k, C_D, C_L,
                      rho_air, m, D)
        kz2 = self.az(v_xn + kx1 * self.h / 2, v_yn + ky1 * self.h / 2,
                      v_zn + kz1 * self.h / 2, w_unit_i, w_unit_j,
                      w_unit_k, C_D, C_L,
                      rho_air, m, D)
        # ---------------------------------------
        kx3 = self.ax(v_xn + kx2 * self.h / 2, v_yn + ky2 * self.h / 2,
                      v_zn + kz2 * self.h / 2, w_unit_i, w_unit_j,
                      w_unit_k, C_D, C_L,
                      rho_air, m, D)
        ky3 = self.ay(v_xn + kx2 * self.h / 2, v_yn + ky2 * self.h / 2,
                      v_zn + kz2 * self.h / 2, w_unit_i, w_unit_j,
                      w_unit_k, C_D, C_L,
                      rho_air, m, D)
        kz3 = self.az(v_xn + kx2 * self.h / 2, v_yn + ky2 * self.h / 2,
                      v_zn + kz2 * self.h / 2, w_unit_i, w_unit_j,
                      w_unit_k, C_D, C_L,
                      rho_air, m, D)
        # ---------------------------------------
        kx4 = self.ax(v_xn + kx3 * self.h, v_yn + ky3 * self.h,
                      v_zn + kz3 * self.h, w_unit_i, w_unit_j, w_unit_k, C_D,
                      C_L, rho_air, m, D)
        ky4 = self.ay(v_xn + kx3 * self.h, v_yn + ky3 * self.h,
                      v_zn + kz3 * self.h, w_unit_i, w_unit_j, w_unit_k, C_D,
                      C_L, rho_air, m, D)
        kz4 = self.az(v_xn + kx3 * self.h, v_yn + ky3 * self.h,
                      v_zn + kz3 * self.h, w_unit_i, w_unit_j, w_unit_k, C_D,
                      C_L, rho_air, m, D)
        # ---------------------------------------
        v_xn1 = v_xn + self.h * (kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6
        v_yn1 = v_yn + self.h * (ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6
        v_zn1 = v_zn + self.h * (kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6
        # ---------------------------------------
        tn1 = tn + self.h  # h = 0.001s
        # ---------------------------------------
        xn1 = xn + (v_xn + v_xn1) * self.h / 2
        yn1 = yn + (v_yn + v_yn1) * self.h / 2
        zn1 = zn + (v_zn + v_zn1) * self.h / 2
        # ---------------------------------------
        return tn1, v_xn1, v_yn1, v_zn1, xn1, yn1, zn1

    def TRACK(self, m, D, rho_air, C_D, C_L, v_ball, theta, phi,
              w_theta, w_phi, altitude):
        """
        Predict trajectory of the ball.

        :param m: float, mass of golf ball, constant
        :param D: float, diameter of golf ball, constant
        :param rho_air: float, air density, constant
        :param C_D: float, drag force coefficient
        :param C_L: float, lift force coefficient
        :param v_ball: float, initial ball speed
        :param theta: float, angle between ball's launch speed and ground
        :param phi: float, azimuth of ball
        :param w_theta: float, angle between ball's rotation speed and ground
        :param w_phi: float, angel between ball's rotation speed's projection
                      n ground and x-axis
        :param altitude: float, constant for specific location
        :return: ball's track and coordinates for final position
        """
        # set initial conditions
        t0 = 0.0
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0
        v_x0 = v_ball * math.cos(theta) * math.cos(phi)
        v_y0 = v_ball * math.cos(theta) * math.sin(phi)
        v_z0 = v_ball * math.sin(theta)

        # angular unit vector in i
        w_unit_i = math.cos(w_theta) * math.cos(w_phi)
        # angular unit vector in j
        w_unit_j = math.cos(w_theta) * math.sin(w_phi)
        # angular unit vector in k
        w_unit_k = math.sin(w_theta)

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
        while tmp_z >= altitude or tmp_vz >= 0 and j + 1 < elements:
            # altitude use 0 here ??
            (show_t[j + 1], show_vx[j + 1], show_vy[j + 1], show_vz[j + 1],
             show_x[j + 1], show_y[j + 1], show_z[j + 1]) = \
                self.RK4(show_t[j], show_x[j], show_y[j], show_z[j],
                         show_vx[j], show_vy[j], show_vz[j],
                         w_unit_i, w_unit_j, w_unit_k,
                         C_D, C_L, rho_air, m, D)
            tmp_z = show_z[j + 1]
            tmp_vz = show_vz[j + 1]
            j = j + 1
            step = j

        # print_show_x = ("%5.3f" % show_x[step]).strip()
        # print_show_y = ("%5.3f" % show_y[step]).strip()
        # print_show_t = ("%5.3f" % show_t[step]).strip()

        # tmp_distance = math.sqrt(
        #     self.show_x[step] ** 2 + self.show_y[step] ** 2)
        # print_distance = ("%5.3f" % tmp_distance).strip()
        # print('    Fly time:           ', print_show_t, '(sec)')
        # print('    Fly distance:       ', print_distance, '(m)')
        # print('    Drop location in X: ', print_show_x, '(m)')
        # print('    Drop location in Y: ', print_show_y, '(m)')
        # print('--------------------------------------------')

        return (show_x[:step + 1], show_y[:step + 1],
                show_z[:step + 1], show_x[step], show_y[step],
                math.sqrt(show_x[step] ** 2 + show_y[step] ** 2))
        # get final value
        # 0,1,2 --> track
        # 3,4,5  --> final position

    def calculation(self, data):
        """
        Feature engineering function: calculate theta and phi for
        ball's rotation speed.

        :param data: data frame, ball's initial factors
        :return: data frame

        Note
        ----
        add column w_theta: float
        angle between ball's rotation speed and ground
        add column w_phi: float
        angel between ball's rotation speed's projection on ground and x-axis
        """
        ball = data[["ball_speed", "launch_angle", "azimuth",
                     "side_spin", "back_spin"]]
        ball.columns = ["launch_speed", "theta", "phi",
                        "side_spin", "back_spin"]
        ball.theta *= self.PI/180  # Degrees to Radians
        ball.phi *= self.PI/180  # Degrees to Radians

        ball.launch_speed *= 0.44704  # mile/h --> m/s
        ball.side_spin /= 60/(2*self.PI)  # rpm to rads per s
        ball.back_spin /= 60/(2*self.PI)  # rpm to rads per s

        w_theta = np.zeros(ball.shape[0])
        w_phi = np.zeros(ball.shape[0])
        for i in range(ball.shape[0]):
            row = ball.iloc[i]
            w_theta[i] = math.atan(row[3] * (math.cos(row[1])) / 
                                   math.sqrt((row[3]*math.sin(row[1])*math.cos(row[2]) -
                                              row[4]*math.sin(row[2]))**2 + (row[3]*math.sin(row[1])*math.sin(row[2]) +
                                                                             row[4]*math.cos(row[2]))**2))
            w_phi[i] = math.atan((-row[3]*math.sin(row[1])*math.sin(row[2]) - row[4]*math.cos(row[2])) / 
                                 (-row[3]*math.sin(row[1])*math.cos(row[2]) + row[4]*math.sin(row[2])))
        ball["w_theta"] = w_theta
        ball["w_phi"] = w_phi
        return ball

    def ML_pred(self, data, reg_carry, reg_tot_dist, reg_peak_height,
                reg_offline_ratio):
        """
        For a given shot, predicts where it will land up:

        :param data: Dataframe with following info on shot:
                      ["ball_speed", "launch_angle", "azimuth",
                      "side_spin", "back_spin"]
        :param reg_carry: linear regression model that predicts carry using
        above data
        :param reg_tot_dist: linear regression model that predicts total
        distance using above data
        :param reg_peak_height: linear regression model that predicts peak
        height using above data
        :param reg_offline_ratio: linear regression model that predicts
        offline ratio using above data
        :return: regression predictions for carry, total distance, peak
        height and offline
        """
        ball = data[["ball_speed", "launch_angle", "azimuth",
                     "side_spin", "back_spin"]]
        ball.columns = ["launch_speed", "theta", "phi",
                        "side_spin", "back_spin"]

        ball.theta *= self.PI/180  # Degrees to Radians
        ball.phi *= self.PI/180  # Degrees to Radians

        ball.launch_speed *= 0.44704  # mile/h --> m/s
        ball.side_spin /= 60/(2*self.PI)  # rpm to rads per s
        ball.back_spin /= 60/(2*self.PI)  # rpm to rads per s

        ball["cos_theta"] = ball.theta.apply(lambda x: math.cos(x))
        ball["sin_theta"] = ball.theta.apply(lambda x: math.sin(x))
        ball["tan_theta"] = ball.theta.apply(lambda x: math.tan(x))

        ball["cos_phi"] = ball.phi.apply(lambda x: math.cos(x))
        ball["sin_phi"] = ball.phi.apply(lambda x: math.sin(x))
        ball["tan_phi"] = ball.phi.apply(lambda x: math.tan(x))

        cols = list(ball.columns)
        for i, col in enumerate(cols):
            rest = cols[:i]+cols[i+1:]
            for r_col in rest:
                ball[col+"_"+r_col] = ball[col]*ball[r_col]

        return (reg_carry.predict(ball)[0], reg_tot_dist.predict(ball)[0],
                reg_peak_height.predict(ball)[0],
                reg_tot_dist.predict(ball)[0] *
                reg_offline_ratio.predict(ball)[0])

    def update_lines(self, num, data, line):
        """
        Assistant function: iteratively update trajectory for ball

        :param num: float, number of points in trajectory
        :param data: array, 3 by n array which records trajectory
        :param line: object
        :return: iterable trajectory
        """
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
        return line,

    def plot_animation(self, x_track, y_track, z_track, gif_name):
        """
        Make an animation plot and save it to local path with the specified
        file name.

        :param x_track: track (x axis)
        :param y_track: track (y axis)
        :param z_track: track (z axis)
        :param gif_name: target gif file name
        :return: None
        """
        track = pd.DataFrame({"x": x_track, "y": y_track, "z": z_track})
        track = np.array(track.T)
        track_small = track[:, np.arange(0, len(x_track), 50)]

        # Attaching 3D axis to the figure
        fig = plt.figure(figsize=(10, 4))
        fig.patch.set_alpha(1)
        ax = p3.Axes3D(fig)

        # starting point
        line = ax.plot(track_small[0, 0:1], track_small[1, 0:1],
                       track_small[2, 0:1], c="g")[0]

        # Setting the axes properties
        ax.set_xlim3d([max(150.0, 1.2*max(x_track)), 0.0])
        ax.set_xlabel('Carry')
        if y_track[-1] > 0:
            ax.set_ylim3d([-max(1.0, 1.2*max(y_track)),
                           max(1.0, 1.2*max(y_track))])
        else:
            ax.set_ylim3d([min(-1.0, 1.2*min(y_track)),
                           -min(-1.0, 1.2*min(y_track))])
#         ax.set_yticks([0])

        ax.set_ylabel('Offline')
        ax.set_zlim3d([0.0, max(1.0, 1.2*max(z_track))])
#         ax.set_zticks([0, 10, 20, 30])
        ax.set_zlabel('Height')
        # ax.set_title('3D Test')
        if y_track[-1] > 0:
            ax.view_init(azim=20, elev=10)
        else:
            ax.view_init(azim=-20, elev=10)

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, self.update_lines,
                                           track_small.shape[1] + 5,
                                           fargs=(track_small, line),
                                           interval=2, blit=True, repeat=True)

        # check if trajectory dir exists        
        if not os.path.isdir('./app/static/trajectory/'):
            os.makedirs('./app/static/trajectory/')

        # save gif file
        line_ani.save('./app/static/trajectory/' + str(gif_name),
                      writer='pillow', fps=30)

    def plot_plotly(self, x_track, y_track, z_track, gif_name):
        """
        Make a plotly plot and save it to local path with the specified
        file name. The gif file and html file have the same name,
        only differing in extension

        :param x_track: track (x axis)
        :param y_track: track (y axis)
        :param z_track: track (z axis)
        :param gif_name: corresponding gif file name
        :return: None
        """
        track = pd.DataFrame({"x": x_track, "y": y_track, "z": z_track})
        track = np.array(track.T)
        track_small = track[:, np.arange(0, len(x_track), 100)]

        x_max = 1.2*max(track_small[0])
        x_min = 1.2*min(track_small[0])

        y_max = max(5, 1.2*max(track_small[1]))
        y_min = min(-5, 1.2*min(track_small[1]))

        if max(track_small[1]) < 0:
            # if y_min > -10:
            #     y_min = -10
            y_max, y_min = y_min, y_max
#         else:
#             if y_max < 10:
#                 y_max = 10

        z_max = 1.2*max(track_small[2])
        z_min = 1.2*min(track_small[2])

        def get_frame(i):
            ball = go.Scatter3d(name='Ball Path', x=track_small[0],
                                y=track_small[1], z=track_small[2],
                                marker=dict(size=1,
                                            color=track_small[0] /
                                            max(track_small[0]),
                                            colorscale=[[0.0, 'rgb(0,0,0)'],
                                                        [1.0, 'rgb(0,0,0)']],),
                                line=dict(color='#1f77b4', width=1))

            ball_2 = go.Scatter3d(name='Ball Position', x=[track_small[0][i]],
                                  y=[track_small[1][i]], z=[track_small[2][i]],
                                  marker=dict(size=5,
                                              color=track_small[0] /
                                              max(track_small[0]),
                                              colorscale=[[0.0, 'rgb(0,255,0)'],
                                                          [1.0, 'rgb(0,0,0)']]),
                                  line=dict(color='#1f77b4', width=1))
            layout = dict(
                width=800,
                height=600,
                autosize=True,
                # title='Your Golf Shot',
                scene=dict(
                    xaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                        range=[x_min, x_max]
                    ),
                    yaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                        range=[y_max, y_min]
                    ),
                    zaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)',
                        range=[z_min, z_max]
                    ),
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=-0.3),
                        eye=dict(x=-1.8, y=-0.1, z=0.2)
                    ),
                    aspectratio=dict(x=2,
                                     y=2*abs(track_small[1][-1] /
                                             track_small[0][-1]),
                                     z=2*abs(track_small[2].max() /
                                             track_small[0][-1])),
                    aspectmode='manual'
                ),
            )

            data = [ball, ball_2]
            return data, layout

        data = [dict(data=get_frame(i)[0], layout=get_frame(i)[1])
                for i in range(track_small.shape[1])]

        fig = dict(data=data[0]["data"], layout=data[0]["layout"])
        fig['frames'] = list()
        for i in range(track_small.shape[1]):
            fig['frames'].append(dict(data=data[i]["data"]))
        fig['layout']['updatemenus'] = [
                {
                    'buttons': [
                        {
                            'args': [None, {'fromcurrent': True,
                                            'transition': {'duration': 0,
                                                           'easing': 'quadratic-in-out'}}],
                            'label': 'Play',
                            'method': 'animate'
                        },
                        {
                            'args': [[None],
                                     {'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate',
                                     'transition': {'duration': 0}}],
                            'label': 'Pause',
                            'method': 'animate'
                        },
                        {
                            'args': [None,
                                     {'fromcurrent': False,
                                      'transition': {'duration': 0,
                                                     'easing': 'quadratic-in-out'}}],
                            'label': 'Play From Start',
                            'method': 'animate'
                        }
                    ],
                    'direction': 'left',
                    'pad': {'r': 5, 't': 5},
                    'showactive': True,
                    'type': 'buttons',
                    'x': 0.1,
                    'xanchor': 'right',
                    'y': 0,
                    'yanchor': 'top'
                }
            ]
        fig['layout']['showlegend'] = False
        # fig['layout']['title']='Nice Shot!!'
        # fig['layout']['titlefont']=dict(size=28)

        plot_div = pyoff.plot(fig, auto_open=False,
                              filename='./app/static/trajectory/' + gif_name)

    # To disable auto open, use the commented version
    def traj(self, data, reg_carry, reg_tot_dist, reg_peak_height,
             reg_offline_ratio):
        """
        Calculate the trajectory of the ball, adjusted using ML

        :param data: shot factors
        :param reg_carry: linear regression model that predicts carry
        using above data
        :param reg_tot_dist: linear regression model that predicts total
        istance using above data
        :param reg_peak_height: linear regression model that predicts peak
        height using above data
        :param reg_offline_ratio: linear regression model that predicts
        offline ratio using above data
        :return: end position of the ball (x, y, z) and the flying track
        """
#         import pdb
#         pdb.set_trace()
        # prediction for 1 observation
        data = pd.DataFrame(data, index=[0])
        ball = self.calculation(data)
        row1 = ball.iloc[0]
        result = self.TRACK(0.04593, 0.04267, 1.2, 0.16, 0.1,
                            row1[0], row1[1], row1[2], row1[5], row1[6], 0)
        track = result[:3]

        coordinate_x = result[3]
        coordinate_y = result[4]
        coordinate_z = result[5]
        
        carry_pred, tot_dist_pred, \
            peak_height_pred, offline_pred = \
            self.ML_pred(data, reg_carry, reg_tot_dist,
                         reg_peak_height, reg_offline_ratio)
        
        offline_carry_pred = (offline_pred*carry_pred)/tot_dist_pred
        
        x_track, y_track, z_track = track
        
        x_sc = carry_pred/coordinate_x
        y_sc = offline_carry_pred/coordinate_y
        z_sc = peak_height_pred/z_track.max()
        
        coordinate_x *= x_sc
        coordinate_y *= y_sc
        coordinate_z *= z_sc
        
        x_track *= x_sc
        y_track *= y_sc
        z_track *= z_sc

        x_final = x_track[-1]
        y_final = y_track[-1]
        z_final = z_track[-1]

        bounce_scale = [1]*3
        bounce_scale[:2] = 0.5*(tot_dist_pred-x_final)/x_final, \
            0.5*(offline_pred-y_final)/y_final
        bounce_scale[2] = np.sqrt(bounce_scale[0]**2+bounce_scale[1]**2)

        x_bounce = []
        y_bounce = []
        z_bounce = []

        x_track1 = x_track
        y_track1 = y_track
        z_track1 = z_track

        for i in range(2):
            s_rate = int(0.1/max(bounce_scale))
            if s_rate < 1: s_rate = 1
            x_bounce.append((x_track*bounce_scale[0])
                            [np.arange(0, len(x_track), s_rate)])
            y_bounce.append((y_track*bounce_scale[1])
                            [np.arange(0, len(x_track), s_rate)])
            z_bounce.append((z_track*bounce_scale[2])
                            [np.arange(0, len(x_track), s_rate)])

            x_track2 = x_track1[-1]+x_bounce[i]
            y_track2 = y_track1[-1]+y_bounce[i]
            z_track2 = z_track1[-1]+z_bounce[i]

            x_track1 = np.array(list(x_track1) + list(x_track2))
            y_track1 = np.array(list(y_track1) + list(y_track2))
            z_track1 = np.array(list(z_track1) + list(z_track2))     

            bounce_scale = [sc*0.6 for sc in bounce_scale]

        s_rate2 = int(x_track1.shape[0]/5)
        x_track1 = list(x_track1)+list(np.linspace(x_track1[-1],
                                                   tot_dist_pred, s_rate2))
        y_track1 = list(y_track1)+list(np.linspace(y_track1[-1],
                                                   offline_pred, s_rate2))
        z_track1 = list(z_track1)+[0]*s_rate2
        
        n_track1 = len(y_track1)
        y_track1 = [y_track1[i]*np.exp((-1+(i/n_track1))/12)
                    for i in range(n_track1)]
        
#         print(x_track1, y_track1, z_track1)

        return coordinate_x, y_track1[-1], x_track1[-1], \
               (x_track1, y_track1, z_track1)
    
    def make_anime(self, track, gif_name):
        """
        Make animation of the trajectory.

        :param track:
        :param gif_name: target gif file name
        :return: None
        """
        x_track, y_track, z_track = track[0], track[1], track[2]
        self.plot_animation(x_track, y_track, z_track, gif_name)

    def make_plotly(self, track, gif_name):
        """
        Make plotly graph of the trajectory.

        :param track: (carry, offline, total distance)
        :param gif_name: corresponding gif file name
        :return: None
        """
        x_track, y_track, z_track = track[0], track[1], track[2]
        self.plot_plotly(x_track, y_track, z_track, gif_name)
