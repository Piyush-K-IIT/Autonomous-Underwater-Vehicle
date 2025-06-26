import numpy as np
import math
from math import sin,cos
import matplotlib.pyplot as plt

class AUV():
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.25, show_animation=True):
        
        
        self.p1 = np.array([0,0,0,1]).T
        self.p2 = np.array([3,0,0,1]).T
        self.p3 = np.array([3,2,0,1]).T
        self.p4 = np.array([0,2,0,1]).T
        
        self.p1_dash = np.array([0,0,2,1]).T
        self.p2_dash = np.array([3,0,2,1]).T
        self.p3_dash = np.array([3,2,2,1]).T
        self.p4_dash = np.array([0,2,2,1]).T

        self.x_data = []
        self.y_data = []
        self.z_data = []
        
        self.show_animation = show_animation

        if self.show_animation:
            plt.ion()
            fig = plt.figure()
            # for stopping simulation with the esc key.
            fig.canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            self.ax = fig.add_subplot(111, projection='3d')

        self.update_pose(x, y, z, roll, pitch, yaw)

    def update_pose(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        
        self.x_data.append(x+1)
        self.y_data.append(y+1)
        self.z_data.append(z+1)

        if self.show_animation:
            self.my_plot()

    def transformation_matrix(self):       # 3X4 matrix
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        
        # rotation matix
        return np.array(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(roll), z]
             ])

    def my_plot(self):  # pragma: no cover
        
        T = self.transformation_matrix()
        print("Shape of T",T.shape)
        
        p1_t = np.matmul(T, self.p1)     # [3X4][4X1] = 3X1
        p2_t = np.matmul(T, self.p2)
        p3_t = np.matmul(T, self.p3)
        p4_t = np.matmul(T, self.p4)

        print ("Shape of p1_t",p1_t.shape)
        
        p1_t_dash = np.matmul(T, self.p1_dash)
        p2_t_dash = np.matmul(T, self.p2_dash)
        p3_t_dash = np.matmul(T, self.p3_dash)
        p4_t_dash = np.matmul(T, self.p4_dash)

        plt.cla()

        self.ax.plot([p1_t[0], p2_t[0], p3_t[0], p4_t[0]],
                     [p1_t[1], p2_t[1], p3_t[1], p4_t[1]],
                     [p1_t[2], p2_t[2], p3_t[2], p4_t[2]],'k.')
        
        self.ax.plot([p1_t_dash[0], p2_t_dash[0], p3_t_dash[0], p4_t_dash[0]],
                     [p1_t_dash[1], p2_t_dash[1], p3_t_dash[1], p4_t_dash[1]],
                     [p1_t_dash[2], p2_t_dash[2], p3_t_dash[2], p4_t_dash[2]],'k.')
        
        # Top Face
        self.ax.plot([p1_t[0], p2_t[0]], 
                     [p1_t[1], p2_t[1]],
                     [p1_t[2], p2_t[2]], 'r-')
        
        self.ax.plot([p1_t[0], p4_t[0]], 
                     [p1_t[1], p4_t[1]],
                     [p1_t[2], p4_t[2]], 'r-')
        
        self.ax.plot([p2_t[0], p3_t[0]], 
                     [p2_t[1], p3_t[1]],
                     [p2_t[2], p3_t[2]], 'r-')
        
        self.ax.plot([p3_t[0], p4_t[0]], 
                     [p3_t[1], p4_t[1]],
                     [p3_t[2], p4_t[2]], 'r-')
        
        # Bottom Face
        
        self.ax.plot([p1_t_dash[0], p2_t_dash[0]], 
                     [p1_t_dash[1], p2_t_dash[1]],
                     [p1_t_dash[2], p2_t_dash[2]], 'r-')
        
        self.ax.plot([p1_t_dash[0], p4_t_dash[0]], 
                     [p1_t_dash[1], p4_t_dash[1]],
                     [p1_t_dash[2], p4_t_dash[2]], 'r-')
        
        self.ax.plot([p2_t_dash[0], p3_t_dash[0]], 
                     [p2_t_dash[1], p3_t_dash[1]],
                     [p2_t_dash[2], p3_t_dash[2]], 'r-')
        
        self.ax.plot([p3_t_dash[0], p4_t_dash[0]], 
                     [p3_t_dash[1], p4_t_dash[1]],
                     [p3_t_dash[2], p4_t_dash[2]], 'r-')

        #  connecting both top and bottom surface
        
        self.ax.plot([p1_t[0], p1_t_dash[0]], 
                     [p1_t[1], p1_t_dash[1]],
                     [p1_t[2], p1_t_dash[2]], 'r-')
        
        self.ax.plot([p2_t[0], p2_t_dash[0]], 
                     [p2_t[1], p2_t_dash[1]],
                     [p2_t[2], p2_t_dash[2]], 'r-')
        
        self.ax.plot([p3_t[0], p3_t_dash[0]], 
                     [p3_t[1], p3_t_dash[1]],
                     [p3_t[2], p3_t_dash[2]], 'r-')
        
        self.ax.plot([p4_t[0], p4_t_dash[0]], 
                     [p4_t[1], p4_t_dash[1]],
                     [p4_t[2], p4_t_dash[2]], 'r-')
        
        
        self.ax.plot(self.x_data, self.y_data, self.z_data, 'r:')
        
        waypoints = [[-5,-5,5],[5,-5,5],[5,0,5],[-5,0,5],[-5,5,5],[5,5,5]]
        
        self.ax.scatter([-5+1,5+1],[-5+1,-5+1],[5+1,5+1],c='red',s = 100)
        self.ax.scatter([5+1,5+1],[-5+1,0+1],[5+1,5+1],c='red', s=100)
        self.ax.scatter([5+1,-5+1],[0+1,0+1],[5+1,5+1],c='red', s=100)
        self.ax.scatter([-5+1,-5+1],[0+1,5+1],[5+1,5+1],c='red', s=100)
        self.ax.scatter([-5+1,5+1],[5+1,5+1],[5+1,5+1],c='red', s=100)
        
        self.ax.plot([-5+1,5+1],[-5+1,-5+1],[5+1,5+1], color='blue', alpha = 0.2)
        self.ax.plot([5+1,5+1],[-5+1,0+1],[5+1,5+1], color='blue' , alpha = 0.2)
        self.ax.plot([5+1,-5+1],[0+1,0+1],[5+1,5+1], color='blue', alpha = 0.2)
        self.ax.plot([-5+1,-5+1],[0+1,5+1],[5+1,5+1], color='blue', alpha = 0.2)
        self.ax.plot([-5+1,5+1],[5+1,5+1],[5+1,5+1], color='blue', alpha = 0.2)

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        self.ax.set_zlim(0, 10)

        plt.pause(0.001)
