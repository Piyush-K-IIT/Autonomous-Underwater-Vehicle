import numpy as np
import math
from math import sin,cos,tan,sqrt
import matplotlib.pyplot as plt


# model parameter
m = 11.5;       # mass in kg
g = 9.8         # acceleration due to gravity
rho = 1021;     # density of water
W = 112.8;      # weight of model
B = 114.8;      # buoyant force on model

xG = 0; yG = 0; zG = 0.02;          # position of centre of gravity in body fixed frame
xB = 0; yB = 0; zB = 0;             # postion of centre of buoyancy in body fixed frame

IxG = 0.16; IyG = 0.16; IzG = 0.16  # principal MoI about CoG
Ixy = 0; Iyz = 0; Ixz = 0;          # product of inertias

Ix = IxG + m*(math.pow(yG,2) + math.pow(zG,2))        # principal MoI about BFF
Iy = IyG + m*(math.pow(xG,2) + math.pow(zG,2))
Iz = IzG + m*(math.pow(xG,2) + math.pow(yG,2))


# Rigid body dynamics
# constant matrix
Mrb = np.array([[m,       0,      0,     0,     m*zG,   -m*yG],
                [0,       m,      0,   -m*zG,    0,      m*xG],
                [0,       0,      m,    m*yG,  -m*xG,       0],
                [0,     -m*zG,   m*yG,   Ix,   -Ixy,     -Ixz],
                [m*zG,    0,    -m*xG, -Ixy,    Iy,      -Iyz],
                [-m*yG,  m*xG,   0,    -Ixz,   -Iyz,       Iz]])

# Diagonal elements of the rigid mass matrix
A11 = -5.5          # kg
A22 = -12.7         # kg
A33 = -14.57        # kg
A44 = -0.12         # kg m^2/rad
A55 = -0.12         # kg m^2/rad
A66 = -0.12         # kg m^2/rad


# Diagonal Matrix
D = np.array([A11, A22, A33, A44, A55, A66])

# added mass on the model
Mad = -np.array(np.diag(D))

# total mass
M = Mrb+Mad

# Thruster Model

# Thrust coefficent matrix [8X6]
# constant matrix

B_t =  np.array([[ 0.707, 0.707, -0.707, -0.707, 0, 0, 0, 0],
                 [-0.707, 0.707, -0.707,  0.707, 0, 0, 0, 0],
                 [   0,     0,     0,      0,   -1, 1, 1,-1],
                 [0.06,   -0.06,  0.06,  -0.06,-0.218,-0.218,0.218,0.218],
                 [0.06,    0.06, -0.06,  -0.06,0.120,-0.120,0.120,-0.120],
                 [-0.1888, 0.1888, 0.1888, -0.1888, 0, 0, 0, 0]])

# Control allocation
# constant matrix
B_t_plus = np.array(np.matmul((B_t.transpose()),np.linalg.inv(np.matmul(B_t,B_t.transpose()))))



# Determining transformation matrix of the model
def calculate_transformation_matrix(phi,theta,psi):
    
    # Transformation for position
    J1 = np.array([[cos(psi)*cos(theta),  -cos(phi)*sin(psi)+cos(psi)*sin(phi)*sin(theta),     sin(psi)*sin(phi)+cos(phi)*cos(psi)*sin(theta)],
                   [sin(psi)*cos(theta),   cos(psi)*cos(phi)+sin(psi)*sin(phi)*sin(theta),    -cos(psi)*sin(phi)+cos(phi)*sin(psi)*sin(theta)],
                   [-sin(theta),              sin(phi)*cos(theta),                                         cos(phi)*cos(theta)                                    ]])


    #Transformation  for orientation
    J2 = np.array([[ 1,   sin(phi)*tan(theta),  cos(phi)*tan(theta)],
                   [ 0,            cos(phi),            -sin(phi)  ],
                   [ 0,   sin(phi)/cos(theta),  cos(phi)/cos(theta)]])

    null = np.zeros((3,3))
    
    # Transformation matrix of the model
    rowOne = np.concatenate((J1,null),axis = 1)
    rowTwo = np.concatenate((null,J2),axis = 1)
    J = np.concatenate((rowOne,rowTwo), axis = 0)
    return J

# Centripetal and Coriolis matrix of rigid body
# v = [u,v,w,p,q,r]  in body frame

def calculate_c_matrix(u,v,w,p,q,r):
    
    nu = np.array([[u,v,w,p,q,r]]).transpose()
    
    crb = -np.array([[0,                0,               0,                 0,                        -m*w,                      0       ],
                    [0,                0,               0,                -m*w,                        0,                       0       ],
                    [0,                0,               0,                -m*v,                      -m*u,                      0       ],
                    [0,               m*w,            -m*v,                0,                        Iz*r,                    -Iy*q     ],
                    [-m*w,             0,             -m*u,              -Iz*r,                        0,                      Ix*p     ],
                    [m*v,             m*u,              0,                Iy*q,                       Ix*p,                     0       ]])

    # Centripetal and Coriolis matrix for added mass
    cad = np.array([[0,       0,         0,       0,        A33*w,    0     ],
                    [0,       0,         0,      -A33*w,    0,        A11*u ],
                    [0,       0,         0,      -A22*v,    A11*u,    0     ],
                    [0,      -A33*w,    A22*v,    0,       -A66*r,    A55*q ],
                    [A33*w,   0,        -A11*u,   A66*r,    0,        -A44*p],
                    [-A22*v,   A11*u,     0,     -A55*q,    A44*p,    0     ]],)

    # total Centripetal and Coriolis matrix
    cn = crb + cad
    cm = np.matmul(cn,nu)
    return cm

# damping matrix
# velocity in {b frame}
def calculate_damping_matrix(u,v,w,p,q,r):
    
    nu = np.array([[u,v,w,p,q,r]]).transpose()
    
    dn =  -np.array([[ -4.03+(-18.18)*abs(u),              0,                0,                        0,                 0,                       0         ],
                    [ 0,                     -6.22 + (-21.66)*abs(v),        0,                        0,                 0,                       0         ],
                    [ 0,                                   0,        -5.18+(-36.99)*abs(w),            0,                 0,                       0         ],
                    [ 0,                                   0,                0,            -0.07 + (-1.55)*abs(p),        0,                       0         ],
                    [ 0,                                   0,                0,                        0,         -0.07 + (-1.55)*abs(q),          0         ],
                    [ 0,                                   0,                0,                        0,                 0,             -0.07+(-1.55)*abs(r)]])

    dm = np.matmul(dn,nu)
    return dm

# restoring effects
def calculate_ge_matrix(phi,theta):
    ge = np.array([[(W-B)*sin(theta),
                  -(W-B)*cos(theta)*sin(phi),
                  -(W-B)*cos(theta)*cos(phi),
                   zG*W*cos(theta)*sin(phi),
                   zG*W*sin(theta),
                    0                      ]]).transpose()
    return ge

# calculate tau from tau_pid
def calculate_tau(tau_pid):
    
    # Thrust coefficients
    K1=K2=K3=K4=K5=K6=K7=K8=40
    
    # Thrust coefficient matrix
    K = np.array(np.diag(np.array([K1,K2,K3,K4,K5,K6,K7,K8])))  # 8X8 matrix
    
    # control signal
    # dependent to tau_pid
    u = np.matmul((np.matmul(np.linalg.inv(K),B_t_plus)),tau_pid)   #[8X8][8X6][6X1] = [8X1]
    # print("control input ", u)

    # thruster model
    tau = np.matmul(np.matmul(B_t,K),u)  # [6X8][8X8][8X1] = [6X1]
    
    # print("tau matrix in {b} frame ", tau)
    return tau

# Function to calculate acceleration
# acc in body frame
# tau in {b} frame
# velocity in {b} frame

def calculate_acc(tau,eta,vel):
    
    phi = eta[3][0]; theta = eta[4][0]
    
    u = vel[0][0]; v = vel[1][0]; w = vel[2][0]
    p = vel[3][0]; q = vel[4][0]; r = vel[5][0]
    
    cm = calculate_c_matrix(u,v,w,p,q,r)
    dm = calculate_damping_matrix(u,v,w,p,q,r)
    ge = calculate_ge_matrix(phi,theta)
    
    n1 = cm + dm + ge    # [6X1]
    Mn = tau - n1        # [6X1]
    
    # print("cm matrix",cm)
    # print("dm matrix",dm)
    # print("ge matrix",ge)
    
    # print("n1 matrix",n1)
    # print("Mn matrix", Mn)
    
    acc = np.matmul(Mn.transpose(),np.linalg.inv(M)).transpose()  # [1X6][6X6] = [1X6]
    # print("acceleration in {b} frame ", acc)

    return acc

# Function to calculate Velocity
# velocity in {b} frame
# acc in {b} frame
def calculate_vel(acc,dt,prev_vel):
    vel = prev_vel + acc * dt
    return vel

# Function to calculate etadot 
# velocity in NED frame
def calculate_etadot(J,vel_b):
    etadot = np.matmul(J,vel_b)
    return etadot

# Function to calculate Position
def calculate_pos(prev_eta,etadot,dt):
    eta = prev_eta + etadot * dt
    return eta

def draw_graph(pos_x,pos_y,pos_z,roll_arr,pitch_arr,yaw_arr,des_x,des_y,des_z,des_roll, des_pitch, des_yaw,time_step,T):
    
    #specify one size for all subplots
    fig, ax = plt.subplots(3, 2, figsize=(10,5))
    fig.tight_layout()
    
    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)
    pos_z = np.array(pos_z)
    
    roll_arr = np.array(roll_arr)
    pitch_arr = np.array(pitch_arr)
    yaw_arr = np.array(yaw_arr)
    
    time_step = np.array(time_step)
    
    #create subplots
    ax[0, 0].step([0,T], [0,des_x],color = 'blue', label = "Desired position", linestyle='dashed') 
    ax[0, 0].plot(time_step, pos_x, color='red', label = "Simulated Position")

    ax[1, 0].step([0,T], [0,des_y],color = 'blue', label = "Desired position", linestyle='dashed') 
    ax[1, 0].plot(time_step, pos_y, color='red', label = "Simulated Position")
    
    ax[2, 0].step([0,T], [10,des_z],color = 'blue', label = "Desired position", linestyle='dashed') 
    ax[2, 0].plot(time_step, pos_z, color='red',label = "Simulated Position")
    
    ax[0, 1].step([0,T], [0,des_roll],color = 'blue', label = "Desired position", linestyle='dashed') 
    ax[0, 1].plot(time_step, roll_arr, color='red', label = "Simulated orientation")
    
    ax[1, 1].step([0,T], [0,des_pitch],color = 'blue', label = "Desired position",linestyle='dashed') 
    ax[1, 1].plot(time_step, pitch_arr, color='red', label = "Simulated orientation")
    
    ax[2, 1].step([0,T], [0,des_yaw],color = 'blue',label = "Desired position", linestyle='dashed') 
    ax[2, 1].plot(time_step, yaw_arr, color='red', label = "Simulated orientation")
    
    #define subplot titles
    ax[0, 0].set_title('Surge Motion')
    ax[1, 0].set_title('Sway Motion')
    ax[2, 0].set_title('Heave Motion')
    ax[0, 1].set_title('Roll Motion')
    ax[1, 1].set_title('Pitch Motion')
    ax[2, 1].set_title('yaw Motion')
    
    # set x axis lebel
    ax[0, 0].set_xlabel('Time(s)')
    ax[1, 0].set_xlabel('Time(s)')
    ax[2, 0].set_xlabel('Time(s)')
    ax[0, 1].set_xlabel('Time(s)')
    ax[1, 1].set_xlabel('Time(s)')
    ax[2, 1].set_xlabel('Time(s)')
    
    # set y axis lebel
    ax[0, 0].set_ylabel('x position(m)')
    ax[1, 0].set_ylabel('y position(m)')
    ax[2, 0].set_ylabel('z position(m)')
    ax[0, 1].set_ylabel('roll(rad/sec)')
    ax[1, 1].set_ylabel('pitch(rad/sec)')
    ax[2, 1].set_ylabel('yaw(rad/sec)')
    
    ax[0,0].legend()
    ax[1,0].legend()
    ax[2,0].legend()
    ax[0,1].legend()
    ax[1,1].legend()
    ax[2,1].legend()
    
    
    plt.pause(1000)
    
# main function
def main():
    # These are initial position and orientation of the vehicle with
    # initial velocity and acceleration
    
    initial_position = np.array([[0,0,10,0,0,0]]).transpose()
    initial_velocity = np.array([[0,0,0,0,0,0]]).transpose()
    initial_acceleration = np.array([[0,0,0,0,0,0]]).transpose()
    
    des_position = np.array([[0,0,10,1,0,0]]).transpose()
    
    global eta,vel,u
    eta = initial_position  # Generalized position and orientation in {n} frame
    vel = initial_velocity         # Generalized linear and angular velocity in {b} frame
    u = np.array([[0,0,0,0,0,0,0,0]]).transpose() # [8X1]
    
    etadot = np.array([[0,0,0,0,0,0]]).transpose() # velocity in {n} frame
    nudot = initial_acceleration # acceleration in {b} frame
    
    eta_des = des_position
    
    # initializing varibles for conrtol system
    prev_error = np.array([[0,0,0,0,0,0]]).transpose()   # 6X1 
    integral = np.array([[0,0,0,0,0,0]]).transpose()    # 6X1
    tau_pid = np.array([[0,0,0,0,0,0]]).transpose()  # 6X1
    derivative = np.array([[0,0,0,0,0,0]]).transpose()  # 6X1
    tau = np.array([[0,0,0,0,0,0]]).transpose()  # 6X1
    
    current_vel = np.array([[0,0,0,0,0,0]]).transpose() 

    pos_x = []
    pos_y = []
    pos_z = []
    
    roll_arr = []
    pitch_arr = []
    yaw_arr = []
    
    time_step = []
    
    dt = 0.01
    t = 0.0
    T = 100.0
    while True:
        while t <= T:
            
            # print("initial position ",eta)
            # print("initial velocity",vel)
            
            # transformation matrix
            J = calculate_transformation_matrix(eta[3][0],eta[4][0],eta[5][0])
            
            # calculation of control torque
            # control gain
            
            kp = np.diag(np.array([3,3,3,4,4,2]))                 # 6X6
            ki = np.diag(np.array([0.2,0.2,0.2,0.3,0.3,0.1]))     # 6x6
            kd = np.diag(np.array([2.5,2.5,0.5,0.5,1,0.5]))       # 6X6
            
            # error in {n} frame
            error = eta_des - eta   # 6X1
            
            # error in {b} frame
            eb = np.matmul(np.linalg.inv(J),error)   # 6X1

            derivative = (eb - prev_error)/dt    # 6X1

            integral = integral + eb*dt  # 6X1
            
            tau_pid = np.matmul(kp,eb) + np.matmul(ki,integral) + np.matmul(kd,derivative)  # [6X6][6X1] 
    
            prev_error = eb
        
            # torque calculation in {b} frame
            tau = calculate_tau(tau_pid)

            # body acceleration calculation
            nudot = calculate_acc(tau,eta,vel)
            
            # body velocity calculation
            vel = calculate_vel(nudot,dt,vel)

            # velocity calculation in {n} frame
            etadot = calculate_etadot(J,vel)
            
            # calculate position in {n} frame
            eta = calculate_pos(eta,etadot,dt)

            roll = eta[3][0]           # phi
            pitch = eta[4][0]          # theta
            yaw = eta[5][0]            # psi
            
            x_pos = eta[0][0]         # x
            y_pos = eta[1][0]         # y
            z_pos = eta[2][0]         # z
            
            # print("x_pos ",x_pos)
            # print("y_pos ",y_pos)
            # print("z_pos" ,z_pos)
            
            pos_x.append(x_pos)
            pos_y.append(y_pos)
            pos_z.append(z_pos)
            
            roll_arr.append(roll)
            pitch_arr.append(pitch)
            yaw_arr.append(yaw)
            
            des_x = des_position[0][0]
            des_y = des_position[1][0]
            des_z = des_position[2][0]
            des_roll = des_position[3][0]
            des_pitch = des_position[4][0]
            des_yaw = des_position[5][0]
            
            time_step.append(t)
                        
            t += dt
        
        draw_graph(pos_x,pos_y,pos_z,roll_arr,pitch_arr,yaw_arr,des_x,des_y,des_z,des_roll,des_pitch,des_yaw,time_step,T)
        break

if __name__ == "__main__":
    main()
