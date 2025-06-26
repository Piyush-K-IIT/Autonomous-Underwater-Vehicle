import numpy as np
import math
from math import sin,cos,sqrt,tan
import matplotlib.pyplot as plt
from TrajectoryGenerator import TrajectoryGenerator
from AUV import AUV

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

# calculation of c_matrix
def calculate_c_matrix(relative_vel):
    
    u_water = relative_vel[0][0]
    v_water = relative_vel[1][0]
    w_water = relative_vel[2][0]
    p_water = relative_vel[3][0]
    q_water = relative_vel[4][0]
    r_water = relative_vel[5][0]
    
    crb = -np.array([[0,                0,               0,                 0,                        -m*w_water,                      0       ],
                    [0,                0,               0,                -m*w_water,                        0,                       0       ],
                    [0,                0,               0,                -m*v_water,                      -m*u_water,                      0       ],
                    [0,               m*w_water,            -m*v_water,                0,                        Iz*r_water,                    -Iy*q_water     ],
                    [-m*w_water,             0,             -m*u_water,              -Iz*r_water,                        0,                      Ix*p_water     ],
                    [m*v_water,             m*u_water,              0,                Iy*q_water,                       Ix*p_water,                     0       ]])

    # Centripetal and Coriolis matrix for added mass
    cad = np.array([[0,       0,         0,       0,        A33*w_water,    0     ],
                    [0,       0,         0,      -A33*w_water,    0,        A11*u_water ],
                    [0,       0,         0,      -A22*v_water,    A11*u_water,    0     ],
                    [0,      -A33*w_water,    A22*v_water,    0,       -A66*r_water,    A55*q_water ],
                    [A33*w_water,   0,        -A11*u_water,   A66*r_water,    0,        -A44*p_water],
                    [-A22*v_water,   A11*u_water,     0,     -A55*q_water,    A44*p_water,    0     ]],)

    # total Centripetal and Coriolis matrix
    cn = crb + cad
    cm = np.matmul(cn,relative_vel)
    return cm

# damping matrix
# velocity in {b frame}
def calculate_damping_matrix(relative_vel):
    
    u_water = relative_vel[0][0]
    v_water = relative_vel[1][0]
    w_water = relative_vel[2][0]
    p_water = relative_vel[3][0]
    q_water = relative_vel[4][0]
    r_water = relative_vel[5][0]
    
    dn =  -np.array([[ -4.03+(-18.18)*abs(u_water),              0,                0,                        0,                 0,                       0         ],
                    [ 0,                     -6.22 + (-21.66)*abs(v_water),        0,                        0,                 0,                       0         ],
                    [ 0,                                   0,        -5.18+(-36.99)*abs(w_water),            0,                 0,                       0         ],
                    [ 0,                                   0,                0,            -0.07 + (-1.55)*abs(p_water),        0,                       0         ],
                    [ 0,                                   0,                0,                        0,         -0.07 + (-1.55)*abs(q_water),          0         ],
                    [ 0,                                   0,                0,                        0,                 0,             -0.07+(-1.55)*abs(r_water)]])

    dm = np.matmul(dn,relative_vel)
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
    print("control input ", u)

    # thruster model
    tau = np.matmul(np.matmul(B_t,K),u)  # [6X8][8X8][8X1] = [6X1]
    
    print("tau matrix in {b} frame ", tau)
    return tau

# Function to calculate relative acceleration
# acc in body frame
# tau in {b} frame
# velocity in {b} frame

def calculate_acc(tau,eta,relative_vel):
    
    phi = eta[3][0]; theta = eta[4][0]; psi = eta[5][0]
    
    cm = calculate_c_matrix(relative_vel)
    dm = calculate_damping_matrix(relative_vel)
    ge = calculate_ge_matrix(phi,theta)
    
    n1 = cm + dm + ge    # [6X1]
    Mn = tau - n1        # [6X1]
    
#     print("cm matrix",cm)
#     print("dm matrix",dm)
#     print("ge matrix",ge)
    
#     print("n1 matrix",n1)
#     print("Mn matrix", Mn)
    
    #print("acceleration in {b} frame ", acc)
    acc = np.matmul(Mn.transpose(),np.linalg.inv(M)).transpose()  # [1X6][6X6] = [1X6]


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
    
# main program

show_animation = True

# Simulation parameters

T = 30

def quad_sim(x_c, y_c, z_c):
    
    # These are initial position and orientation of the vehicle with
    # initial velocity and acceleration
    
    initial_position = np.array([[-5,-5,5,0,0,0]]).transpose()
    initial_velocity = np.array([[0,0,0,0,0,0]]).transpose()
    initial_acceleration = np.array([[0,0,0,0,0,0]]).transpose()
    
    #des_position = np.array([[1,1,10,1,1,1]]).transpose()
    
    global eta,vel,u
    eta = initial_position  # Generalized position and orientation in {n} frame
    vel = initial_velocity         # Generalized linear and angular velocity in {b} frame
    u = np.array([[0,0,0,0,0,0,0,0]]).transpose() # [8X1]
    
    etadot = np.array([[0,0,0,0,0,0]]).transpose() # velocity in {n} frame
    nudot = initial_acceleration # acceleration in {b} frame
    
    #eta_des = des_position
    
    # initializing varibles for conrtol system
    prev_error = np.array([[0,0,0,0,0,0]]).transpose()   # 6X1 
    integral = np.array([[0,0,0,0,0,0]]).transpose()    # 6X1
    tau_pid = np.array([[0,0,0,0,0,0]]).transpose()  # 6X1
    derivative = np.array([[0,0,0,0,0,0]]).transpose()  # 6X1
    tau = np.array([[0,0,0,0,0,0]]).transpose()  # 6X1
    
    
    
    # interaction of ocean currents
    
    #sideslip angle
    beta_c = math.pi/3
    
    #angle of attack
    gama_c = math.pi/6
    
    # velocity of ocean current in m/s
    Vc = 0.2
    
    # current velocity in {n} frame
    current_vel_n = np.array([[Vc*cos(gama_c)*cos(beta_c),Vc*sin(beta_c),Vc*sin(gama_c)*cos(beta_c),0,0,0]]).transpose()
    
    # current velocity in {b} frame
    current_vel_b = np.array([[0,0,0,0,0,0]]).transpose()
    
    
    # Relative velocity in {V} frame or {B} frame
    relative_vel = vel - current_vel_b
    
    # current velocity in body frame
    linear_current_vel_b = current_vel_b[:3, :1]
    

    pos_x = []
    pos_y = []
    pos_z = []
    
    roll_arr = []
    pitch_arr = []
    yaw_arr = []
    
    time_step = []
    
    des_roll_arr = []
    des_pitch_arr = []
    des_yaw_arr = []
    
    
    
    des_x = []
    des_y = []
    des_z = []
    
    
    
    dt = 0.1
    t = 0.0

    x = eta[0][0];  y = eta[1][0]; z = eta[2][0]; phi = eta[3][0]; theta = eta[4][0]; psi = eta[5][0] 
     
    auv = AUV(x=x, y=y, z=z, roll=phi,
                  pitch=theta, yaw=psi, size=2, show_animation=show_animation)

    i = 0
    n_run = 5  
    irun = 0
    
    while True:
        while t <= T:
            des_x_pos = calculate_position(x_c[i], t)
            des_y_pos = calculate_position(y_c[i], t)
            des_z_pos = calculate_position(z_c[i], t)
            
            des_roll = 0
            des_pitch = 0
            des_yaw = 0
    
            eta_des = np.array([des_x_pos,des_y_pos,des_z_pos,[des_roll],[des_pitch],[des_yaw]])
            
            # transformation matrix
            J = calculate_transformation_matrix(eta[3][0],eta[4][0],eta[5][0])
            
            J1 = J[:3, :3]
    
            linear_current_vel_b = np.matmul(J1.transpose(),current_vel_n[:3, :1])
            
            # relative velocity in body frame
            relative_vel = vel - np.concatenate((linear_current_vel_b, np.zeros((3,1))), axis=0)
            
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
            nudot = calculate_acc(tau,eta,relative_vel)
            
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
            
            pos_x.append(x_pos)
            pos_y.append(y_pos)
            pos_z.append(z_pos)
            
            roll_arr.append(roll)
            pitch_arr.append(pitch)
            yaw_arr.append(yaw)
            
            des_x.append(des_x_pos)
            des_y.append(des_y_pos)
            des_z.append(des_z_pos)
            
            des_roll_arr.append(des_roll)
            des_pitch_arr.append(des_pitch)
            des_yaw_arr.append(des_yaw) 
            
            time_step.append(t)
                        

            
            print("Waypoints ", waypoints[1][2])
            j = 0
            for j in range(len(waypoints)):
                if(x_pos > waypoints[j][0] and x_pos < waypoints[(j+1)%len(waypoints)][0]):
                    currentWaypoint = waypoints[j]
                else:
                    currentWaypoint = waypoints[(j+1)%len(waypoints)]
                
                 
                # if(np.trunc(x_pos) == 5):
                #     des_yaw = 1.57
                #     eta[0][0] = 5.0
                
            auv.update_pose(x_pos, y_pos, z_pos, roll, pitch, yaw)
            t += dt
        
        t = 0.0
        i = (i + 1) % len(waypoints)
        irun += 1
        if irun >= n_run:
            plt.pause(10000)
            break

    print("Done")
    
    draw_graph(pos_x,pos_y,pos_z,roll_arr,pitch_arr,yaw_arr,des_x,des_y,des_z,des_roll,des_pitch,des_yaw,time_step,T)
        

def calculate_position(c, t):
    """
    Calculates a position given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the position

    Returns
        Position
    """
    return c[0] * t**5 + c[1] * t**4 + c[2] * t**3 + c[3] * t**2 + c[4] * t + c[5]


def calculate_velocity(c, t):
    """
    Calculates a velocity given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the velocity

    Returns
        Velocity
    """
    return 5 * c[0] * t**4 + 4 * c[1] * t**3 + 3 * c[2] * t**2 + 2 * c[3] * t + c[4]


def calculate_acceleration(c, t):
    """
    Calculates an acceleration given a set of quintic coefficients and a time.

    Args
        c: List of coefficients generated by a quintic polynomial
            trajectory generator.
        t: Time at which to calculate the acceleration

    Returns
        Acceleration
    """
    return 20 * c[0] * t**3 + 12 * c[1] * t**2 + 6 * c[2] * t + 2 * c[3]


def main():
    """
    Calculates the x, y, z coefficients for the four segments
    of the trajectory
    """
    
    #List of size four with four array
    
    x_coeffs = [[], [], [], [], [], []]
    y_coeffs = [[], [], [], [], [], []]
    z_coeffs = [[], [], [], [], [], []]
    
   
    #waypoints = [[-5, -5, 5], [5, -5, 5], [5, 5, 5], [-5, 5, 5]]
    # lawn mower
    global waypoints
    global currentWaypoint
    global startingWaypoint
    global endingWaypoint
    
    waypoints = [[-5,-5,5],[5,-5,5],[5,0,5],[-5,0,5],[-5,5,5],[5,5,5]]
    
    startingWaypoint = waypoints[0]

    endingWaypoint = waypoints[len(waypoints)-1]
    
    for i in range(len(waypoints)):  # i = 0,1,2,3,4,5
        # crating an object of trajectoryGenerator
        # waypoint[0] = [-5,-5,0] ==> starting posiition of the auv
        # waypoint[(i+1)%6] = waypoint[1] = [5,-5,5] ==> next waypoint
        # T = time
        
        if( i < len(waypoints)-1):   # i =0,1,2,3,4
            
            # iteration i = 0 ==> 0 --> 1 # path 1
            #           i = 1 ==> 1 --> 2 # path 2
            #           i = 2 ==> 2 --> 3 # path 3
            #           i = 3 ==> 3 --> 4 # path 4
            #           i = 4 ==> 4 --> 5 # path 5
            
            traj = TrajectoryGenerator(waypoints[i], waypoints[(i + 1) % len(waypoints)], T)
            traj.solve()
            x_coeffs[i] = traj.x_c   # after first iteration x_coeffs[0] = [x_c]
            y_coeffs[i] = traj.y_c
            z_coeffs[i] = traj.z_c
                        
    quad_sim(x_coeffs, y_coeffs, z_coeffs)



if __name__ == "__main__":
    main()
