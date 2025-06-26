import numpy as np
import math
from math import sin,cos,sqrt
import matplotlib.pyplot as plt
from TrajectoryGenerator import TrajectoryGenerator
from AUV import AUV


# model parameter
m = 57.1;       # mass in kg
g = 9.8         # acceleration due to gravity
rho = 1021;     # density of water
W = 560;        # weight of model
B = 560;        # buoyant force on model

a = 0.326; b = 0.28; L = 0.755      # dimensions of robot(assumed to be a pentagonal prism)
xG = 0; yG = 0; zG = 0;             # position of centre of gravity in body fixed frame
xB = 0; yB = 0; zB = 0;             # postion of centre of buoyancy in body fixed frame

IxG = 1.1; IyG = 3.65; IzG = 4.1    # principal MoI about CoG
Ixy = 0; Iyz = 0; Ixz = 0;          # product of inertias

Ix = IxG + m*(yG^2 + zG^2)         # principal MoI about BFF
Iy = IyG + m*(xG^2 + zG^2)
Iz = IzG + m*(xG^2 + yG^2)

# Displaying principal MOI
print("Ix = ",Ix)
print("Iy = ",Iy)
print("Iz = ",Iz)

#_____________________________________________________________________________________________________________________________________________
# mass matrix
# mass of rigid body
Mrb = np.array([[m,       0,      0,     0,     m*zG,   -m*yG],
                [0,       m,      0,   -m*zG,    0,      m*xG],
                [0,       0,      m,    m*yG,  -m*xG,       0],
                [0,     -m*zG,   m*yG,   Ix,   -Ixy,     -Ixz],
                [m*zG,    0,    -m*xG, -Ixy,    Iy,      -Iyz],
                [-m*yG,  m*xG,   0,    -Ixz,   -Iyz,       Iz]])

print("Mrb Matrix = \n",Mrb)
print("\n Size of the Rigid Body Mass Matrix = ",Mrb.shape)

# Diagonal elements of the rigid mass matrix
A11 = 0.1*m
A22 = 1.25*rho*L*math.pow(b,2)
A33 = 1.175*rho*L*math.pow(a,2)
A44 = rho*math.pow(L,2)*(0.086*math.pow(a,3)+0.0875*math.pow(b,3))
A55 = rho*math.pow(a,2)*(0.098*math.pow(L,3)+0.115*math.pow(b,3))
A66 = rho*math.pow(b,2)*(0.12*math.pow(a,3)+0.104*math.pow(L,3))

print("\nDiagonal elements \n",A11,A22,A33,A44,A55,A66)

# Diagonal Matrix
D = np.array([A11, A22, A33, A44, A55, A66])
print("\nSize of Diagonal Matrix =",D.shape,"\n")

# added mass on the model
Mad = np.array(np.diag(D))



print("\nAdded Mass Matrix =\n",Mad)
print("\nSize of Added Mass Matrix =",Mad.shape)

# total mass
M = Mrb+Mad
print("\nTotal Mass Matrix= \n",M)
print("\nTotal Mass Matrix Size = ", M.shape)

#________________________________________________________________________________________________________________________________________
# Thruster Model

av = a/2
ah = a/2
lh = L/2
lv = lh/2   # 

# b matrix for thruster force .T
# thrust configuration matrix

B_t =  np.array([[   -1/sqrt(2),      -1/sqrt(2),        1/sqrt(2),        1/sqrt(2),      0,    0,   0,   0],
                 [   -1/sqrt(2),       1/sqrt(2),        1/sqrt(2),       -1/sqrt(2),      0,    0,   0,   0],
                 [        0,               0,                0,                0,         -1,   -1,  -1,  -1],
                 [        0,               0,                0,                0,          av,  -av, -av, av],
                 [        0,               0,                0,                0,          lv,   lv, -lv,-lv],
                 [-(lh+ah)/sqrt(2),  -(lh+ah)/sqrt(2), -(lh+ah)/sqrt(2), -(lh+ah)/sqrt(2),  0,     0,  0,   0]])

print("\nB-matrix",B_t)
print("\nshape of B-matrix",B_t.shape)

# Control allocation
B_t_plus = np.array(np.matmul((B_t.T),np.linalg.inv((np.matmul(B_t,B_t.T)))))

print("\nB_t_plus Matrix = ",B_t_plus)
print("\nShape of B_t_plus matrix",B_t_plus.shape)


#______________________________________________________________________________________________________________________________________________

# Determining transformation matrix of the model
def calculate_transformation_matrix(phi,theta,psi):
    
    # Transformation for position
    J1 = np.array([[np.cos(psi)*np.cos(theta),  -np.cos(phi)*np.sin(psi)+np.cos(psi)*np.sin(phi)*np.sin(theta),     np.sin(psi)*np.sin(phi)+np.cos(phi)*np.cos(psi)*np.sin(theta)],
                   [np.sin(psi)*np.cos(theta),   np.cos(psi)*np.cos(theta)+np.sin(psi)*np.sin(phi)*np.sin(theta),  -np.cos(psi)*np.sin(phi)+np.cos(phi)*np.sin(psi)*np.sin(theta)],
                   [-np.sin(theta),              np.sin(phi)*np.cos(theta),                                         np.cos(phi)*np.cos(theta)                                    ]])


    #Transformation  for orientation
    J2 = np.array([[ 1,   np.sin(phi)*np.tan(theta),  np.cos(phi)*np.tan(theta)],
                   [ 0,            np.cos(phi),            -np.sin(phi)        ],
                   [ 0,   np.sin(phi)/np.cos(theta),  np.cos(phi)/np.cos(theta)]])

    null = np.zeros((3,3))

    # print("J1 Shape",J1.shape)
    
    # Transformation matrix of the model
    rowOne = np.concatenate((J1,null),axis = 1)
    rowTwo = np.concatenate((null,J2),axis = 1)

    J = np.concatenate((rowOne,rowTwo), axis = 0)

    print("Transformation Matrix(J) =\n\n", J,"\n")
    print("Size of Transformation Matrix =", J.shape)
    
    return J
#_______________________________________________________________________________________________________________________________________________________

# Centripetal and Coriolis matrix of rigid body
def calculate_c_matrix(u,v,w,p,q,r):
    
    nu = np.array([[u,v,w,p,q,r]]).transpose()
    
    crb = np.array([[0,                0,               0,                 m*(yG*q+zG*r),        -m*(xG*q-w),          -m*(xG*r+v)],
                    [0,                0,               0,                -m*(yG*p+w),            m*(zG*r+xG*p),       -m*(yG*r-u)],
                    [0,                0,               0,                -m*(zG*p-v),           -m*(xG*q+u),           m*(zG*p+yG*q)],
                    [-m*(yG*q+zG*r),   m*(yG*p+w),      m*(zG*p-v),        0,                    -q*Iyz-p*Ixz+r*Iz,     r*Iyz+p*Ixy-q*Iy],
                    [m*(xG*q-w),      -m*(zG*r+xG*p),   m*(xG*q+u),        q*Iyz+p*Ixz-r*Iz,      0,                   -r*Ixz-q*Ixy+p*Ix],
                    [m*(xG*r+v),       m*(yG*r-u),     -m*(zG*p+yG*q),    -r*Iyz-p*Ixy+q*Iy,      r*Ixz+q*Ixy-p*Ix,     0]])

    print("Crb Matrix = \n", crb)
    print("size of crb Matrix: ",crb.shape)

    # Centripetal and Coriolis matrix for added mass

    cad = np.array([[0,       0,         0,       0,        A33*w,   -A22*v ],
                    [0,       0,         0,      -A33*w,    0,        A11*u ],
                    [0,       0,         0,       A22*v,    A11*u,    0     ],
                    [0,       A33*w,    -A22*v,   0,        A66*r,  -A55*q  ],
                    [-A33*w,   0,         A11*u,  -A66*r,    0,        A44*p],
                    [A22*v,   A11*u,     0,       A55*q,   -A44*p,    0     ]],)

    # total Centripetal and Coriolis matrix
    cn = crb + cad
    cm = np.matmul(cn,nu)
    
    print("centripetal and Coriolis matrix ",cm)
    print("Shape of cm matrix",cm.shape)
    
    return cm

# damping matrix
def calculate_damping_matrix(u,v,w,p,q,r):
    
    nu = np.array([[u,v,w,p,q,r]]).transpose()
    
    dn =   np.array([[ 34.55*abs(u),          0,                0,           0,           0,                   0],
                    [ 0,                     104.4*abs(v),            0,           0,    0,            0],
                    [ 0,                     0,        146.5*abs(w),        0,           0,                   0],
                    [ 0,                     0,                0,        0.68*abs(p),    0,                   0],
                    [ 0,                     0,                0,           0,          5.34*abs(q),          0],
                    [ 0,                     0,                0,           0,           0,           3.1*abs(r)]])


    dm = np.matmul(dn,nu)
    
    print("Damping matrix ", dm)
    print("Shape of dm matrix",dm.shape)
    
    return dm

# restoring effects
def calculate_ge_matrix(phi,theta):
    ge = np.array([[(W-B)*np.sin(theta),
                  -(W-B)*np.cos(theta)*np.sin(phi),
                  -(W-B)*np.cos(theta)*np.cos(phi),
                  -(yG*W-yB*B)*np.cos(theta)*np.cos(phi)+(zG*W-zB*B)*np.cos(theta)*np.sin(phi),
                   (zG*W-zB*B)*np.sin(theta)+(xG*W-xB*B)*np.cos(theta)*np.cos(phi),
                  -(xG*W-xB*B)*np.cos(theta)*np.sin(phi)-(yG*W-yB*B)*np.sin(theta)]]).T
    
    print("ge matrix ",ge)
    print("Shape of ge matrix",ge.shape)
    
    return ge

def calculate_tau(tau_pid):
    
    # Thrust coefficients
    K1=K2=K3=K4=K5=K6=K7=K8=40
    
    # THrust coefficient matrix
    K = np.array(np.diag(np.array([K1,K2,K3,K4,K5,K6,K7,K8])))  # 8X8 matrix
    #print("Shape of k", K.shape)
    
    # control signal 
    u = np.array([[0,0,0,0,0,0,0,0]]).transpose()
    u = np.matmul((np.matmul(np.linalg.inv(K),B_t_plus)),tau_pid)   #[8X8][8X6][6X1] = [8X1]
    
    print("control signal ",u.shape)
    print("shape of B_t",B_t.shape)  # 6X8
    print("shape of K",K.shape)
    
    print("control signal dimension ",u.ndim)
    print("dimension of B_t",B_t.ndim)  # 6X8
    print("dimension of K",K.ndim)

    
    # thruster model
    tau = np.matmul(np.matmul(B_t,K),u)  # [6X8][8X8][8X1] = [6X1]
    print("tau = ",tau.shape)
    
    return tau

# Function to calculate acceleration
def calculate_acc(tau,eta,nu):
    
    phi = eta[3][0]; theta = eta[4][0]
    
    u = nu[0][0]; v = nu[1][0]; w = nu[2][0]
    p = nu[3][0]; q = nu[4][0]; r = nu[5][0]
    
    cm = calculate_c_matrix(u,v,w,p,q,r)
    
    dm = calculate_damping_matrix(u,v,w,p,q,r)
    
    ge = calculate_ge_matrix(phi,theta)
    
    n1 = cm + dm + ge    # 6X1
    
    Mn = tau - n1       # [6X1]
    
    print("shape of tau,",tau)     #6X1
    print("shape of n1",n1.shape)  # 6X1 
    print("shape of Mn",Mn.shape)  # 6X1
    
    nudot = np.matmul(Mn.T,np.linalg.inv(M)).T  # [1X6][6X6] = [1X6]
    
    print("nudot matrix", nudot)
    print("shape of nudot matrix",nudot.shape)
    
    return nudot
    
#Function to calculate Velocity(etadot)
def calculate_velo(nudot,J,dt,etadot):
    etadot = etadot + nudot * dt
    print("Valocity ",etadot)
    return np.matmul(J,etadot)


# Function to calculate Position
def calculate_pos(eta,etadot,dt):
    eta = eta + etadot * dt
    print("Position ",eta)
    return eta

def draw_graph(pos_x,pos_y,pos_z,time_step):
    
    #specify one size for all subplots
    fig, ax = plt.subplots(2, 2, figsize=(10,7))
    fig.tight_layout()
    
    pos_x = np.array(pos_x)
    pos_y = np.array(pos_y)
    pos_z = np.array(pos_z)
    
    time_step = np.array(time_step)
    
    #create subplots
    ax[0, 0].plot(time_step, pos_x, color='red')
    ax[0, 1].plot(time_step, pos_y, color='blue')
    ax[1, 0].plot(time_step, pos_z, color='green')
    
    #define subplot titles
    ax[0, 0].set_title('Surge Motion')
    ax[0, 1].set_title('Sway Motion')
    ax[1, 0].set_title('Heave Motion')
    
    # set x axis lebel
    ax[0, 0].set_xlabel('Time(s)')
    ax[0, 1].set_xlabel('Time(s)')
    ax[1, 0].set_xlabel('Time(s)')
    
    # set y axis lebel
    ax[0, 0].set_ylabel('x position(m)')
    ax[0, 1].set_ylabel('y position(m)')
    ax[1, 0].set_ylabel('z position(m)')
    
    plt.pause(10)
    


# main program

show_animation = True

# Simulation parameters

T = 20

def quad_sim(x_c, y_c, z_c):
    """
    Calculates the necessary thrust and torques for the quadrotor to
    follow the trajectory described by the sets of coefficients
    x_c, y_c, and z_c.
    """
    
    # These are initial position and orientation of the vehicle with
    # initial velocity and acceleration
    
    x = -5
    y = -5
    z = 5
    
    u = 0
    v = 0
    w = 0
    
    x_acc = 0
    y_acc = 0
    z_acc = 0
    
    psi = 0
    theta = 0
    phi = 0
    
    p = 0
    q = 0
    r = 0
    
    des_roll = 0
    des_pitch = 0
    des_yaw = 0
    
    global eta
    eta = np.array([[x,y,z,psi,theta,phi]]).transpose()  # Generalized position and orientation
    nu = np.array([[u,v,w,p,q,r]]).transpose()   # Generalized linear and angular velocity
    
    etadot = np.array([[0,0,0,0,0,0]]).transpose()
    nudot = np.array([[0,0,0,0,0,0]]).transpose()
    
    
    # initializing varibles for conrtol system
    prev_error = np.array([[0,0,0,0,0,0]]).transpose()   # 6X1 
    integral = np.array([[0,0,0,0,0,0]]).transpose()    # 6X1
    tau_pid = np.array([[0,0,0,0,0,0]]).transpose()  # 6X1
    derivative = np.array([[0,0,0,0,0,0]]).transpose()  # 6X1
    tau = np.array([[0,0,0,0,0,0]]).transpose()
    current_vel = np.array([[0,0,0,0,0,0]])


    # considering ocean currents
    water_vel = nu - current_vel

    dt = 0.1
    t = 0.0

    auv = AUV(x=x, y=y, z=z, roll=phi,
                  pitch=theta, yaw=psi, size=2, show_animation=show_animation)

    i = 0
    n_run = 1  
    irun = 0
    
    pos_x = []
    pos_y = []
    pos_z = []
    
    time_step = []

    while True:
        num = 0
        while t <= T:
            des_x_pos = calculate_position(x_c[i], t)
            des_y_pos = calculate_position(y_c[i], t)
            des_z_pos = calculate_position(z_c[i], t)
            
            des_x_vel = calculate_velocity(x_c[i], t)
            des_y_vel = calculate_velocity(y_c[i], t)
            des_z_vel = calculate_velocity(z_c[i], t)
            
            des_x_acc = calculate_acceleration(x_c[i], t)
            des_y_acc = calculate_acceleration(y_c[i], t)
            des_z_acc = calculate_acceleration(z_c[i], t)
            
            eta_des = np.array([des_x_pos,des_y_pos,des_z_pos,[des_roll],[des_pitch],[des_yaw]])
            
            # transformation matrix
            
            J = calculate_transformation_matrix(eta[3][0],eta[4][0],eta[5][0])
            print("J matrix ",J)
            
            # calculation of control torque
    
            kp = np.diag(np.array([3,3,3,4,4,2]))                 # 6X6
            kd = np.diag(np.array([0.2,0.2,0.2,0.3,0.3,0.1]))     # 6x6
            ki = np.diag(np.array([2.5,2.5,0.5,0.5,1,0.5]))       # 6X6

            error = eta_des - eta   # 6X1
            print("Error ", error)

            eb = np.matmul(np.linalg.inv(J),error)   # 6X1

            derivative = (eb - prev_error)/dt    # 6X1

            integral = integral + eb*dt  # 6X1
            
            tau_pid = np.matmul(kp,eb) + np.matmul(ki,integral) + np.matmul(kd,derivative)  # [6X6][6X1] 
    
            prev_error = eb
            
            # torque calculation 
            tau = calculate_tau(tau_pid)

            # body acceleration calculation
            nudot = calculate_acc(tau,eta,etadot)

            # inertial position calculation
            etadot = calculate_velo(nudot,J,dt,etadot)
            
            # inertial position calculation
            roll_torque = tau[3][0]    # K
            pitch_torque = tau[4][0]   # M
            yaw_torque = tau[5][0]     # N

            roll_vel = nu[3][0]        # p
            pitch_vel = nu[4][0]       # q 
            yaw_vel = nu[5][0]         # r 

            roll = eta[3][0]           # phi
            pitch = eta[4][0]          # theta
            yaw = eta[5][0]            # psi
            
            x_acc = nudot[0][0]        
            y_acc = nudot[1][0]
            z_acc = nudot[2][0]
            
            x_vel = nu[0][0]          # u
            y_vel = nu[1][0]          # v
            z_vel = nu[2][0]          # w
            
            x_pos = eta[0][0]         # x
            y_pos = eta[1][0]         # y
            z_pos = eta[2][0]         # z
            
            print("x_pos ",x_pos)
            print("y_pos ",y_pos)
            print("z_pos" ,z_pos)
            
            pos_x.append(x_pos)
            pos_y.append(y_pos)
            pos_z.append(z_pos)
            
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
            
            eta = eta + tau_pid * dt
                        
            t += dt
            num += 1
        
        print("num ", num)
        
        t = 0.0
        i = (i + 1) % len(waypoints)
        irun += 1
        if irun >= n_run:
            plt.pause(10)
            break

    print("Done")
    
    draw_graph(pos_x,pos_y,pos_z,time_step)

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
