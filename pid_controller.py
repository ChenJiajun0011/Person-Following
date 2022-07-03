from geometry_msgs.msg import Twist

def stop_controls():
    twist = Twist()

    twist.linear.x = 0
    twist.linear.y = 0
    twist.linear.z = 0

    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = 0

    return twist

def get_controls(x, z, Kp_l, Ki_l, Kd_l, Kp_a, Ki_a, Kd_a, il, ia, dl, da):
    """
    PID controller - Proportional (P), Integral (I), Derivative (D)
    Kp: Proportional Gain
    Ki: Integral Gain
    Kd: Derivative Gain
    l: linear
    a: angular
    """

    i_error_l = il #integral error for linear
    i_error_a = ia #integral error for angular
    d_error_l = dl #derivative error for linear
    d_error_a = da #derivative error for angular

    #proportional error for linear

    # min_distance = 0.5
    # p_error_l = z - 0.5 

    #min_distance = 1.5
    p_error_l = z - 1.5
    #proportional error for angular
    p_error_a = x - 320 

    if abs(p_error_l) < 0.2:
        p_error_l = 0
    if (0 < p_error_a < 30) or (-30 < p_error_a < 0):
        p_error_a = 0

    i_error_l += p_error_l
    i_error_a += p_error_a
    curr_d_error_l = p_error_l - d_error_l
    curr_d_error_a = p_error_a - d_error_a

    linear = Kp_l*p_error_l + Ki_l*i_error_l + Kd_l*curr_d_error_l
    angular = Kp_a*p_error_a + Ki_a*i_error_a + Kd_a*curr_d_error_a
    

    # maximum speed & angular 
    if linear > 0.1:  #0.6  
        linear = 0.1  #maximum forward speed

    if angular > 0.1:  #0.3
        angular = 0.1  # maximum angular turn 0.1

    if linear < -0.1: #-0.6
        linear = -0.1 # minimum backward

    if angular < -0.1: # -0.3
        angular = -0.1 # maximum angular turn

    twist = Twist()

    twist.linear.x = linear
    twist.linear.y = 0
    twist.linear.z = 0

    twist.angular.x = 0
    twist.angular.y = 0
    twist.angular.z = angular
    
    error = (i_error_l, i_error_a, d_error_l, d_error_a)

    return twist, error

def get_average_distance(depth_frame, box):
    if np.sum(box) == 0:
        return 0

    center_x = box[0] + (box[2]-box[0])//2
    center_y = box[1] + (box[3]-box[1])//2
    sum_ = np.zeros((100,100))
    for i in range(-50, 50):
        for j in range(-50,50):
            x_coor = max(0, min(center_x+i, box[2]))
            y_coor = max(0, min(center_y+j, box[3]))
            try:
                sum_[i,j] = depth_frame.get_distance(x_coor, y_coor)
            except:
                print("out of frame!")
                continue
    smooth = gaussian_filter(sum_, sigma=7)
    const = np.sum(sum_)
    return const/(10000)