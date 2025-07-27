import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from agiros_msgs.msg import QuadState, Command


class StateStruct:
    def __init__(self, pos=np.zeros(3), 
                        vel=np.zeros(3),
                        acc = np.zeros(3),
                        jerk = np.zeros(3), 
                        snap = np.zeros(3),
                        rot=R.from_quat(np.array([0.,0.,0.,1.])), 
                        ang=np.zeros(3)):
        
        self.pos = pos # R^3
        self.vel = vel # R^3
        self.acc = acc
        self.jerk = jerk
        self.snap = snap
        self.rot = rot # Scipy Rotation rot.as_matrix() rot.as_quat()
        self.ang = ang # R^3
        self.t = 0.



def quad_state_from_msg(msg):
    position = np.array(
        [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    )
    quat = np.array(
        [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
    )

    velocity = np.array(
        [
            msg.velocity.linear.x,
            msg.velocity.linear.y,
            msg.velocity.linear.z,
        ]
    )

    omega = np.array(
        [
            msg.velocity.angular.x,
            msg.velocity.angular.y,
            msg.velocity.angular.z,
        ]
    )

    return position, quat, velocity, omega


def commands_to_msg(thrust, rates):
    msg = Command()
    msg.header.stamp = rospy.Time.now()
    msg.is_single_rotor_thrust = False
    msg.collective_thrust = thrust
    msg.bodyrates.x = rates[0]
    msg.bodyrates.y = rates[1]
    msg.bodyrates.z = rates[2]
    return msg