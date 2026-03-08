import numpy

class MadgwickFilter:
    def __init__(self, beta=0.1):
        self.beta = beta
        self.q = [1.0, 0.0, 0.0, 0.0]  # initial quaternion

    def update(self, ax, ay, az, gx, gy, gz, dt):
        """
            Madgwick's IMU algorithm implementation
            
            ax,ay,az : accelerometer [g]
            gx,gy,gz : gyro [rad/s] 

            dt : time step [s]
            beta : learning rate for gradient descent (default 0.1)   

            returns roll pitch yaw in radians 
        """

        q1, q2, q3, q4 = self.q

        # Normalize accelerometer measurement
        norm = (ax * ax + ay * ay + az * az) ** 0.5
       
        
        ax /= norm
        ay /= norm
        az /= norm

        # Gradient descent algorithm corrective step
        # Objective function f = estimated_gravity - measured_gravity
        f1 = 2 * (q2 * q4 - q1 * q3) - ax
        f2 = 2 * (q1 * q2 + q3 * q4) - ay
        f3 = 2 * (0.5 - q2 * q2 - q3 * q3) - az

        # Gradient: J^T * f (Jacobian transpose times objective function)
        s1 = -2 * q3 * f1 + 2 * q2 * f2
        s2 =  2 * q4 * f1 + 2 * q1 * f2 - 4 * q2 * f3
        s3 = -2 * q1 * f1 + 2 * q4 * f2 - 4 * q3 * f3
        s4 =  2 * q2 * f1 + 2 * q3 * f2

        norm_s = (s1 * s1 + s2 * s2 + s3 * s3 + s4 * s4) ** 0.5
        if norm_s > 0:    
            s1 /= norm_s
            s2 /= norm_s
            s3 /= norm_s
            s4 /= norm_s
        else:
            s1 = s2 = s3 = s4 = 0.0

        # Compute rate of change of quaternion
        qDot1 = 0.5 * (-q2 * gx - q3 * gy - q4 * gz) - self.beta * s1
        qDot2 = 0.5 * (q1 * gx + q3 * gz - q4 * gy) - self.beta * s2
        qDot3 = 0.5 * (q1 * gy - q2 * gz + q4 * gx) - self.beta * s3
        qDot4 = 0.5 * (q1 * gz + q2 * gy - q3 * gx) - self.beta * s4

        # Integrate quaternion rate of change
        q1 += qDot1 * dt
        q2 += qDot2 * dt
        q3 += qDot3 * dt
        q4 += qDot4 * dt

        # Normalize quaternion
        norm_q = (q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4) ** 0.5
        q1 /= norm_q
        q2 /= norm_q
        q3 /= norm_q
        q4 /= norm_q

        # Store updated quaternion
        self.q = [q1, q2, q3, q4]

        return self._quat_to_euler(self.q)

    def _quat_to_euler(self, q):
        w, x, y, z = q
        roll  = numpy.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = numpy.arcsin(numpy.clip(2*(w*y - z*x), -1.0, 1.0))
        yaw   = numpy.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return roll, pitch, yaw