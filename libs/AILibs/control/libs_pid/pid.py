import numpy

'''
    This is a textbook-style PID controller.
    It directly applies the proportional, integral, and derivative terms 
    in continuous form without discretization or safeguards.
    Note:
        - This version is primarily for educational or illustrative purposes.
        - Not robust for practical applications (e.g., lacks anti-windup, 
          derivative filtering, or saturation limits).
'''
class PIDTextbook:

    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.err_sum  = 0.0
        self.err_now  = 0.0
        self.err_prev = 0.0 
    

    def forward(self, xr, x):
        '''
        Inputs:
            xr : desired reference value (float or numpy scalar)
            x  : current system output (float or numpy scalar)

        Returns:
            result : control signal (float or numpy scalar)
        '''
        # Shift previous error
        self.err_prev = self.err_now
        # Compute new error
        self.err_now = xr - x
        # Integrate error (accumulate over time)
        self.err_sum += self.err_now

        # Compute PID control law:
        # u = Kp * e + Ki * ∫e dt + Kd * de/dt
        result  = self.kp * self.err_now
        result += self.ki * self.err_sum
        result += self.kd * (self.err_now - self.err_prev)

        return result







'''
    Discrete-time PID controller:
    This version uses a difference equation form suitable for digital control.
    
    Discrete PID update law:
        u(n+1) = u(n) + k0*e(n) + k1*e(n-1) + k2*e(n-2)

    Advantages:
        - Avoids derivative noise amplification
        - Supports anti-windup and output-rate limiting
'''
class PID:

    def __init__(self, kp, ki, kd, antiwindup=10**10, du_max=10**10):
        # Precompute equivalent discrete-time gains (based on Tustin or backward difference)
        self.k0 = kp + ki + kd      # coefficient for e(n)
        self.k1 = -kp - 2.0 * kd    # coefficient for e(n-1)
        self.k2 = kd                # coefficient for e(n-2)

        # Initialize error storage (three previous samples)
        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0
        
        # Output constraints
        self.antiwindup = antiwindup  # output saturation (integral windup prevention)
        self.du_max     = du_max      # maximum change in control signal (rate limit)

    def forward(self, xr, x, u_prev):
        '''
        Inputs:
            xr     : required (reference) output, float
            x      : current measured system output, float
            u_prev : previous control signal, float (for incremental update)

        Returns:
            u : next control signal, float
        '''
        # Shift error history
        self.e2 = self.e1
        self.e1 = self.e0
        # Compute current error
        self.e0 = xr - x

        # Compute control signal increment (Δu)
        du = self.k0*self.e0 + self.k1*self.e1 + self.k2*self.e2

        # Limit maximum change rate (Δu)
        du = numpy.clip(du, -self.du_max, self.du_max)

        # Compute total control signal with anti-windup (output saturation)
        u = numpy.clip(u_prev + du, -self.antiwindup, self.antiwindup)

        return u

    def reset(self):
        '''
        Resets the stored error history.
        Useful when reinitializing the controller or restarting control loops.
        '''
        self.e0 = 0.0
        self.e1 = 0.0
        self.e2 = 0.0