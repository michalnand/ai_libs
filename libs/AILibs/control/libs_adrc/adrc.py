import numpy
  
class LinearADRC2ndOrder:  
    def __init__(self, b0, wc, wo, dt):  
        """  
        b0: Nominal gain estimate  
        wc: Controller bandwidth (rad/s)  
        wo: Observer bandwidth (rad/s) (~ 3 to 5 times wc)  
        dt: Sample time (seconds)  
        """  
        self.b0 = b0  
        self.dt = dt      
        
        # Controller Gains (pole placement at -wc)  
        self.kp = wc**2  
        self.kd = 2 * wc
        
        # Observer Gains (pole placement at -wo)  
        self.l1 = 3 * wo  
        self.l2 = 3 * (wo**2)  
        self.l3 = wo**3  
        
        # Observer States: [y_hat, dy_hat, f_total_hat]  
        self.z = numpy.zeros(3)  

    def reset(self):
        self.z = numpy.zeros(3)     

    def forward(self, xr, x, u_curr):  
        # 1. Calculate estimation error  
        e_obs = x - self.z[0]  
        
        # 2. Update ESO using Euler integration  
        dz0 = self.z[1] + self.l1 * e_obs  
        dz1 = self.z[2] + self.l2 * e_obs + self.b0 * u_curr 
        dz2 = self.l3 * e_obs   
        
        self.z[0] += dz0 * self.dt  
        self.z[1] += dz1 * self.dt  
        self.z[2] += dz2 * self.dt  
        
        # 3. Virtual PD control input  
        u0 = self.kp * (xr - self.z[0]) - self.kd * self.z[1]  

        # 4. Total control input with disturbance cancellation  
        u = (u0 - self.z[2]) / self.b0  
        
        return u  
