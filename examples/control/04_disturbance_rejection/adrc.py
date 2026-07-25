
import numpy as np  
import matplotlib.pyplot as plt  
  
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
        self.z = np.zeros(3)  
    
    def forward(self, r, y):  
        # 1. Calculate estimation error  
        e_obs = y - self.z[0]  
        
        # 2. Update ESO using Euler integration  
        dz0 = self.z[1] + self.l1 * e_obs  
        dz1 = self.z[2] + self.b0 * getattr(self, 'u_prev', 0.0) + self.l2 * e_obs  
        dz2 = self.l3 * e_obs  
        
        self.z[0] += dz0 * self.dt  
        self.z[1] += dz1 * self.dt  
        self.z[2] += dz2 * self.dt  
        
        # 3. Virtual PD control input  
        u0 = self.kp * (r - self.z[0]) - self.kd * self.z[1]  
        
        # 4. Total control input with disturbance cancellation  
        u = (u0 - self.z[2]) / self.b0  
        
        self.u_prev = u  
        return u  

if __name__ == "__main__":
    # --- Simulation Setup ---  
    dt = 0.001  
    sim_time = 5.0  
    steps = int(sim_time / dt)  
    
    # Plant: Real system has different b and unmodeled friction/dynamics  
    # y_ddot = -0.5*y_dot - 2*y + 1.5*u + disturbance  
    def plant(y, y_dot, u, dist):  
        y_ddot = -0.5 * y_dot - 2.0 * y + 1.5 * u + dist  
        return y_ddot  
    
    # Instantiate ADRC (we guess b0 = 1.0, even though real b = 1.5)  
    adrc = LinearADRC2ndOrder(b0=1.0, wc=10.0, wo=40.0, dt=dt)  
    
    # Storage for results  
    y_hist, r_hist, u_hist, f_est_hist = [], [], [], []  
    
    y, y_dot = 0.0, 0.0  
    r = 1.0 # Step reference  
    
    for step in range(steps):  
        t = step * dt  
        # Inject an external disturbance force at t = 2.5s  
        dist = 5.0 if t >= 2.5 else 0.0  

        # Controller step  
        u = adrc.update(r, y)  

        # Physical system integration (Euler method)  
        y_ddot = plant(y, y_dot, u, dist)  
        y_dot += y_ddot * dt  
        y += y_dot * dt  

        # Save data  
        y_hist.append(y)  
        r_hist.append(r)  
        u_hist.append(u)  
        f_est_hist.append(adrc.z[2])  
    
    print(f"Simulation completed across {steps} steps.")  
    print(f"Final output value y: {y_hist[-1]:.4f} (Target: {r})")  
    print(f"Estimated disturbance at end: {f_est_hist[-1]:.2f}")  