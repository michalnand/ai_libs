import numpy
import cv2
import time

class MotorDemo:


    def __init__(self, motor_model, kalman_filter):
        self.motor_model    = motor_model
        self.kalman_filter  = kalman_filter

        self.motor_model.reset()
        self.kalman_filter.reset()

        self.steps = 0

        self.brake_force    = 0.0
        self.inertia_change = 0.0

        self.xr     = []
        self.xhat   = []
        self.a      = []

    def step(self):
        self.steps+= 1
        
        if self.steps%10 == 0:
            key = cv2.waitKey(1)
        
            # escape key
            if key == 27:
                return False
            
            elif key == ord('b'):
                self.brake_force = 0.08
                print("brake force")

            elif key == ord('m'):
                self.inertia_change = 0.004
                print("mass change")

            elif key == ord('d'):
                self.brake_force = 0.0
                self.inertia_change = 0.0
            

        if (self.steps//2000)%2 != 0:
            u_in = 0.2
        else:
            u_in = 0.0

       

        u_tmp = numpy.array([[u_in]])

        

        xr, yr = self.motor_model.step(u_tmp, self.brake_force, self.inertia_change)

        xm, a = self.kalman_filter.step(yr, u_tmp)

        print(self.brake_force, self.inertia_change, a)

        if self.steps%20 == 0:
            self._add_log(xr, xm, a)

        if self.steps%10 == 0:
            real_system_img = self._get_motor_image(xr[0][0], "real motor", (0.4, 0.0, 1.0))
            model_img       = self._get_motor_image(xm[0][0], "model", (1.0, 0.0, 0.4))

            image = numpy.concatenate([real_system_img, model_img], axis=1)

            graph = self._get_graph_image(self.xr, self.xhat, self.a)
            image = numpy.concatenate([image, graph], axis=0)

            image = self._add_legend(image)

            cv2.imshow("motor demo", image)

        
            
        return True
    

    def _get_motor_image(self, angle, text, color = (1, 1, 1), size = 400):
        img = numpy.zeros((size, size, 3), dtype=numpy.float32)

        r = size*0.5*0.8
        img = cv2.circle(img, center=(size//2, size//2), radius=int(r), color=color, thickness=2)

        x = int(size//2 + (r-r*0.125)*numpy.cos(angle))
        y = int(size//2 + (r-r*0.125)*numpy.sin(angle))

        img = cv2.circle(img, center=(x, y), radius=int(r*0.15), color=color, thickness=-1)


        img = cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 1, 1), 2, cv2.LINE_AA)

        return img
    
    def _get_graph_image(self, xr, xhat, a, color_a = (0.4, 0.0, 1.0), color_b = (1.0, 0.0, 0.4), h = 200, w = 800):
        img = numpy.zeros((h, w, 3), dtype=numpy.float32)

        if len(xr) < 10:
            return img

        xr   = numpy.array(xr)
        xhat = numpy.array(xhat)
        a    = numpy.array(a)


        img = self._plot_line(img, xr[:, 1], color_a, w, h, 4.0)
        img = self._plot_line(img, xhat[:, 1]+0.2, color_b, w, h, 4.0)
        img = self._plot_line(img, a[:], (0.0, 1.0, 0.0), w, h, 200.0)

       
        return img
    

    
    def _add_log(self, xr, xhat, a, max_len = 200):
        if len(self.xr) > max_len:
            self.xr.pop(0)

        if len(self.xhat) > max_len:
            self.xhat.pop(0)

        if len(self.a) > max_len:
            self.a.pop(0)

        self.xr.append(xr[:, 0])
        self.xhat.append(xhat[:, 0])
        self.a.append(a)


    def _plot_line(self, img, x, color, w, h, scale):
        n_points = x.shape[0]

        points_scaled = 0.5*h - x*scale
        points_scaled = numpy.vstack([numpy.arange(n_points)*w/n_points, points_scaled]).T


        points_scaled = numpy.clip(points_scaled, 0, w)
        points_scaled = numpy.array(points_scaled, dtype=int)

        points_scaled = points_scaled.reshape((-1, 1, 2))

        img = cv2.polylines(img, [points_scaled], False, color, 2)

        return img

    def _add_legend(self, img):

        w = img.shape[1]
        h = img.shape[0]

        print(w, h)

        img = cv2.putText(img, "motor velocity", (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0.4, 0.0, 1.0), 2, cv2.LINE_AA)
        img = cv2.putText(img, "model velocity", (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 0.0, 0.4), 2, cv2.LINE_AA)
        img = cv2.putText(img, "anomaly score",  (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0.0, 1.0, 0.4), 2, cv2.LINE_AA)

        if self.brake_force > 0:
            img = cv2.putText(img, "brake force ON",  (200, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 1.0, 1.0), 2, cv2.LINE_AA)
        else:
            img = cv2.putText(img, "brake force OFF",  (200, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0.5, 0.5, 0.5), 2, cv2.LINE_AA)

        if self.inertia_change > 0:
            img = cv2.putText(img, "extra mass force ON",  (200, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1.0, 1.0, 1.0), 2, cv2.LINE_AA)
        else:
            img = cv2.putText(img, "extra mass force OFF",  (200, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0.5, 0.5, 0.5), 2, cv2.LINE_AA)

        return img