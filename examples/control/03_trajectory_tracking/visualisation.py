import numpy
import cv2

class TrajectoryRenderer:

    def __init__(self, min_pos = 0.0, max_pos = 1.0, height = 512, width = 512, history_length = 1000):
        self.height = height
        self.width  = width

        self.min_pos = min_pos
        self.max_pos = max_pos

        self.history_length = history_length

        self.past_positions = []

    def step(self, positions, colors, labels):
        result_im = numpy.zeros((self.height, self.width, 3))

        if len(self.past_positions) >= self.history_length:
             self.past_positions.pop(0)
        
        self.past_positions.append(positions)

        if len(self.past_positions) > 5:
            result_im = self._plot_trajectory(result_im, numpy.array(self.past_positions), colors)

        for n in range(len(positions)):
            x_pos = float(positions[n][0])
            y_pos = float(positions[n][1])

            color = colors[n]

            x_pos = self._normalize_pos(x_pos, self.min_pos, self.max_pos, 0.1*self.width, 0.9*self.width)
            y_pos = self._normalize_pos(y_pos, self.min_pos, self.max_pos, 0.1*self.height, 0.9*self.height)

            x_pos = int(numpy.clip(x_pos, 0, self.width))       
            y_pos = int(numpy.clip(y_pos, 0, self.height))
        

        # plot positions
        for n in range(len(positions)):
            x_pos = float(positions[n][0])
            y_pos = float(positions[n][1])

            color = colors[n]

            x_pos = self._normalize_pos(x_pos, self.min_pos, self.max_pos, 0.1*self.width, 0.9*self.width)
            y_pos = self._normalize_pos(y_pos, self.min_pos, self.max_pos, 0.1*self.height, 0.9*self.height)

            x_pos = int(numpy.clip(x_pos, 0, self.width))       
            y_pos = int(numpy.clip(y_pos, 0, self.height))
                                    
            r = int(1.0 + self.height/50.0 + 10*(1.0 - n/len(positions)))
            result_im = cv2.circle(result_im, center=(x_pos, y_pos), radius=r, color=color, thickness=-1)
        
        # plot labels
        for n in range(len(positions)):
            color = colors[n]
            text_str = str(labels[n])

            x_pos = self.width//2
            y_pos = self.height//2 + 25*n

            result_im = cv2.putText(result_im, text_str, (x_pos, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


        cv2.imshow("trajectory_tracking", result_im)

        key = cv2.waitKey(1)

        if key == 27:
            return 1
        else:
            return 0

    def _normalize_pos(self, x, in_min, in_max, out_min, out_max):
        k = (out_max - out_min)/(in_max - in_min)
        q = out_min - k*in_min  
        y = k*x + q

        return y
    
    def _plot_trajectory(self, result_im, positions, colors):
        
        for n in range(positions.shape[1]):
            points = positions[:, n]
            
            points[:, 0] = self._normalize_pos(points[:, 0], self.min_pos, self.max_pos, 0.1*self.width, 0.9*self.width)
            points[:, 1] = self._normalize_pos(points[:, 1], self.min_pos, self.max_pos, 0.1*self.height, 0.9*self.height)
            points[:, 0] = numpy.clip(points[:, 0], 0, self.width)
            points[:, 1] = numpy.clip(points[:, 1], 0, self.height)

            points = points.astype(numpy.int32).reshape((-1, 1, 2))           

            result_im = cv2.polylines(result_im, [points], isClosed=False, color=colors[n], thickness=2)
    
        return result_im

