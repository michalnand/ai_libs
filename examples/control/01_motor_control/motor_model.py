import AILibs
import numpy

class MotorModel(AILibs.DynamicalSystem):

    def __init__(self, b, Kt, Ke, R, J, dt):
        
        a_tmp = -(b + Kt*Ke/R)/J
        b_tmp = Kt/(R*J)    
        
        a_mat = numpy.array([[a_tmp]])
        b_mat = numpy.array([[b_tmp]])

        AILibs.DynamicalSystem.__init__(self, a_mat, b_mat, None, dt)

        

