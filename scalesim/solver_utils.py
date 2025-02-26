import math


class solver(object):

    def __init__(self):
        self.num_iter = 100    # Risha: Upper limit on number of iterations
        self.threshold = 0.02  # Risha: Parameter to store the threshold for convergence

    # reset topology parameters
    def reset(self):
        print("All data reset")
        self.num_iter = 100    
        self.threshold = 0.02 

    # return number of iterations
    def get_num_iter(self):
        return self.num_iter

    def get_num_iter(self):
        return self.num_iter

    def get_A_dims(self,layer_id):
        return (109,109)

    def get_x_dims(self,layer_id):
        return (109,1)

    def get_b_dims(self,layer_id):
        return (109,1)

    def get_xin_dims(self,layer_id):
        return (109,1)

if __name__ == '__main__':
    sv = solver()