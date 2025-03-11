import math


class solver(object):

    def __init__(self):
        self.num_iter = 1    # Risha: Upper limit on number of iterations
        self.threshold = 1  # Risha: Parameter to store the threshold for convergence
        self.A_r = 1           #Added rows and column sizes for operands 
        self.A_c = 1
        self.x_r = 1
        self.x_c = 1
        self.b_r = 1    
        self.b_c = 1
        self.xin_r = 1
        self.xin_c = 1

    # reset topology parameters
    def reset(self):
        print("All data reset")
        self.num_iter = 10  
        self.threshold = 1

    # return number of iterations
    def get_num_iter(self):
        return self.num_iter

    def get_A_dims(self,layer_id):
        return (self.A_r,self.A_c)

    def get_x_dims(self,layer_id):
        return (self.x_r,self.x_c)

    def get_b_dims(self,layer_id):
        return (self.b_r,self.b_c)

    def get_xin_dims(self,layer_id):
        return (self.xin_r,self.xin_c)

    def set_num_iter(self,iter):
        self.num_iter = iter

    def set_A_dims(self,A_r,A_c):
        self.A_r = A_r
        self.A_c = A_c

    def set_x_dims(self,x_r,x_c):
        self.x_r = x_r
        self.x_c = x_c

    def set_b_dims(self,b_r,b_c):
        self.b_r = b_r
        self.b_c = b_c

    def set_xin_dims(self,xin_r,xin_c):
        self.xin_r = xin_r
        self.xin_c = xin_c

    

if __name__ == '__main__':
    sv = solver()