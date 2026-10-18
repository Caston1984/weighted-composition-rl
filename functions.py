import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly


def generate_xy(func, xrange,num=500):
    """
    Returns the x and the y axis to be plot
    :param func: function to be evaluated
    :param xrange: max and min of the range
    :param num: number of points to evaluate
    :return:
    """
    xmin = np.min(xrange)
    xmax = np.max(xrange)
    x_axis = np.linspace(xmin,xmax, num=num)
    y_axis = np.apply_along_axis(np.vectorize(func),0,x_axis)
    return x_axis,y_axis

class step:
    def __init__(self):
        return
    def eval(self,x):
        p=[]
        if x < 0.2:
            p = 0.2
        if x >= 0.2 and x < 0.3:
            p = 0.8
        if x >= 0.3:
            p = 0.1
        return p

class complex_trig:
    def __init__(self):
        return
    def eval(self,x):
        p = 1 / (12 * (np.sin(13 * x) * np.sin(27 * x) + 1))
        return p

class uniform:
    def __init__(self):
        return
    def eval(self,x):
        return 0.5

class linear:
    def __init__(self):
        return

    def eval(self,x):
        return 0.1 + (x-0.1)*0.5

class triangle:
    def __init__(self):
        return
    def eval(self,x):
        p=[]
        if x < 0.3:
            p = 0.1 + x
        else:
            p = 0.4 - 0.25*x
        return p

class binormal:
    def __init__(self):
        return
    def eval(self,x):
        norm1 = stats.norm(loc=0.4,scale=0.05)
        norm2 = stats.norm(loc=0.8, scale=0.05)
        p = (norm1.pdf(x)/norm1.pdf(0.4) + norm2.pdf(x)/norm2.pdf(0.8))
        return p

class normal:
    def __init__(self):
        return
    def eval(self,x):
        norm = stats.norm(loc=0.8,scale=0.5)
        p = norm.pdf(x)/norm.pdf(0.8)
        return p

def getMaxFunc(x_axis,y_axis):
    """

    :param x_axis:
    :param y_axis:
    :return:
    """
    index = np.argmax(y_axis)
    return x_axis[index], y_axis[index]


class randomPoly:
    def __init__(self):
        random_n = np.random.randint(1,10,1)[0]
        # generating 30 uniform points
        random_coeff_x = np.random.uniform(0,1,30)
        random_coeff_y = np.random.uniform(0,1,30)

        #print random_coeff
        self.ffit = poly.polyfit(random_coeff_x, random_coeff_y, random_n)

        x_axis = np.linspace(0, 1, num=500)
        y_axis = np.apply_along_axis(np.vectorize(self.init_eval), 0, x_axis)
        self.ymax = np.max(y_axis)
        self.ymin = np.min(y_axis)
       # print self.ymax, self.ymin


    def init_eval(self,x):
        return poly.polyval(x,self.ffit)

    def eval(self,x):
        """Normalize between 0 and ~0.833"""
        p = (self.init_eval(x) - self.ymin)/((self.ymax - self.ymin)*1.2)
        if p > 1:
            p = 1
        if p < 0:
            p = 0
        return p



if __name__ == "__main__":
    pfunc = randomPoly().eval
    x,y = generate_xy(pfunc,[0,1])
    plt.plot(x, y)
    plt.show()
    # randompolyfunc = randomPoly().eval
    # print randompolyfunc(0.5)
    # print randompolyfunc(0.8)
    # print randompolyfunc(0.5)
    # print randompolyfunc(0)


