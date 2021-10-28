import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from skimage import filters

from cvxopt import solvers, matrix, div, exp, mul, log

def J_energy(image):

    #The coordinates of the image
    coords = [(x[0], x[1]) for x in np.ndindex(image.shape)]
    #define tau
    tau = filters.threshold_otsu(image)
    #define y_x for every element in the region
    y = matrix(np.array([image[x[0], x[1]] - tau for x in coords],dtype=np.double),(len(coords),1))
    #define nable s(x) for each element
    delta_s = matrix(np.array([[x[0]**2, x[1]**2, 2*x[0]*x[1], x[0], x[1], 1] for x in coords]), (len(coords), 6))
    def F(theta = None, z = None):
        # choose starting point and as such define dimension of theta
        if theta is None: return 0, matrix(0.0, (6,1))
        # s contains region.area many rows, each row contains a scalar.
        s = delta_s * theta
        phi = log(1 + exp(-mul(y,s)))
        J = sum(phi)
        kappa = div(1, 1+exp(mul(y,s)))
        DJ = matrix(1, (1, len(coords))) * (-mul(mul(y, kappa) * matrix(1, (1, 6)), delta_s))
        if z is None: return J, DJ
        nu = mul(y**2, kappa - kappa**2)
        #define matrix to multiply from the right for summation
        SumMatrix = matrix(np.tile([[1,0,0,0,0,0], 
                                    [0,1,0,0,0,0], 
                                    [0,0,1,0,0,0], 
                                    [0,0,0,1,0,0], 
                                    [0,0,0,0,1,0], 
                                    [0,0,0,0,0,1]], len(coords)))
        eta = matrix([delta_s[i,:].T * delta_s[i,:] * nu[i] for i in range(len(coords))])
        H = z[0]* SumMatrix * eta
        return J, DJ, H
    solv = solvers.cp(F)
    return solv['x'], solv['primal objective']

def Solv(image):
    #Set solver options
    solvers.options['show_progress'] = False
    #Initalize model parameter theta
    theta = matrix(np.zeros((6,1)),(6,1))
    #try to solve minimizing Problem of Energie J with high accuracy
    try:
        solvers.options['feastol'] = 1e-7
        theta, f = J_energy(image)
    except:
        solvers.options['feastol'] = 1e-2
        theta, f = J_energy(image)
    return theta, f

def segmented(image, theta,threshold):
    coords = [(x[0], x[1]) for x in np.ndindex(image.shape)]
    delta_s = matrix(np.array([[x[0]**2, x[1]**2, 2*x[0]*x[1], x[0], x[1], 1] for x in coords]),(len(coords),6))
    s = delta_s * theta
    s = np.reshape(s,image.shape)
    image = np.concatenate((image,image,image),axis=2)
    image[...,0][abs(s)<threshold] = 0
    image[...,1][abs(s)<threshold] = 1
    image[...,2][abs(s)<threshold] = 0
    return image

def main(path_to_data):

    image = plt.imread(path_to_data)[...,0]
    theta = Solv(image)[0]
    segmentation = segmented(image,theta,20)
    plt.imshow(segmentation)
    plt.colorbar()
    plt.show()
    
if __name__ == '__main__' :
    path = 'dna-0.png'
    main(path)
