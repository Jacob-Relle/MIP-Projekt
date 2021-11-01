import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

from skimage import filters

from cvxopt import solvers, matrix, spmatrix, div, exp, mul, log

def J_energy(image, coords):

    #The coordinates of the image
    #coords = [(x[0], x[1]) for x in np.ndindex(image.shape)]
    #define tau
    tau = filters.threshold_otsu(image)
    #define y_x for every element in the region
    y = matrix(np.array([image[x[0], x[1]] - tau for x in coords],dtype=np.double)
               ,(len(coords),1))
    #define nable s(x) for each element
    delta_s = matrix(np.array([[x[0]**2, x[1]**2, 2*x[0]*x[1], x[0], x[1], 1] for x in coords])
                     ,(len(coords), 6))
    def F(theta = None, z = None):
        # choose starting point and as such define dimension of theta
        if theta is None: return 3, matrix(0.0, (6,1))
        # define main objective 'minimize Energie J'
        s = delta_s * theta
        phi = log(1 + exp(-mul(y,s)))
        J = sum(phi) 
        kappa = div(1, 1+exp(mul(y,s)))
        #define the derivativ of the Energie J
        DJ = matrix(1, (1, len(coords))) * (-mul(mul(y, kappa) * matrix(1, (1, 6)), delta_s))
        #Define the first constraint 2*A_3^2-4*A_1*A_2 <= 0 to ensure the result is an ellipse
        f_1 = 4*theta[2]**2 - (4*theta[0]*theta[1])
        Df_1 = matrix([-4*theta[1],-4*theta[0],8*theta[2],0,0,0],(1,6))
        #define the second constraint that A_1 and A_2 have negativ sign
        f_2 = theta[0]
        f_3 = theta[1]
        Df_2 = matrix([1,0,0,0,0,0],(1,6))
        Df_3 = matrix([0,1,0,0,0,0],(1,6))
        #combine constraints
        f = matrix([J,f_1,f_2,f_3],(4,1))
        Df = matrix([DJ,Df_1,Df_2,Df_3],(4,6))
        if z is None: return f, Df
        nu = mul(y**2, kappa - kappa**2)
        #define matrix to multiply from the right for summation
        SumMatrix = matrix(np.tile([[1,0,0,0,0,0], 
                                    [0,1,0,0,0,0], 
                                    [0,0,1,0,0,0], 
                                    [0,0,0,1,0,0], 
                                    [0,0,0,0,1,0], 
                                    [0,0,0,0,0,1]], len(coords)))
        eta = matrix([delta_s[i,:].T * delta_s[i,:] * nu[i] for i in range(len(coords))])
        H_1 = spmatrix([-4,-4,8],[1,0,2],[0,1,2],(6,6))
        H = z[0]*SumMatrix*eta + z[1]* H_1 
        return f, Df, H
    solv = solvers.cp(F)
    return solv['x'], solv['primal objective']

def Solv(image, coords):
    #Set solver options
    solvers.options['show_progress'] = False
    #Initalize model parameter theta
    theta = matrix(np.zeros((6,1)),(6,1))
    #try to solve minimizing Problem of Energie J with high accuracy
    try:
        solvers.options['feastol'] = 1e-7
        theta, f = J_energy(image, coords)
    except:
        try:
            solvers.options['feastol'] = 1e-2
            theta, f = J_energy(image, coords)
        except:
            theta = matrix(-1*np.ones(6),(6,1))
            f = np.inf        
    #if the result is a parbola set the energie to inf 
    if (2*theta[2])**2-4*theta[0]*theta[1] >= 0 and theta[0] != 0:
        theta = matrix(-1*np.ones(6),(6,1))
        f = np.inf  

    return theta, f

def segmented(image, theta, threshold):
    coords = [(x[0], x[1]) for x in np.ndindex(image.shape)]
    delta_s = matrix(np.array([[x[0]**2, x[1]**2, 2*x[0]*x[1], x[0], x[1], 1] for x in coords])
                     ,(len(coords),6))
    s = delta_s * theta
    s = np.reshape(s,image.shape)
    s = s*(s-threshold)
    image = image[...,np.newaxis]
    image = np.concatenate((image,image,image),axis=2)
    image[...,0][s<=0] = 0
    image[...,1][s<=0] = 1
    image[...,2][s<=0] = 0
    return image

def main(path_to_data):

    image = plt.imread(path_to_data)[...,0]
    coords = [(x[0], x[1]) for x in np.ndindex(image.shape)]
    theta = Solv(image, coords)[0]
    segmentation = segmented(image,theta,20)
    plt.imshow(segmentation)
    plt.colorbar()
    plt.show()
    
if __name__ == '__main__' :
    path = 'Single_nuclei/dna-0.png'
    main(path)
