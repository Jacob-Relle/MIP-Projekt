import numpy as np
from skimage.measure import regionprops, find_contours
from joblib import Parallel, delayed
from cvxopt import matrix

from Single_nuclei_segmentation import Solv

#create the subimages that will then be minimized
def create_images(fragments, PrototypeList):
    ListOfCoords = []
    for k in range(len(PrototypeList)):
        coords = np.concatenate([regionprops(fragments)[i-1].coords for i in list(PrototypeList[k])])
        ListOfCoords.append(coords)
    return ListOfCoords

#minimize the prototype sets. uses parallelization
def optimise_regions(image, ListOfCoords):
    """
    Find all minimisers and minimum value of energy function for each region of image given by ListOfCoords.

    Input
    -----
    image: 2d-ndarray or matrix
        Image for which we compute minimum energy

    ListOfCoords: list of lists
        each element is a list of int tuples (x, y) of pixel-coordinates in image

    Result
    ------
    (theta, f)
        theta: list of CVX 6x1 matrix of length ListOfCoords
            TODO
        f: list of floats of length ListOfCoords
            TODO
    """
    theta = []
    f = []
    r = Parallel(n_jobs = -3, verbose = 10)(delayed(Solv)(image, coords) for coords in ListOfCoords)
    theta, f = zip(*r)
    return theta, f

#Alg II
def global_solution(f,alpha,fragments,PrototypeList):
    #Set Variables we dont need Z but f_used
    n = len(PrototypeList) 
    u = np.zeros(n)
    V = set()
    Z = set(PrototypeList)
    f_used = np.copy(f)
    regions = regionprops(fragments)
    #First Loop over copy of fragments
    for Zk in PrototypeList:
        V = V.union(Zk)
    while V != set():
        c = np.zeros(n)
        #Loop over number of Subgraphs in PrototypeList
        for k in range(n):
            #Set intersection length
            Zk_V_labels = V.intersection(set(PrototypeList[k]))
            Zk_V_area = 0
            for label in Zk_V_labels:
                Zk_V_area += regions[label-1].area
            #Set c if intersection is non empty 
            if  Zk_V_area !=0:
                c[k] = (f[k] + alpha) / Zk_V_area
            #Else set it to nan to ignore it
            else:
                c[k] = np.nan
        #Get the argmin of c
        if np.nanmin(c)==np.inf:
            break
        k_min = np.nanargmin(c)
        #Set u of argmin to 1
        u[k_min] = 1 
        #Remove the nodes of Z_k
        V -= PrototypeList[k_min]
    
    #Second loop over not used elements of f
    while Z != set():
        if np.nanmin(f_used) == np.inf:
            break
        #Set current element of interest k_prim
        k_prim = np.nanargmin(f_used)
        #check if u[_k_prim hasnt been used in first loop
        if u[k_prim]==0:
            #crate subvector v of u
            v = np.copy(u)
            #loop over all elements of u that are non zero
            for ind in np.nonzero(u)[0]:
                #check if Z_ind is subset of Z_kprim
                if not PrototypeList[ind].issubset(PrototypeList[k_prim]):
                    #if not set v to 0 to ignore it in the union
                    v[ind] = 0
            #check if union of all left subsets is equal to Z_kprim
            union = set()
            for k in np.nonzero(v)[0]:
                for label in PrototypeList[k]:
                    union.add(label)
            if set(PrototypeList[k_prim]) == union:
                #check smth...
                if f[k_prim]+alpha < np.dot(v,f + (alpha*np.ones(n))):
                    #Set u values of the used subsets to 0 and the union to 1
                    u -= v
                    u[k_prim] = 1
        #make f equal to nan for the used region
        f_used[k_prim] = np.nan
        Z -= {PrototypeList[k_prim]}
    return u

#compute segmented picture
def multi_segmentation(image, fragments, PrototypeList, f, alpha, theta):
    u = global_solution(f, alpha, fragments, PrototypeList)
    coords = [(x[0], x[1]) for x in np.ndindex(image.shape)]
    delta_s = matrix(np.array([[x[0]**2, x[1]**2, 2*x[0]*x[1], x[0], x[1], 1] for x in coords]),(len(coords),6))
    ListOfContours = []
    for k in range(len(u)):
        if u[k]==1:
            PrototypeList = delta_s * theta[k]
            PrototypeList = np.reshape(PrototypeList,image.shape)
            PrototypeList[PrototypeList>0]=  1
            PrototypeList[PrototypeList<0]= -1
            ListOfContours.append(find_contours(PrototypeList))
    return ListOfContours