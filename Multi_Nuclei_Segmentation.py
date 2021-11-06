import cvxopt
import numpy as np
from skimage.measure import regionprops, find_contours
from joblib import Parallel, delayed
from cvxopt import matrix

from Single_nuclei_segmentation import Solv

#create the subimages that will then be minimized
def create_images(fragments: np.ndarray, prototype_list: list[frozenset]) -> list[list]:
    """
    create a list of a list of coordinates. 
    Each element contains coordinates of a subimage obtained by the 
    labels of the corresponding element in prototype_list. These labels match the labeled matrix fragments.

    Parameters
    ----------

    fragments: 2d-ndarray, a labeled matrix in the shape of an image.
            Each label depicts a fragment of an image.

    prototype_list: list of frozensets
        Each set contains labels corresponding to the labels in fragments. 

    Returns
    -------

    coords_list: list of lists
        each element is a list of int tuples (x, y) of pixel-coordinates of the corresponding image
    """

    coords_list = []
    for k in range(len(prototype_list)):
        coords = np.concatenate([regionprops(fragments)[i-1].coords for i in list(prototype_list[k])])
        coords_list.append(coords)
    return coords_list

#minimize the prototype sets. uses parallelization
def optimise_regions(image: np.ndarray, coords_list: list[list]) -> tuple[list[cvxopt.matrix], list[float]]:
    """
    Find all minimisers and minimum value of energy function J for each region of image given by coords_list.

    Parameters
    -----

    image: 2d-ndarray or matrix
        Image for which we compute minimum energy

    coords_list: list of lists
        each element is a list of int tuples (x, y) of pixel-coordinates in image

    Returns
    ------

    theta: list of CVX 6x1 matrix of length coords_list
        each element is the minimizer of energy function J for the subimage of the same index in coords_list.
        TODO explain what the elements of theta represent
    f: list of floats of length coords_list
        each element is the minimum of the energy function J for the subimage of the same index in coords_list,
        i.e the value J evaluted at point theta.
    """
    theta = []
    f = []
    r = Parallel(n_jobs = -3, verbose = 10)(delayed(Solv)(image, coords) for coords in coords_list)
    theta, f = zip(*r)
    return theta, f

#Alg II
def global_solution(fragments : np.ndarray, prototype_list: list[frozenset] , f: list[float],  alpha = None) -> list[int]:

    """
    Find prototype regions with minimal energy. 

    Parameters
    ----------
    fragments: ndarray, an int labeled matrix

    prototype_list: list of frozensets of length n

    f: list of floats of length n
        energy values to be minimzed along the prototype_list

    alpha: float
        if None, default is np.median(f).

    Returns
    -------

    u: ndarry
        Binary vector of length n.
    """

    #Set Variables we dont need Z but f_used
    if alpha is None:
        alpha = np.median(f)

    n = len(prototype_list) 
    u = np.zeros(n)
    V = set()
    Z = set(prototype_list)
    f_used = np.copy(f)
    regions = regionprops(fragments)
    #First Loop over copy of fragments
    for Zk in prototype_list:
        V = V.union(Zk)
    while V != set():
        c = np.zeros(n)
        #Loop over number of Subgraphs in prototype_list
        for k in range(n):
            #Set intersection length
            Zk_V_labels = V.intersection(set(prototype_list[k]))
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
        V -= prototype_list[k_min]
    
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
                if not prototype_list[ind].issubset(prototype_list[k_prim]):
                    #if not set v to 0 to ignore it in the union
                    v[ind] = 0
            #check if union of all left subsets is equal to Z_kprim
            union = set()
            for k in np.nonzero(v)[0]:
                for label in prototype_list[k]:
                    union.add(label)
            if set(prototype_list[k_prim]) == union:
                #check smth...
                if f[k_prim]+alpha < np.dot(v,f + (alpha*np.ones(n))):
                    #Set u values of the used subsets to 0 and the union to 1
                    u -= v
                    u[k_prim] = 1
        #make f equal to nan for the used region
        f_used[k_prim] = np.nan
        Z -= {prototype_list[k_prim]}
    return u

#compute segmented picture
def multi_segmentation(image, fragments: np.ndarray, prototype_list: list[frozenset], theta = list[cvxopt.matrix], f = list[float], alpha = None) -> list[list]:
    """
    Parameters
    ----------
    image: 2d-ndarray or matrix
        Image for which we compute minimum energy

    fragments: 2d-ndarray, a labeled matrix of the same type and shape as image.
        Each label depicts a fragment of the input picture ``image``.

    prototype_list: list of frozensets of length n
        Each set contains labels corresponding to the labels in fragments.

    theta: 

    f: list of floats of length n

    alpha: 

    Returns
    -------
    contours_list: list 
        List of length n containing lists with the contour lines of the optimal segmentation
        
    """

    if alpha is None:
        alpha = np.median(f)

    u = global_solution(fragments, prototype_list, f, alpha)
    coords = [(x[0], x[1]) for x in np.ndindex(image.shape)]
    delta_s = matrix(np.array([[x[0]**2, x[1]**2, 2*x[0]*x[1], x[0], x[1], 1] for x in coords]),(len(coords),6))
    contours_list = []
    for k in range(len(u)):
        if u[k]==1:
            prototype_list = delta_s * theta[k]
            prototype_list = np.reshape(prototype_list,image.shape)
            prototype_list[prototype_list>0]=  1
            prototype_list[prototype_list<0]= -1
            contours_list.append(find_contours(prototype_list))
    return contours_list