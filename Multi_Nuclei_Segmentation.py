import numpy as np
from skimage.measure import regionprops
from joblib import Parallel, delayed
from tqdm import tqdm
from cvxopt import matrix

from Single_nuclei_segmentation import J_energy, Solv

def create_images(image, Omega, S):
    ListOfImages = []
    for k in range(len(S)):
        ListOfRegions = [regionprops(Omega)[i-1] for i in list(S[k])]
        bboxes = np.array([region.bbox for region in ListOfRegions])
        bbox = (min(bboxes[:,0]), max(bboxes[:,2]), min(bboxes[:,1]),  max(bboxes[:,3]))
        sub_img = image[bbox[0]:bbox[1], bbox[2]:bbox[3]]
        ListOfImages.append(sub_img)
    return ListOfImages

def optimise_fragments(ListOfImages):
    theta = []
    f = []
    r = Parallel(n_jobs = -3, verbose = 10)(delayed(Solv)(image) for image in ListOfImages)
    theta, f = zip(*r)
    return theta, f

def global_solution(f,alpha,Omega,S):
    #Set Variables we dont need Z but f_used
    n = len(S) 
    u = np.zeros(n)
    V = set()
    Z = set(S)
    f_used = np.copy(f)
    regions = regionprops(Omega)
    #First Loop over copy of Omega
    for Zk in S:
        V = V.union(Zk)
    while V != set():
        c = np.zeros(n)
        #Loop over number of Subgraphs in S
        for k in range(n):
            #Set intersection length
            Zk_V_labels = V.intersection(set(S[k]))
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
        k_min = np.nanargmin(c)
        #Set u of argmin to 1
        u[k_min] = 1 
        #Remove the nodes of Z_k
        V -= S[k_min]
    
    #Second loop over not used elements of f
    while Z != set():
        #Set current element of interest k_prim
        k_prim = np.nanargmin(f_used)
        #check if u[_k_prim hasnt been used in first loop
        if u[k_prim]==0:
            #crate subvector v of u
            v = np.copy(u)
            #loop over all elements of u that are non zero
            for ind in np.nonzero(u)[0]:
                #check if Z_ind is subset of Z_kprim
                if not S[ind].issubset(S[k_prim]):
                    #if not set v to 0 to ignore it in the union
                    v[ind] = 0
            #check if union of all left subsets is equal to Z_kprim
            union = set()
            for k in np.nonzero(v)[0]:
                for label in S[k]:
                    union.add(label)
            if set(S[k_prim]) == union:
                #check smth...
                if f[k_prim]+alpha < np.dot(v,f + (alpha*np.ones(n))):
                    #Set u values of the used subsets to 0 and the union to 1
                    u -= v
                    u[k_prim] = 1
        #make f equal to nan for the used region
        f_used[k_prim] = np.nan
        Z -= {S[k_prim]}
    return u

def multi_segmentation(image, fragments, PrototypeList, f, theta):
    alpha = np.median(f)
    u = global_solution(f, alpha, fragments, PrototypeList)
    coords = [(x[0], x[1]) for x in np.ndindex(image.shape)]
    delta_s = matrix(np.array([[x[0]**2, x[1]**2, 2*x[0]*x[1], x[0], x[1], 1] for x in coords]),(len(coords),6))
    for k in range(len(u)):
        if u[k]==1:
            s_k = delta_s * theta[k]
            try:
                s = np.maximum(s,s_k)
            except:
                s = s_k
    s = np.reshape(s,image.shape)
    s[s>0] = 1
    s[s<0] = -1
    return s

def segment_EV(image, Omega, Z, f, theta):
    alpha = np.median(f)
    u = global_solution(f, alpha, Omega, Z)
    segmentation = np.zeros_like(image)
    for k in range(len(u)):
        if u[k]==1:
            regions = []
            for i in Z[k]:
                regions.append(regionprops(Omega)[i-1])
                n = len(regions)
                area = 0
                sub_img = np.zeros_like(Omega, dtype = bool)
                for reg in regions:
                    area += reg.area
                    sub_img += Omega == reg.label
                coords = regions[0].coords
                for i in range(1, n):
                      coords = np.append(coords, regions[i].coords, axis = 0)
                delta_s = matrix(np.array([[x[0]**2, x[1]**2, 2*x[0]*x[1], x[0], x[1], 1] for x in coords]), (area, 6))
                s = delta_s * theta[k]
                for i,pixel in enumerate(coords ,start= 0):
                      if s[i]>=0:
                            segmentation[pixel[0],pixel[1]] = 1
    return segmentation