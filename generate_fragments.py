import numpy as np
import matplotlib.pyplot as plt
import itertools
import networkx as nx 

from skimage import img_as_ubyte, util, filters
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.morphology import disk
from skimage.measure import label, regionprops, euler_number
from skimage.future.graph import RAG

from tqdm import tqdm

def generate_fragments(img,std_dev,int_threshold,min_seed_dist,max_search_depth,max_frag_dist, background = 0):
    img = img_as_ubyte(img)
    #smooth image with gausian filter
    g = gaussian_filter(img,std_dev)
    #flatten the lower background percentile of the gaussian filter to 0 (for background)
    g[g < np.percentile(g, background)] = 0
    
    #Fehlerabfangen
    if std_dev < 0:
        raise ValueError("smoothing strength needs to be positive")
    if int_threshold > 1 or int_threshold < 0:
        raise ValueError("relative intensity threshold needs to be between 0 and 1")
    if min_seed_dist < 1:
        raise ValueError("min seed distance needs to be >= 1")
    if max_search_depth < 0:
        raise ValueError("maximum search error needs to be positive")
    if max_frag_dist < 0:
        raise ValueError("maximum fragments distance needs zo be positive")
    if background > 100 or background < 0:
        raise ValueError('background is a percentage value between 0 and 100')
        
    #Create Delta Ball
    B = disk(min_seed_dist)
    #Set the markers in the Image according to the formula (P is an image)
    P = (g==filters.rank.maximum(g,B)) & ((1-int_threshold)*g >= filters.rank.minimum(g,B))
    #Fuse adjacient markers and label each group of markers
    g_markers = label(P)
    #Get the regions of the markers to access centroid later
    PI = regionprops(g_markers)

    #Create Watershed regions
    Omega = watershed(util.invert(g),markers=g_markers)
    #Create the adjacency graph of the labeled image
    G = RAG(Omega,connectivity=2)

    #Remove edges if centroids are to far away
    for edge in G.edges():
        if np.linalg.norm(np.array(PI[edge[0]-1].centroid) - np.array(PI[edge[1]-1].centroid)) >= max_frag_dist:
            G.remove_edge(edge[0],edge[1])

    S = set()
    #iterate over all conected components (cc)
    for nodes_in_cc in tqdm(nx.algorithms.connectivity.edge_kcomponents.k_edge_subgraphs(G,1)):
        #Set the subgraph of curent cc
        H = G.subgraph(nodes_in_cc)
        #Loop over the nodes of the subgraph
        for v in nodes_in_cc:
            #Add the isolated region of the curent node
            if euler_number(Omega == v) == 1:
                S.add(frozenset([v]))
            #Loop over the distance from the curent node within the graph
            for distance in range(1,max_search_depth+1):
                #Get a dictonary containg the distance from curent node
                distance_from_origin = nx.single_source_shortest_path_length(H, v,cutoff=distance)
                #Loop over the amount of nodes
                for node_amount in range(distance,len(distance_from_origin.values())):
                    #Loop over all subsets of nodes that contain v and contain node_amount of nodes
                    for node_subset in itertools.combinations([node for node in distance_from_origin.keys()], node_amount):
                        node_subset = set([v]).union(set(node_subset))
                        #Check other requirments and add the node set to S if they are fullfilled
                        if frozenset(node_subset) not in S:
                            if nx.is_connected(H.subgraph(node_subset)):
                                #check that the merge of the regions is simply connected
                                sub_img = np.zeros_like(Omega, dtype = bool)
                                for region_label in node_subset:
                                    sub_img += Omega == region_label
                                if euler_number(sub_img) == 1:
                                     S.add(frozenset(node_subset))

    return Omega,S