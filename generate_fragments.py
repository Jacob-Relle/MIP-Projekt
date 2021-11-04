import numpy as np
import itertools
import networkx as nx 

from skimage import img_as_ubyte, filters
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
from skimage.morphology import disk
from skimage.measure import label, regionprops, euler_number
from skimage.future.graph import RAG

from tqdm import tqdm

def generate_fragments(img, std_dev, int_threshold, min_seed_dist, max_search_depth, max_frag_dist):
    """
    Create fragments and prototype set out of watershed transformation of image.
    
    Parameters
    ----------
    img: 
    std_dev: float
        standard deviation for gaussian filter

    int_threshold: float
        intensity threshold between 0 and 1

    min_seed_distance: float
        distance between seeds, determines how big single fragments should be

    max_search_depth: float
        positive number, determines how many fragments should be pieced together to create an element of a prototype list

    max_frag_dist: float

        positive number, ...
    
    Returns
    -------
    (fragemnts, PrototypeList)
        fragments: ndarray, a labeled matrix of the same type and shape as image.
            Each label depicts a fragment of the input picture ``image``.

        ProtoytpeList: set of frozensets
            each is a prototype set consisting of labeles
    """
    
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
    
    
    img = img_as_ubyte(img)
    #smooth image with gausian filter
    g = gaussian_filter(img,std_dev)
    
    #Create Delta Ball
    B = disk(min_seed_dist)
    #Set the markers in the Image according to the formula (P is an image)
    P = (g==filters.rank.maximum(g,B)) & ((1-int_threshold)*g >= filters.rank.minimum(g,B))
    #Fuse adjacient markers and label each group of markers
    g_markers = label(P)
    #Get the regions of the markers to access centroid later
    PI = regionprops(g_markers)

    #Create Watershed regions
    fragments = watershed(255-g,markers=g_markers)
    #Create the adjacency graph of the labeled image
    G = RAG(fragments,connectivity=2)

    #Remove edges if centroids are to far away
    for edge in G.edges():
        if np.linalg.norm(np.array(PI[edge[0]-1].centroid) - np.array(PI[edge[1]-1].centroid)) > max_frag_dist:
            G.remove_edge(edge[0],edge[1])

    ProtoytypeSet = set()
    #iterate over all conected components (cc)
    for nodes_in_cc in tqdm(nx.algorithms.connectivity.edge_kcomponents.k_edge_subgraphs(G,1)):
        #Set the subgraph of curent cc
        H = G.subgraph(nodes_in_cc)
        #Loop over the nodes of the subgraph
        for v in nodes_in_cc:
            #Add the isolated region of the curent node
            if euler_number(fragments == v) == 1:
                ProtoytypeSet.add(frozenset([v]))
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
                        if frozenset(node_subset) not in ProtoytypeSet:
                            if nx.is_connected(H.subgraph(node_subset)):
                                #check that the merge of the regions is simply connected
                                sub_img = np.zeros_like(fragments, dtype = bool)
                                for region_label in node_subset:
                                    sub_img += fragments == region_label
                                if euler_number(sub_img) == 1:
                                     ProtoytypeSet.add(frozenset(node_subset))

    return fragments,ProtoytypeSet