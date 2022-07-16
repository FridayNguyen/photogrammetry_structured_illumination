import numpy as np
from skimage import io
from scipy.spatial import Delaunay

from collections import defaultdict

def gray_to_num(gray_string):
    out = gray_string[0]
    
    for i in range(1, len(gray_string)):
        if gray_string[i] == out[-1]:
            out += '0'
        else:
            out += '1'
    return int(out,2)
    
    

def decode(imprefix,start,threshold):
    """
    Given a sequence of 20 images of a scene showing projected 10 bit gray code, 
    decode the binary sequence into a decimal value in (0,1023) for each pixel.
    Mark those pixels whose code is likely to be incorrect based on the user 
    provided threshold.  Images are assumed to be named "imageprefixN.png" where
    N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")
 
    Parameters
    ----------
    imprefix : str
       Image name prefix
      
    start : int
       Starting index
       
    threshold : float
       Threshold to determine if a bit is decodeable
       
    Returns
    -------
    code : 2D numpy.array (dtype=float)
        Array the same size as input images with entries in (0..1023)
        
    mask : 2D numpy.array (dtype=logical)
        Array indicating which pixels were correctly decoded based on the threshold
    
    """
    
    # we will assume a 10 bit code
    nbits = 10
    
    # don't forget to convert images to grayscale / float after loading them in
    imnames = np.arange(start,start+20)
    imnames = [imprefix + str(num).zfill(2) + '.png' for num in imnames]
    imgs = list()
    for fname in imnames:          
        I = io.imread(fname, as_gray=True)        
        
        #convert to float data type if necessary
        if (I.dtype == np.uint8):
            I = I.astype(float) / 256                
    
        #finally, store the array in our list of images
        imgs.append(I)                
    
    mask = np.ones(imgs[0].shape)
    
    binary_imgs = list()
    
    for i in range(0, len(imgs), 2):
        first = imgs[i]
        second = imgs[i+1]

        diff = first - second
        binary = np.where(diff > 0, 1, 0)
        thresh = threshold
        mask[np.absolute(diff) < thresh] = 0
        binary_imgs.append(binary)
        
    final_string = np.copy(binary_imgs[0].astype(str))
    
    for i in range(1, len(binary_imgs)):        
        final_string = np.core.defchararray.add(final_string,binary_imgs[i].astype(str))   
        
        
    final_string = np.vectorize(gray_to_num)(final_string)
    
    code = final_string
    
    return code,mask

def coorToColor(imprefix_color, pts2):
    im = io.imread(imprefix_color + '01.png', as_gray=False)        
    if (im.dtype == np.uint8):
        im = im.astype(float) / 256
        
    color = np.zeros((3,pts2.shape[1]))
    
    for i in range(pts2.shape[1]):
        pixel = im[pts2[1, i], pts2[0, i]]
        color[0, i] = pixel[0]
        color[1, i] = pixel[1]
        color[2, i] = pixel[2]
    
    return color

def getForegroundMask(imprefix_color, threshold):
    
    fore = io.imread(imprefix_color + '01.png', as_gray=True)        
    if (fore.dtype == np.uint8):
        fore = fore.astype(float) / 256
        
    back = io.imread(imprefix_color + '00.png', as_gray=True)        
    if (back.dtype == np.uint8):
        back = back.astype(float) / 256
        
    mask = np.zeros(fore.shape)
        
    diff = fore - back
    mask[np.absolute(diff) > threshold] = 1
    
    return mask

def smooth(points,triangles):
    points = points.T
    out = np.zeros(points.shape)

    neighbors = makeNeighborDict(triangles.simplices.copy())
    
    for i in range(len(points)):
        connected = np.array(list(neighbors[i]))

        mean = np.mean(points[connected],axis=0)
        out[i] = mean
        
    return out.T

def makeNeighborDict(tri):
    out = defaultdict(set)
    for face in tri:
        for vertex in face: # nested loop is fine here since this loops a max of 3
            for other_vertex in face: # again this is fine since there's only 2 other vertices in the face (please forgive me) 
                if vertex != other_vertex:  # a vertex cant be its own neighbor
                    out[vertex].add(other_vertex)
    return out


def processMesh(t_pts3, t_pts2L, t_pts2R, t_colors):

    boxlimits = np.array([-200,200,-2,19,-25,-10])
    trithresh = 0.5

    # # bounding box pruning
    bad_ind = list()
    for i in range(0,t_pts3.shape[1]):
        if t_pts3[0][i] < boxlimits[0] or t_pts3[0][i] > boxlimits[1]:
            bad_ind.append(i)
        elif t_pts3[1][i] < boxlimits[2] or t_pts3[1][i] > boxlimits[3]:
            bad_ind.append(i)
        elif t_pts3[2][i] < boxlimits[4] or t_pts3[2][i] > boxlimits[5]:
            bad_ind.append(i)
            
    t_pts3 = np.delete(t_pts3,bad_ind,1)
    t_pts2L = np.delete(t_pts2L,bad_ind,1)
    t_pts2R = np.delete(t_pts2R,bad_ind,1)
    t_colors = np.delete(t_colors,bad_ind,1)



    # triangulate the 2D points to get the surface mesh
    tri_raw = Delaunay(t_pts2L.T)
    tri = tri_raw.simplices.copy()
    t_pts3 = smooth(t_pts3,tri_raw)
    t_pts3 = smooth(t_pts3,tri_raw)

    neighbors = makeNeighborDict(tri)

    bad_ind = list()

    for vertex,neighbors_set in neighbors.items():
        for neighbor in neighbors_set:
            a = t_pts3.T[vertex]
            b = t_pts3.T[neighbor]
            edge_length = np.linalg.norm(a-b)
            if edge_length > trithresh:
                bad_ind.append(vertex)
                break

    t_pts3 = np.delete(t_pts3,bad_ind,1)
    t_pts2L = np.delete(t_pts2L,bad_ind,1)
    t_pts2R = np.delete(t_pts2R,bad_ind,1)
    t_colors = np.delete(t_colors,bad_ind,1)

    tri_raw = Delaunay(t_pts2L.T)
    tri = tri_raw.simplices.copy()

    # tokeep = np.arange(0, len(t_pts3))
    # tokeep = np.setdiff1d(tokeep, bad_ind)

    # map = np.zeros(t_pts3.shape[1])
    # map[tokeep] = np.arange(0,tokeep.shape[0])
    # tri = map[tri]

    return t_pts3, t_pts2L, t_pts2R, t_colors, tri