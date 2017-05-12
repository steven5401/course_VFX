#import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.misc import imresize
from scipy.spatial.distance import cdist

def getResponse(img, sigma_d=1.0, sigma_i=1.5, k = 0.04):
    # 1. Compute x and y derivatives of image
    dy, dx = np.gradient(img)
    GI = gaussian_filter(img, sigma_d)
    Ix = dx * GI
    Iy = dy * GI

    # 2. Compute products of derivatives at every pixel
    Ix2 = Ix**2
    Iy2 = Iy**2
    Ixy = Ix * Iy

    # 3. Compute the sums of the products of derivatives at each pixel
    Sx2 = gaussian_filter(Ix2, sigma_i)
    Sy2 = gaussian_filter(Iy2, sigma_i)
    Sxy = gaussian_filter(Ixy, sigma_i)

    # 4. Compute determinant and trace of the matrix M at each pixel 
    det = (Sx2 * Sy2) - (Sxy**2)
    trace = Sx2 + Sy2

    # 5. Compute the response of the detector at each pixel
    R = det - k*(trace**2)

    return clipEdge(R)

def nms(img, num = 500, threshold = 10):
    # clip by the threshold
    img = np.where( img > threshold, img, 0)
    # non-maximum suppression by maximum filter in 5x5 regions
    Imax = maximum_filter(img, (5,5))
    img = img * (img == Imax)

    # sort points by strength and find positions
    idx = np.argsort(img.flatten())[ : :-1]
    #print(idx)
    idx = idx[:num]
    y = (idx / Imax.shape[1])
    x = (idx % Imax.shape[1])
    # concatenate positions and values
    fts = np.vstack((x, y, img.flatten()[idx])).transpose()

    return fts

def clipEdge(img):
    # neglect points near the border
    img[:20, :] = 0
    img[-20:, :] = 0
    img[:, :20] = 0
    img[:, -20:] = 0

    return img

def normalize(X):
    X_mean = np.mean(X)
    X_std = np.std(X)
    X = (X - X_mean) / (X_std + 1e-7)

    return X

def harris_descriptor(img, fts, r=3):
    # smooth the image with a gaussian filter
    GI = gaussian_filter(img, 4)
    w = r*2+1
    rs = (w*5-1)/2

    # create a numpy array to hold feature descriptors
    descriptor = np.zeros((fts.shape[0], w**2), dtype=float)
    #print(descriptor.shape)

    for i in range(fts.shape[0]):
        # get the position of the feature
        x, y = fts[i, 0:2].astype(int)
        #print(type(x))
        #print(type(r))

        # get the patch of the feature
        patch = img[y-rs:y+rs+1, x-rs:x+rs+1]
        #patch = patch.flatten()

        # normalize the descriptor
        patch = normalize(patch)
        patch = imresize(patch, 0.2)
        #print(patch.shape)
        #patch = patch.reshape(w,w)

        descriptor[i] = patch.flatten()

    return descriptor

def harris_detector(img):
    '''
    return a list which shape is (feature points num, 3)
    each row contains x coordinate, y coordinate, a score that indicates how good the points is as a feature
    '''
    img = img.astype(float)
    R = getResponse(img)

    # 6. Threshold on value of R; compute nonmax suppression.
    fts = nms(R) 

    # return descriptors
    #return harris_descriptor(img, fts)
    return fts

def matching(d1, d2):
    '''
    return a list which shape is (match points num, 2)
    each row contains index in d1 and index in d2 of one match
    '''
    dists = cdist(d1, d2)
    print(dists.shape)
    sort_index = np.argsort(dists, 1)
    #print(sort_index.shape)

    # find best and second matches
    best_dist_idx = sort_index[:, 0]
    second_dist_idx = sort_index[:, 1]
    best_dist = np.zeros(best_dist_idx.shape)
    second_dist = np.zeros(second_dist_idx.shape)
    #print(dist1_idx.shape)
    for i in range(dists.shape[0]):
        best_dist[i] = dists[i, best_dist_idx[i]]
        second_dist[i] = dists[i, second_dist_idx[i]]
    print(best_dist.shape)
    #print(best_dist)

    # get score by best / second distance
    score = best_dist / second_dist.mean()
    #print(score.shape)

    # find the indices of the best Matches by threshold
    best_match = np.argwhere(score < 0.5)
    second_match = best_dist_idx[best_match]
    #print(best_match)
    print('Best match num :'),
    print(best_match.shape[0])
    #print(second_match.shape)

    # put the matches in a single array and return as type int
    matches = np.hstack([best_match, second_match])
    return matches.astype(int)
