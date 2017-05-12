from scipy import ndimage
from scipy.misc import imsave
import numpy as np
import re
import feature
import random
import math
import sys

def CC(num = 18, name = 'prtn', focal_name = 'pano.txt'):#name is name of first image, num is number of images
    focals = []
    with open(focal_name) as f:
        lines = f.readlines()
        for line in lines:
            a = line.split()
            if (len(a) == 1 and re.search('[0-9]', a[0][-1])):
                focals.append(float(a[0]))
    #focals = np.array(focals)
    imgs_cylindral=[]
    for n in xrange(num):
        source = ndimage.imread(name + str(n) + '.jpg')
        x_center = source.shape[0] / 2
        y_center = source.shape[1] / 2
        result = np.zeros(source.shape)
        for x in xrange(source.shape[0]):
            for y in xrange(source.shape[1]):
                x_center_as_origin = x - x_center
                y_center_as_origin = -y + y_center
                new_x_cao = int(np.ceil(focals[n] * np.arctan(x_center_as_origin/focals[n])))
                new_y_cao = int(np.ceil(focals[n] * y_center_as_origin / np.sqrt(x_center_as_origin*x_center_as_origin + focals[n]*focals[n])))
                new_x = new_x_cao + x_center
                new_y = -new_y_cao + y_center
                #print new_x, new_y
                result[new_x][new_y] = source[x][y]
        #imsave('c_' + name + str(n) + '.jpg', result)
        imgs_cylindral.append(result)
    return imgs_cylindral

def ToGray(img_list):
    y=[]
    for x in img_list:
        y.append(0.114*x[..., 0] + 0.587*x[..., 1] + 0.299*x[..., 2])
    return y

def RANSAC(matchFP_list):
    '''
    return x shift amount, y shift amount, inlier list=[p1x, p1y, p2x, p2y]
    '''
    sample_count = 10
    Preal_inliers = 0.5
    threshold = 1000
    Psuccess = 0.99
    dx = 0.0
    dy = 0.0
    # Calculate the iteration of the Algorithm
    K = int(math.log(1-Psuccess)/math.log(1-(Preal_inliers)**6))
    RANSAC_iter = 0
    matched_RANSAC =[]
    matched = []
    match_count = 0
    print "Total iter: ", K
    for k in range(K):
        #print "RANSAC_iter: ", k
        shift_x = 0.0
        shift_y = 0.0
        sample_n = 0
        match_count_temp = 0
        for n in range(sample_count):
            sample = random.randint(0,len(matchFP_list)-1)
            [fp1x, fp1y, fp2x, fp2y] = matchFP_list[sample]
            shift_x += fp2x - fp1x
            shift_y += fp2y - fp1y
            matched_RANSAC.append([fp1x, fp1y, fp2x, fp2y])
        #print "shift : ", shift_y, shift_x
        shift_x = shift_x /  sample_count
        shift_y = shift_y / sample_count
        #print "shift : ", shift_y, shift_x

        for idx in range(len(matchFP_list)):
            [fp1x, fp1y, fp2x, fp2y] = matchFP_list[idx]
            #print np.square(fp1.y - fp2.y - shift_y) + np.square(fp1.x - fp2.x - shift_x)
            if np.square(fp2y - fp1y - shift_y) + np.square(fp2x - fp1x - shift_x) < threshold:
                matched_RANSAC.append([fp1x, fp1y, fp2x, fp2y])
                match_count_temp+=1

        if match_count_temp > match_count:
            match_count = match_count_temp
            dx = shift_x
            dy = shift_y
            matched = []
            matched = matched_RANSAC
            matched_RANSAC =[]

        else:
            matched_RANSAC =[]
        #print "dx : ", dx , " dy : ", dy, " inlier: ", match_count

    return dx , dy, matched
def main(num = 18,  name = 'prtn', focal_name = 'pano.txt', output_name = 'out.jpg'):
    imgs_cylindral = CC(num, name, focal_name);#convert to cylindral coordinate
    imgs_cylindral_gray = ToGray(imgs_cylindral)#to gray scale
    total_shift_list = []
    panorama_h_len = 0
    panorama_w_len = 0
    for i in xrange(len(imgs_cylindral_gray) - 1):#pairwise matching
        feature_points1 = feature.harris_detector(imgs_cylindral_gray[i])
        feature_description1 = feature.harris_descriptor(imgs_cylindral_gray[i], feature_points1)
        feature_points2 = feature.harris_detector(imgs_cylindral_gray[i + 1])
        feature_description2 = feature.harris_descriptor(imgs_cylindral_gray[i + 1], feature_points2)
        matches = feature.matching(feature_description1, feature_description2)
        match_points1 = feature_points1[matches[:, 0], 0:2]#only xy coordinate
        match_points2 = feature_points2[matches[:, 1], 0:2]
        match_list = np.hstack((match_points1, match_points2))
        [shift_x, shift_y, matches_ransac] = RANSAC(match_list)
        total_shift_list.append([shift_x, shift_y])
    total_shift_list = np.array(total_shift_list).astype(int)
    print 'total_shift_list size:', len(total_shift_list)#should be num - 1
    print total_shift_list#shift_x is horizontal direction
    panorama_h_len = imgs_cylindral[-1].shape[0]
    panorama_w_len = imgs_cylindral[-1].shape[1]
    '''
    start stitch
    '''
    panorama_w_len += int(np.ceil(np.sum(total_shift_list, axis=0)[0]))
    panorama_h_len += abs(int(2*np.ceil(np.sum(total_shift_list, axis=0)[1])))
    print 'start stitching, panorama size(h,w):', panorama_h_len, panorama_w_len
    panorama = np.zeros((panorama_h_len, panorama_w_len, 3))#determine size of panorama
    accumulate_shift_w = 0
    accumulate_shift_h = -abs(int(np.ceil(np.sum(total_shift_list, axis=0)[1])))
    overlap_width = 0
    #test = np.zeros((800,800,3))
    for i in xrange(len(imgs_cylindral)):
        if i >= 1:
            overlap_width += imgs_cylindral[i - 1].shape[1] - total_shift_list[i - 1][0]
        for j in xrange(int(imgs_cylindral[i].shape[0])):
            for k in xrange(int(imgs_cylindral[i].shape[1])):
                if np.array_equal(panorama[accumulate_shift_h - j,accumulate_shift_w - k],np.zeros(3)):
                    panorama[accumulate_shift_h - j, \
                             accumulate_shift_w - k, :] = imgs_cylindral[i][-j, -k, :]
                    '''
                    if i == 1:
                        test[-j, -k, :] = imgs_cylindral[i][-j, -k, :]
                    '''
                elif np.array_equal(imgs_cylindral[i][-j, -k], np.zeros(3)):
                    continue
                else:
                    if overlap_width == 0:
                        print i, k, overlap_width
                        assert overlap_width != 0
                    t = k / float(overlap_width)
                    if not t <= 1:
                        print i, k, overlap_width
                        assert t <=1
                    panorama[accumulate_shift_h - j, \
                             accumulate_shift_w - k, :] = t*imgs_cylindral[i][-j, -k, :] + (1-t)*panorama[accumulate_shift_h - j, accumulate_shift_w - k, :]
                    #test[-j, -k, :] = t*imgs_cylindral[i][-j, -k, :]
                    #print k, overlap_width, t     
        if i == len(imgs_cylindral) - 1:
            break
        accumulate_shift_h -= total_shift_list[i][1]#I think that vertical shift should be less than 4 pixels
        accumulate_shift_w -= total_shift_list[i][0]

    imsave(output_name, panorama)
    #imsave('t.jpg', test)
    print 'finish'

name_ = sys.argv[1]
num_ = int(sys.argv[2])
output_name_ = 'out.jpg'
focal_name_ = 'pano.txt'
if len(sys.argv) > 3:
    output_name_ = str(sys.argv)[3]
if len(sys.argv) > 4:
    focal_name_ = str(sys.argv)[4]
main(name = name_, num = num_, output_name = output_name_, focal_name = focal_name_)