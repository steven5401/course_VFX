from scipy import ndimage
from scipy.misc import imsave
import numpy as np
import re
def CC(name = 'prtn', focal_name = 'pano.txt', num=18):#name is name of first image, num is number of images
    focals = []
    with open(focal_name) as f:
        lines = f.readlines()
        for line in lines:
            a = line.split()
            if (len(a) == 1 and re.search('[0-9]', a[0][-1])):
                focals.append(float(a[0]))
    #focals = np.array(focals)
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
        imsave('c_' + name + str(n) + '.jpg', result)
CC()
print 'finish'