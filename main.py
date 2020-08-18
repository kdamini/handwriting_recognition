import os
os.chdir('C:\Users\kdami\Downloads')
from sklearn import svm
clf = svm.SVC(gamma = 0.001, C= 100)
from scipy.misc import imread # using scipy's imread
import numpy as np
from skimage.transform import resize

def boundaries(binarized,axis):
    # variables named assuming axis = 0; algorithm valid for axis=1
    # [1,0][axis] effectively swaps axes for summing
    rows = np.sum(binarized,axis = [1,0][axis]) > 0
    rows[1:] = np.logical_xor(rows[1:], rows[:-1])
    change = np.nonzero(rows)[0]
    ymin = change[::2]
    ymax = change[1::2]
    height = ymax-ymin
    too_small = 10 # real letters will be bigger than 10px by 10px
    ymin = ymin[height>too_small]
    ymax = ymax[height>too_small]
    return zip(ymin,ymax)

def separate(img):
    orig_img = img.copy()
    print orig_img.shape
    pure_white = 255
    white = np.max(img)
    black = np.min(img)
    thresh = (white+black)/2.0
    binarized = img<thresh
    row_bounds = boundaries(binarized, axis = 0) 
    cropped = []
    for r1,r2 in row_bounds:
        img = binarized[r1:r2,:]
        col_bounds = boundaries(img,axis=1)
        rects = [r1,r2,col_bounds[0][0],col_bounds[0][1]]
        cropped.append(np.array(orig_img[rects[0]:rects[1],rects[2]:rects[3]]/pure_white))
    
    return cropped



def partition(data, target, p):#p represents percentage, so p=50 means 50%
    n= len(target)
    m= p*n/100
    
    train_data = data[:m,:]
    train_target = target[:m]
    test_data = data[m:]
    test_target = target[m:]
    return train_data, train_target, test_data, test_target

s=(30,100)
data= np.zeros(s)

i=0
big_t = imread("t.jpg", flatten = True) # flatten = True converts to grayscale


imgt = separate(big_t) # separates big_img (pure white = 255) into array of little images (pure white = 1.0)

for img in imgt:
    img = resize(img, (10,10))
    img= img.flatten()
    data[i]=img
    i+=3

i=2
big_h = imread("h.jpg", flatten = True) # flatten = True converts to grayscale


imgh = separate(big_h) # separates big_img (pure white = 255) into array of little images (pure white = 1.0)
for img in imgh:
    img = resize(img, (10,10))
    img= img.flatten()
    data[i]=img
    i+=3
    
big_b = imread("b.jpg", flatten = True) # flatten = True converts to grayscale
i=1

imgb = separate(big_b) # separates big_img (pure white = 255) into array of little images (pure white = 1.0)
for img in imgb:
    img = resize(img, (10,10))
    img= img.flatten()
    data[i]=img
    i+=3

 
target=[]
for i in range(len(imgh)):
    target.append(0)
    target.append(1)
    target.append(2)
target= np.array(target)



result= partition(data, target, 30)
clf.fit(result[0],result[1])
n= len(result[3])
print "Predicted:" + str(clf.predict(result[2]))
print "Truth:" + str(result[3])
print 'Accuracy:' + str(sum(clf.predict(result[2])==result[3])*1.0*100/n)
