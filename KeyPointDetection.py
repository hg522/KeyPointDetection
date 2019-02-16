# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:01:59 2018

@author: Himanshu Garg
UBName : hgarg
UB Person : 50292195
"""
import cv2
import numpy as np
import math

octave_sigma_map = {
            "o1":[1/math.sqrt(2),1,math.sqrt(2),2,2*math.sqrt(2)],
            "o2":[math.sqrt(2),2,2*math.sqrt(2),4,4*math.sqrt(2)],
            "o3":[2*math.sqrt(2),4,4*math.sqrt(2),8,8*math.sqrt(2)],
            "o4":[4*math.sqrt(2),8,8*math.sqrt(2),16,16*math.sqrt(2)]
        }


def flrange(start, stop, step):
    l = []
    st = start
    while st < stop:
        l.append(st)
        st+=step
    return l

def getProductSum(l1,l2,N):
    return sum([l1[k]*l2[k] for k in range(N)])


def convertToList(nparray):
    nlist = []
    for ind,i in enumerate(nparray):
        temp = list(i)
        temp = [x*1.0 for x in list(i)]
        nlist.append(temp)
    return nlist

def copyImage(img):
    im = []
    for row in img:
        temp=[]
        for col in row:
            temp.append(col)
        im.append(temp)
    return im

def doPadding(img,pd):
    rimg = copyImage(img)
    for row in rimg:
        for i in range(pd):
            row.insert(0,row[0])
            row.append(row[len(row)-1])
    #zeroarray = [0.0]*len(rimg[0])
    for i in range(pd):
        rimg.insert(0,rimg[0])
        rimg.append(rimg[len(rimg)-1])
    return rimg

def removePadding(img,pd):
    rimg = copyImage(img)
    for i in range(pd):
        del rimg[i]
        del rimg[len(rimg) - i -1]
    for row in rimg:
        for i in range(pd):
            del row[i]
            del row[len(row) - i - 1]
    return rimg

def get1DGaussianKernal(sigma,size):
    c1 = 2*sigma*sigma
    c2 = c1*math.pi
    kernel = []
    N = size/2
    nlist = flrange(-N,N+1,1)
    step = 0.001

    for i in nlist:
        j = i
        itr = i + 1
        summation = 0
        if itr <= nlist[len(nlist)-1]:
            while j < itr:
                summation+=math.exp(-((j**2)/c1))/c2
                j+=step
            kernel.append(summation)
            summation = 0
            
    kernel = [val/sum(kernel) for val in kernel]
    return kernel


def flipKernel(kernel):
    temp = kernel[::-1]
    k = []
    for i in temp:
        k.append(i[::-1])
    return k
'''
def calculateConvolution(kernel, image):
    fimg = []
    N = len(kernel)
    pd = math.floor(N/2)
    for indxr,row in enumerate(image):
        if indxr < pd or indxr > len(image) - pd - 1:
            continue
        temp = []
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            SUM = 0
            for l in range(N):
                r = image[indxr - pd + l][indc - pd:indc + pd + 1]
                SUM+= getProductSum(r,kernel[l],N)
            temp.append(SUM)
        fimg.append(temp)
    return fimg
'''
  
    
def doXYConvolution(kernel,image,direction):
    fimg = []
    img = copyImage(image)
    
    if direction == "Y" or direction == "y" :
        img = [list(i) for i in zip(*img)]
    N = len(kernel)
    pd = math.floor(N/2)
    for indxr,row in enumerate(img):
        if indxr < pd or indxr > len(img) - pd - 1:
            continue
        temp = []
        for indc,col in enumerate(row):
            if indc < pd or indc > len(row) - pd - 1:
                continue
            r = img[indxr][indc - pd:indc + pd + 1]
            SUM = getProductSum(r,kernel,N)
            temp.append(SUM)
        fimg.append(temp)
     
    if direction == "Y" or direction == "y" :
        fimg = [list(i) for i in zip(*fimg)]
    return fimg

    
def getAllGaussianConv(oct_name,orgImg, ksize):
    gImgArr = []
    sigMap = octave_sigma_map[oct_name]
    for sigma in sigMap:
        #k = get2DGaussianKernel(get1DGaussianKernal(sigma,ksize))
        #k = create_kernel(sigma)
        k = get1DGaussianKernal(sigma,ksize)
        img = doXYConvolution(k,orgImg,"x")
        img = doPadding(img,math.floor(ksize/2))
        img = doXYConvolution(k,img,"y")
        #img = calculateConvolution(k,orgImg)
        
        gImgArr.append(np.asarray(img,dtype=np.float32))
        
    return gImgArr

def multiplyMatrices(m1,m2):
      return  [[sum(a*b for a,b in zip(m1_row,m2_col)) for m2_col in zip(m2)] for m1_row in m1]

def get2DGaussianKernel(k1):
    k2 = [[i] for i in k1]
    kernel = multiplyMatrices(k2,k1)
    '''
    sum = 0
    for r in kernel:
        for c in kernel[0]:
            sum = sum + c
    print(sum)
    ''' 
    return kernel

def getHalfScaleImg(img):
    himg = []
    rows = math.floor(len(img)/2)
    cols = math.floor(len(img[0])/2)
    for i in range(rows):
        temp = []
        for j in range(cols):
            temp.append(img[2*i][2*j])
        himg.append(temp)
    return himg

def findMaxMinOf2DList(img):
    maxArr = []
    minArr = []
    for r in img:
        maxArr.append(max(r)) 
        minArr.append(min(r))
    mx = max(maxArr)
    mn = min(minArr)
    return mx,mn


def getDOG(img1,img2):
    
    dimg = img1 - img2
    return np.asarray(dimg,dtype=np.float32)
    
    '''
    dimg = []
    if len(img1) == len(img2) and len(img1[0]) == len(img2[0]):
        dimg = [[(c1-c2) for c1,c2 in zip(r1,r2)] for r1,r2 in zip(img1,img2)]
        #mx,mn = findMaxMinOf2DList(dimg)
        #dimg = [[c/max(mx,abs(mn)) for c in r] for r in dimg]
        #print("max,min:",mx,mn)
    else:
        print("images should be same size")
    return np.asarray(dimg,dtype=np.uint8)
    '''

def getAllDOGs(imgArr):
    dogArr = []
    size = len(imgArr)
    for i in range(size-1):
        temp = getDOG(imgArr[i],imgArr[i+1])
        dogArr.append(temp)
    return dogArr


def getKeyPoints(img1,img2,img3):
    #kimg = copyImage(img2)
    rows = len(img2)
    cols = len(img2[0])
    kimg = [[0 for c in range(cols)] for r in range(rows)]
    #mx,mn = findMaxMinOf2DList(img2)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            
            temp = []
            val = img2[i][j]
            im1r1 = img1[i-1][j-1:j+2]
            for y in im1r1:
                temp.append(y)
            im1r2 = img1[i][j-1:j+2]
            for y in im1r2:
                temp.append(y)
            im1r3 = img1[i+1][j-1:j+2]
            for y in im1r3:
                temp.append(y)
            im2r1 = img2[i-1][j-1:j+2]
            for y in im2r1:
                temp.append(y)
            im2r2 = [img2[i][j-1],img2[i][j+1]]
            for y in im2r2:
                temp.append(y)
            im2r3 = img2[i+1][j-1:j+2]
            for y in im2r3:
                temp.append(y)
            im3r1 = img3[i-1][j-1:j+2]
            for y in im3r1:
                temp.append(y)
            im3r2 = img3[i][j-1:j+2]
            for y in im3r2:
                temp.append(y)
            im3r3 = img3[i+1][j-1:j+2]
            for y in im3r3:
                temp.append(y)
            mx = max(temp)
            mn = min(temp)
            if val > mx or val < mn:
                #if val == 0:
                #    val = mx
                kimg[i][j] = val
            else:
                kimg[i][j] = 0
    kimg,kps = removeInvalidKeyPoints(kimg)
    
    return np.asarray(kimg,dtype=np.uint8),kps

def getAllKeyPoints(dogArr):
    i = 0
    size = len(dogArr)
    kpArr = []
    keyArr = []
    if size >= 3:
        for i in range(size-2):
            im,kps = getKeyPoints(dogArr[i],dogArr[i+1],dogArr[i+2])
            kpArr.append(im)
            keyArr.append(kps)
    return kpArr,keyArr


def removeInvalidKeyPoints(img):
    thimg = copyImage(img)
    mx,mn = findMaxMinOf2DList(thimg)
    Ithreshold = 0.15
    h = len(thimg)
    w = len(thimg[0])
    kps = []
    for i in range(h):
        for j in range(w):
            temp = []
            if thimg[i][j] <= Ithreshold * mx:
                thimg[i][j] = 0
            else:
                thimg[i][j] = 255
                temp.append(i)
                temp.append(j)
                kps.append(temp)
    return thimg,kps
    
def drawKeypoints(img,keypts,scale,kptArr):
    for i in keypts:
        for j in i:
            img[scale*j[0]][scale*j[1]] = 255
            x = scale*j[0]
            y = scale*j[1]
            tmp = []
            tmp.append(x)
            tmp.append(y)
            kptArr.append(tmp)
    return img

def displayUint8(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, np.asarray(img,dtype=np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def displayFloat32(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, np.asarray(img,dtype=np.float32))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def displayDef(name,img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, np.asarray(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def displayArrayOfImagesUint8(imgArr):
    for i in imgArr:
        displayUint8('', i)
        
def displayArrayOfImagesFloat32(imgArr):
    for i in imgArr:
        displayFloat32('', i)
        
def displayArrayOfImagesDef(imgArr):
    for i in imgArr:
        displayDef('', i)

def writeOctaveGaussImages(imgArr, number):
    for index,i in enumerate(imgArr):
        name = "Octave_" + str(number) + "_Gaussian_" + str(index+1) + ".png"
        cv2.imwrite(name,np.asarray(i,dtype=np.uint8))
        print("****" + name + "****")
        print("height:",len(i))
        print("width:",len(i[0]))

def writeOctaveDogImages(imgArr, number):
    for index,i in enumerate(imgArr):
        name = "Octave_" + str(number) + "_DOG_" + str(index+1) + ".png"
        cv2.imwrite(name,np.asarray(i,dtype=np.uint8))
        print("****" + name + "****")
        print("height:",len(i))
        print("width:",len(i[0]))       
    
def writeOctaveKeypointImages(imgArr, number):
    for index,i in enumerate(imgArr):
        name = "Octave_" + str(number) + "_Keypoints_" + str(index+1) + ".png"
        cv2.imwrite(name,np.asarray(i,dtype=np.uint8))
        print("****" + name + "****")
        print("height:",len(i))
        print("width:",len(i[0])) 
        
ksize = 7   
threshold = 1 
padding = math.floor(ksize/2)
img = cv2.imread("task2.jpg",0)
s1org = convertToList(img)
s1orgImgList = doPadding(s1org,padding)
oct1Imgs = getAllGaussianConv("o1",s1orgImgList,ksize)
#writeOctaveGaussImages(oct1Imgs,1)
oct1DOGs = getAllDOGs(oct1Imgs)
#writeOctaveDogImages(oct1DOGs,1)
oct1KPs,oct1pts = getAllKeyPoints(oct1DOGs)
#writeOctaveKeypointImages(oct1KPs,1)

s2org = getHalfScaleImg(s1org)
s2orgImgList = doPadding(s2org,padding)
oct2Imgs = getAllGaussianConv("o2",s2orgImgList,ksize)
#writeOctaveGaussImages(oct2Imgs,2)
oct2DOGs = getAllDOGs(oct2Imgs)
#writeOctaveDogImages(oct2DOGs,2)
oct2KPs,oct2pts = getAllKeyPoints(oct2DOGs)
#writeOctaveKeypointImages(oct2KPs,2)

s3org = getHalfScaleImg(s2org)
s3orgImgList = doPadding(s3org,padding)
oct3Imgs = getAllGaussianConv("o3",s3orgImgList,ksize)
#writeOctaveGaussImages(oct3Imgs,3)
oct3DOGs = getAllDOGs(oct3Imgs)
#writeOctaveDogImages(oct3DOGs,3)
oct3KPs,oct3pts = getAllKeyPoints(oct3DOGs)
#writeOctaveKeypointImages(oct3KPs,3)

s4org = getHalfScaleImg(s3org)
s4orgImgList = doPadding(s4org,padding)
oct4Imgs = getAllGaussianConv("o4",s4orgImgList,ksize)
#writeOctaveGaussImages(oct4Imgs,4)
oct4DOGs = getAllDOGs(oct4Imgs)
#writeOctaveDogImages(oct4DOGs,4)
oct4KPs,oct4pts = getAllKeyPoints(oct4DOGs)
#writeOctaveKeypointImages(oct4KPs,4)

fkptArr = []
finalImg = drawKeypoints(img,oct1pts,1,fkptArr)
finalImg = drawKeypoints(img,oct2pts,2,fkptArr)
finalImg = drawKeypoints(img,oct3pts,4,fkptArr)
finalImg = drawKeypoints(img,oct4pts,8,fkptArr)

#cv2.imwrite('Keypoints.png',np.asarray(finalImg,dtype=np.uint8))
print("Total Keypoints:",len(fkptArr))
sortedfkptArr = sorted(fkptArr,key=lambda l:l[1])
sortedfkptArr = sortedfkptArr[0:5]
print("Left Most 5 Keypoints:",sortedfkptArr)


displayUint8('Original',s1org)
displayArrayOfImagesUint8(oct1Imgs)
displayArrayOfImagesFloat32(oct1DOGs)
displayArrayOfImagesUint8(oct1KPs)

displayUint8('Scale 2 Original',s2org)

displayArrayOfImagesUint8(oct2Imgs)
displayArrayOfImagesFloat32(oct2DOGs)
displayArrayOfImagesUint8(oct2KPs)

displayUint8('Scale 3 Original',s3org)

displayArrayOfImagesUint8(oct3Imgs)
displayArrayOfImagesFloat32(oct3DOGs)
displayArrayOfImagesUint8(oct3KPs)

displayUint8('Scale 4 Original',s4org)

displayArrayOfImagesUint8(oct4Imgs)
displayArrayOfImagesFloat32(oct4DOGs)
displayArrayOfImagesUint8(oct4KPs)
displayUint8('',finalImg)

