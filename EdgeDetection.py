import cv2
import matplotlib.pyplot as plt
import numpy as np

#Getting path to image and reading it using openCV    
path=input("Enter path to image: ")
img=cv2.imread(path)

#Converting image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Blurring the image to reduce the impact of noise
blur = cv2.GaussianBlur(gray, (3, 3), 0)

#converting image to numpy array
img_arr=np.array(blur)

#get size of image to define pad
r,c=img_arr.shape
pad=[0 for i in range(c)]

print(img_arr.shape)

#add padding above and below
l=list(img_arr)
l.insert(0,pad)
l.append(pad)
img_arr=np.array(l)

#add padding on the transpose (left and right)
transp=list(np.transpose(img_arr))
pad=[0 for i in range(r+2)]
transp.insert(0,pad)
transp.append(pad)
transp=np.array(transp)
img_arr=np.transpose(transp)

print(img_arr.shape)

#Defining the sobel kernels
sobel_x=[[-1/2,0,1/2],[-1,0,1],[-1/2,0,-1/2]]
sobel_y=[[-1/2,-1,-1/2],[0,0,0],[-1/2,-1,-1/2]]

r,c=img_arr.shape

#Convolve with sobel_x kernel
for i in range(r-3+1):
    for j in range(c-3+1):
        sum=0
        for k in range(3):
            for l in range(3):
                sum+=sobel_x[l][k]*img_arr[i+l][j+k]
        img_arr[i+1][j+1]=sum

#Convolve with sobel_y kernel
for i in range(c-3+1):
    for j in range(r-3+1):
        sum=0
        for k in range(3):
            for l in range(3):
                sum+=sobel_y[k][l]*img_arr[j+k][i+l]
                
plt.imshow(img_arr)
plt.show()





