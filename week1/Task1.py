import cv2
import matplotlib.pyplot as plt
import numpy as np


#Getting path to image and reading it using openCV
path=input("Enter path to image: ")
img=cv2.imread(path)


#Converting to RGB for comparison
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img=np.array(rgb)

#Converting image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 


#Blurring the image to reduce the impact of noise
blur = cv2.GaussianBlur(gray, (3, 3), 0)


#converting image to numpy array
img_arr=np.array(blur)

#get size of image to define pad
r,c=img_arr.shape


pad=[0 for i in range(c)]

print(f"\nShape of image before adding padding: {img_arr.shape}")

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

print(f"Shape of image after adding padding: {img_arr.shape}")


#Defining the sobel kernels
sobel_x_1=[[-1/2,0,1/2],[-1,0,1],[-1/2,0,1/2]]
sobel_x_2=[[1/2,0,-1/2],[1,0,-1],[1/2,0,-1/2]]
sobel_y_1=[[-1/2,-1,-1/2],[0,0,0],[1/2,1,1/2]]
sobel_y_2=[[1/2,1,1/2],[0,0,0],[-1/2,-1,-1/2]]


r,c=img_arr.shape

img_arr1=np.zeros((r,c))
img_arr2=np.zeros((r,c))
img_arr3=np.zeros((r,c))
img_arr4=np.zeros((r,c))


#Convolve with sobel_x kernels
for i in range(r-3+1):
    for j in range(c-3+1):
        sum=0
        for k in range(3):
            for l in range(3):
                sum+=sobel_x_1[l][k]*img_arr[i+l][j+k]
        img_arr1[i+1][j+1]=sum

for i in range(r-3+1):
    for j in range(c-3+1):
        sum=0
        for k in range(3):
            for l in range(3):
                sum+=sobel_x_2[l][k]*img_arr[i+l][j+k]
        img_arr2[i+1][j+1]=sum

#Convolve with sobel_y kernels
for i in range(c-3+1):
    for j in range(r-3+1):
        sum=0
        for k in range(3):
            for l in range(3):
                sum+=sobel_y_1[k][l]*img_arr[j+k][i+l]
        img_arr3[j+1][i+1]=sum

for i in range(c-3+1):
    for j in range(r-3+1):
        sum=0
        for k in range(3):
            for l in range(3):
                sum+=sobel_y_2[k][l]*img_arr[j+k][i+l]
        img_arr4[j+1][i+1]=sum

#Result
img_arr = abs(img_arr1) + abs(img_arr2) + abs(img_arr3) + abs(img_arr4)

#Retain only edges
for i in range(r):
    for j in range(c):
        
        if img_arr[i][j]<120:
            img_arr[i][j]=0
        '''
        else:
            img_arr[i][j]=255
        '''


#Save convolved image

cv2.imwrite('Convolved.jpg', img_arr)
print("\nConvloved image saved successfully")

#Plot original image
plt.subplot(1,2,1)
plt.imshow(img,cmap='viridis')
plt.title("Original")
plt.axis("off")

#plot convolved image
plt.subplot(1,2,2)
plt.imshow(img_arr,cmap='gray')
plt.title("Convolved")
plt.axis("off")

plt.show()

