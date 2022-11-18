#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install --upgrade pip')
get_ipython().system('pip3 install opencv-python')


# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import sys
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
# reading in an image
img = mpimg.imread('newimage.jpg')
plt.figure(figsize=(10,8))
print('This image is:', type(img), 'with dimensions:', img.shape)
plt.imshow(img)
plt.show()


# In[2]:


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# In[3]:


gray = grayscale(img)
plt.figure(figsize=(10,8))
plt.imshow(gray, cmap='gray')
plt.show()


# In[4]:


def gaussian_blur(img, krenel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size),0)
kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

plt.figure(figsize=(10,8))
plt.imshow(blur_gray, cmap='gray')
plt.show()


# In[5]:


def canny(img, low_threshold, high_threshold): 
    canny_img = cv2.Canny(img, low_threshold, high_threshold)
    return canny_img

low_threshold = 70
high_threshold = 210

edges = canny(blur_gray, low_threshold, high_threshold)

plt.figure(figsize=(10,8))
plt.imshow(edges, cmap='gray')
plt.show()


# In[6]:


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,)*channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# In[7]:


imshape = img.shape
vertices = np.array([[(100,imshape[0]),(300,230),(420,230),(imshape[1]-20,imshape[0])]],dtype=np.int32)
mask = region_of_interest(edges, vertices)


# In[8]:


plt.imshow(mask, cmap='gray')
plt.show()
print(img.shape)


# In[14]:


def draw_lines(img, lines, color=[255,0,0], thickness=4):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),color, thickness)
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                           minLineLength=min_line_len,
                           maxLineGap=max_line_gap)
    #line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    #draw_lines(line_img, lines)
    return lines


# In[18]:


rho = 1
theta = np.pi/180
threshold = 30
min_line_len = 10
max_line_gap = 20

lines = hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)

line_arr = hough_lines(R01_img, 1, np.pi/180, 30, 10, 20)
line_arr = np.squeeze(line_arr)

slope_degree = (np.arctac2(line_arr[:,1]-line_arr[:,0]-line_arr[:,2]) * 180)/np.pi
#plt.imshow(lines, cmap='gray')
#plt.show()


# In[17]:


def weighted_img(img, initial_img, a=0.8, b=1., c = 0.):
    return cv2.addWeighted(initial_img, a, img, b, c)
lines_edges = weighted_img(lines, img, a=0.8,b=1.,c=0.)

plt.imshow(lines_edges)
plt.show()


# In[ ]:





# In[ ]:




