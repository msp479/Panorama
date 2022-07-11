#!/usr/bin/env python
# coding: utf-8

# In[52]:


"""
Image Stitching Problem
(Due date: Nov. 26, 11:59 P.M., 2021)

The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 
Note that different left/right images might be used when grading your code. 

To this end, you need to find keypoints (points of interest) in the given left and right images.
Then, use proper feature descriptors to extract features for these keypoints. 
Next, you should match the keypoints in both images using the feature distance via KNN (k=2); 
cross-checking and ratio test might be helpful for feature matching. 
After this, you need to implement RANSAC algorithm to estimate homography matrix. 
(If you want to make your result reproducible, you can try and set fixed random seed)
At last, you can make a panorama, warp one image and stitch it to another one using the homography transform.
Note that your final panorama image should NOT be cropped or missing any region of left/right image. 

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
If you intend to use SIFT feature, make sure your OpenCV version is 3.4.2.17, see project2.pdf for details.
"""




import cv2
import numpy as np
np.random.seed(0) # you can use this line to set the fixed random seed if you are using np.random
import random
import math
# random.seed(<int>) # you can use this line to set the fixed random seed if you are using random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result panorama image which is stitched by left_img and right_img
    """

    # TO DO: implement your solution here
 
    #Determining the key points and description of both left and right images.
    sift = cv2.xfeatures2d.SIFT_create()
    kp_right, des_right = sift.detectAndCompute(right_img, None)
    kp_left, des_left= sift.detectAndCompute(left_img, None)


    #Implementing the functionality of KNN algorithm to find top 2 best matches. 
    #Here we find top 2 pixels in left image that match closely with each pixel in the right image.
    ds=des_right.shape[0]
    feature_dist=np.zeros(shape=(ds,1))
    feature_dist_index=np.zeros(shape=(ds,1))
    key_pts=np.zeros(shape=(ds,4))

    for i in range(ds):
        feature_dist=np.linalg.norm(des_right[i]-des_left, axis=1)
        feature_dist_sorted=np.sort(feature_dist)
        feature_dist_index=np.argsort(feature_dist)

        key_pts[i][0]=feature_dist_sorted[0]
        key_pts[i][1]=feature_dist_sorted[1]
        key_pts[i][2]=feature_dist_index[0]
        key_pts[i][3]=feature_dist_index[1]


    #Ratio testing is performed to refine the matching points further.
    #This is controlled with the parameter eta.
    eta=0.75
    matching_pts=[]
    matching_pts_ind=[]

    for i in range(ds):
        if key_pts[i][0] < eta*key_pts[i][1]:
            matching_pts.append(int(key_pts[i][2]))
            matching_pts_ind.append(i)
 
    #All relvant points that are found to be matching in both left and right images are stored below.
    left_img_pts = np.float32([kp_left[k].pt for k in matching_pts])
    right_img_pts = np.float32([kp_right[k].pt for k in matching_pts_ind])   

    
    #Below arrays are initialized which are used in ransac approach.
    left_img_pts_cal = np.zeros((len(left_img_pts), 3))
    right_img_pts_cal = np.zeros((len(right_img_pts), 3))

    for i in range(len(left_img_pts)):
        left_img_pts_cal[i] = np.append(left_img_pts[i], 1)
        right_img_pts_cal[i] = np.append(right_img_pts[i], 1)

    #RANSAC Implementation.
    n = 4
    t = 5
    epoch = 5000
    inlier_count = 0
    top_inliers = []

    for j in range(epoch):
        #We randomly select n(=4) pixel points from right image. This is used in Homography matrix calculation.
        random_indices_ransac = np.random.choice(right_img_pts_cal.shape[0], size=n, replace=False)
        img=[]
        img_dash=[]
        for i in random_indices_ransac:
            img.append(right_img_pts_cal[i])
            img_dash.append(left_img_pts_cal[i])
    
        #Using SVD Algorithm to obtain the Homograhphy matrix H.
        ##Initializing and formulating the A-matrix.
        A=np.zeros((n*2,9))   
        i=0
        k=0
        zero_list = [0, 0, 0]

        for i in range(0,2*n,2):
            j = i+1
            A[i]=np.append(np.append(img[k],zero_list), -img_dash[k][0]*img[k])
            A[j]=np.append(np.append(zero_list, img[k]), -img_dash[k][1]*img[k])
            k+= 1

        U, Sigma, Vt = np.linalg.svd(A)
        H = Vt[8, :]
        H = np.reshape(H, (3, 3))


        #Using the Homography matrix, we now determine the coressponding left image pixels wrt right image pixels.
        inlier_kp = []
        for i in range(0, len(left_img_pts_cal)):
            #We only consider the pixels that are not part of the above randomly selected pixels.
            if i not in random_indices_ransac:
                calculated_left_pts = np.dot(H, right_img_pts_cal[i])
                normalized_left_pts = [pt/calculated_left_pts[2] for pt in calculated_left_pts]

                #If the difference between the actual and calculated left pixels differ with the value less than "t(=5)", 
                # then we consider the points as inliers.
                if np.linalg.norm(left_img_pts_cal[i]-normalized_left_pts) < t:
                    inlier_kp.append(i)

        #If the inlier count for the current iteration is more than the previous count, then we update the count and list.
        if len(inlier_kp) > inlier_count:
            inlier_count = len(inlier_kp)
            top_inliers = inlier_kp


    #Using SVD Algorithm to obtain the final Homograhphy matrix H.
    left_img_mat = []
    right_img_mat = []
    for i in top_inliers:
        left_img_mat.append(left_img_pts_cal[i])
        right_img_mat.append(right_img_pts_cal[i])
        
    A = np.zeros((2 * inlier_count, 9))
    i=0
    k=0
    zero_list = [0, 0, 0]
    
    for i in range(0,2*n,2):
        j = i+1
        A[i]=np.append(np.append(img[k],zero_list), -img_dash[k][0]*img[k])
        A[j]=np.append(np.append(zero_list, img[k]), -img_dash[k][1]*img[k])
        k+= 1

    U, Sigma, Vt = np.linalg.svd(A)
    H = Vt[8, :]
    H = np.reshape(H, (3, 3))




    #Warping and Image Stitching.
    l1, w1, d1= left_img.shape
    l2, w2, d2= right_img.shape

    corners_left_img = np.float32([[0, 0], [w1, 0], [w1, l1], [0, l1]]).reshape(-1, 1, 2)
    corners_right_img = np.float32([[0, 0], [w2, 0], [w2, l2], [0, l2]]).reshape(-1, 1, 2)

    #Performing perspective transformartion.
    corners_right_img_updated = cv2.perspectiveTransform(corners_right_img, H)
    total_corners = np.concatenate((corners_left_img, corners_right_img_updated), axis=0 )

    #Finding the Min and Max x and y co-ordinates in both images.
    x_min = math.floor(total_corners[:, 0, 0].min())
    y_min = math.floor(total_corners[:, 0, 1].min())
    x_max = math.ceil(total_corners[:, 0, 0].max())
    y_max = math.ceil(total_corners[:, 0, 1].max())
    
    #Initializing a translation matrix to adjust the image position.
    translation_mat=[[1,0,0], [0,1,0], [0,0,1]]
    translation_mat=np.array(translation_mat)
    translation_mat[0, 2] = -x_min
    translation_mat[1, 2] = -y_min
    translation_mat_updated = np.dot(translation_mat, H)

    #Warping the image to get the final result.
    corner_shape=(math.ceil(x_max - x_min), math.ceil(y_max - y_min))
    result_img = cv2.warpPerspective(right_img, translation_mat_updated, corner_shape)
    result_img[-y_min:l2 + (-y_min), -x_min:w2 + (-x_min)] = left_img


    return result_img

    #raise NotImplementedError
    #return result_img
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg', result_img)

