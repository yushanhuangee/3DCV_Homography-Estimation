import sys
import numpy as np
import cv2
import math

def get_sift_correspondences(img1, img2, pairs_num):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv2.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    #apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    good_matches = sorted(good_matches, key=lambda x: x.distance) #ascending

    # Select top k pairs from the matching result
    img_draw_match = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:pairs_num], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv.imshow('match', img_draw_match)
    #cv.waitKey(0)
    cv2.imwrite('match_{}.jpg'.format(str(pairs_num)), img_draw_match)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    return points1, points2

def get_line_correspondences(img1, img2, pairs_num):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv2.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(crosscheck=True)

    good_matches = []
    
    matches = matcher.knnMatch(des1, des2, k=2)
    #apply ratio test
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good_matches.append(m)
    
    
    good_matches = sorted(good_matches, key=lambda x: x.distance) #ascending

    # Select top k pairs from the matching result
    img_draw_match = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:pairs_num], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #cv.imshow('match', img_draw_match)
    #cv.waitKey(0)
    cv2.imwrite('match_{}.jpg'.format(str(pairs_num)), img_draw_match)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    return points1, points2

def homography_estimate(p1, p2, pairs_num):

    #DLT
    aList = []
    assert (pairs_num == len(p1) == len(p2)), "pairs_num != len(p1) != len(p2)"
    for i in range(pairs_num):
        u_s, v_s, _ = p1[i]
        u_t, v_t, _ = p2[i]
        a1 = [0.0, 0.0, 0.0, -u_s, -v_s, -1.0, v_t * u_s, v_t * v_s, v_t]
        a2 = [u_s, v_s, 1.0, 0.0, 0.0, 0.0, -u_t * u_s, -u_t * v_s, -u_t]
        aList.append(a1)
        aList.append(a2)
    A = np.asarray(aList)
    #svd composition
    u, s, v = np.linalg.svd(A)
    h = v[-1,:]
    H = h.reshape((3, 3))
    return H
    
def homogeneous_representation(points): #N*2
    points_num = points.shape[0]
    ones = np.ones((points_num,1))
    return np.concatenate((points,ones), axis=1)


def points_normalize(points): #N*3
    #do point normalization and transfrom to homogeneous transformations representation
    pairs_num = points.shape[0]
    mean = np.mean(points, axis=0)
    mu, mv, _ = mean
    avg_d = np.sum(np.linalg.norm((points - mean), axis=1))/pairs_num
    ratio = avg_d/math.sqrt(2)
    T = np.asarray([[1/ratio, 0.0, -mu/ratio], [0.0, 1/ratio, -mv/ratio], [0.0, 0.0, 1.0]])
    normalized_points = np.matmul(T, points.T).T
    return normalized_points, T #N*3

def reprojection_error(p_s, p_t, H):

    p_s = homogeneous_representation(p_s)
    p_t = homogeneous_representation(p_t)
    estimated_p_t = np.dot(H, p_s.T)
    estimated_p_t = (estimated_p_t/estimated_p_t[2, :].reshape(1, -1)).transpose(1,0)

    error = np.sum(np.linalg.norm((estimated_p_t - p_t), axis=1))/p_s.shape[0]

    print("error:", error)
    
def warping(img, H, target_size):
    M, N = target_size
    target_location = homogeneous_representation(np.mgrid[0:M, 0:N].reshape((2,-1)).T)
    original_location = np.dot(np.linalg.inv(H), target_location.T)
    original_location = (original_location/original_location[2, :].reshape(1, -1))[:2, :].transpose(1,0)

    original_location = original_location.reshape(M, N, 2).astype(np.float32)
    
    warped = cv2.remap(img, original_location, None, cv2.INTER_LINEAR).transpose(1, 0, 2)
    return warped

def outlier_remove(points1, points2, img1_shape, img2_shape):
    h1, w1, _ = img1_shape
    h2, w2, _ = img2_shape
    
    relative_p1_w = np.floor(points1[:,0]/(w1/3)).astype('int')
    relative_p1_h = np.floor(points1[:,1]/(h1/3)).astype('int')
    relative_p1 = np.stack((relative_p1_w, relative_p1_h))
    relative_p2_w = np.floor(points2[:,0]/(w2/3)).astype('int')
    relative_p2_h = np.floor(points2[:,1]/(h2/3)).astype('int')
    relative_p2 = np.stack((relative_p2_w, relative_p2_h))

    difference  = np.sum(np.absolute(relative_p2 - relative_p1), axis=0)
    new_points1 = []
    new_points2 = []
    for i in range(points1.shape[0]):
        if difference[i] <1:
            new_points1.append(points1[i])
            new_points2.append(points2[i])

            

    return np.asarray(new_points1), np.asarray(new_points2)

if __name__ == '__main__':
    img1 = cv2.imread(sys.argv[1])
    img2 = cv2.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    
    pairs_num = 20
    normalize = False
    points1, points2 = get_sift_correspondences(img1, img2, pairs_num)
    points1, points2 = outlier_remove(points1, points2, img1.shape, img2.shape)

    if normalize:

        normalized_ps, T_ps = points_normalize(homogeneous_representation(points1[:pairs_num]))
        normalized_pt, T_pt = points_normalize(homogeneous_representation(points2[:pairs_num]))
        normalized_H = homography_estimate(normalized_ps, normalized_pt, pairs_num)
        H = np.matmul(np.linalg.inv(T_pt), np.matmul(normalized_H, T_ps))

    else:
        H = homography_estimate(homogeneous_representation(points1[:pairs_num]), homogeneous_representation(points2[:pairs_num]), pairs_num)
        

    p_s = gt_correspondences[0]
    p_t = gt_correspondences[1]
    
    reprojection_error(p_s, p_t, H)

    rectified_img = warping(img1.transpose(1,0,2), H, img2.shape[:2]).transpose(1,0,2)
    cv2.imwrite('rectified_p1_1.png', rectified_img)


    
    
