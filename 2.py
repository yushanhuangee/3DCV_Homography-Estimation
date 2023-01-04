import sys
import numpy as np
import math
import cv2

selected_corners = 0

def detect_location(event,x,y,flags,param):
    global selected_corners
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),15,(255,0,0),-1)
        selected_corners+=1
        print(x,y)


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

def backward_warp(img, H, target_size):
    M, N = target_size
    target_location = homogeneous_representation(np.mgrid[0:M, 0:N].reshape((2,-1)).T)
    original_location = np.dot(np.linalg.inv(H), target_location.T)
    original_location = (original_location/original_location[2, :].reshape(1, -1))[:2, :].transpose(1,0)

    original_location = original_location.reshape(M, N, 2).astype(np.float32)
    
    warped = cv2.remap(img, original_location, None, cv2.INTER_LINEAR).transpose(1, 0, 2)

    
    '''
    #bilinear interpolation
    #find 4 adjacent points 
    n_a = np.floor(original_location)
    n_b = np.copy(n_a)
    n_b[0,:,:]+=1
    n_d = np.copy(n_a)
    n_d[1,:,:]+=1
    n_c = n_a + 1
    adj_p = np.stack((n_a, n_b, n_c, n_d))

    
    #calculate 4 weights
    w_a = (n_c[0,:,:] - original_location[0,:,:])*(n_c[1,:,:] - original_location[1,:,:])
    w_b = (original_location[0,:,:] - n_d[0,:,:])*(n_d[1,:,:] - original_location[1,:,:])
    w_c = (original_location[0,:,:] - n_a[0,:,:])*(original_location[1,:,:] - n_a[1,:,:])
    w_d = (n_b[0,:,:] - original_location[0,:,:])*(original_location[1,:,:] - n_b[1,:,:])

    #find 4 rgb value
    rgb = []
    for i in range(4):
        x, y =adj_p[i].astype('int')
        r_indices = (x + N*y)*3
        g_indices = (x + N*y)*3 + 1
        b_indices = (x + N*y)*3 + 2
        src = img .flatten()
        r = np.take(src, r_indices, mode='wrap')
        g = np.take(src, g_indices, mode='wrap')
        b = np.take(src, b_indices, mode='wrap')
        rgb.append(np.stack((r,g,b)))

    rgb = np.asarray(rgb)  #4*3*M*N
    map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
    map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)

    '''
  
    

    
    return warped


if __name__ == '__main__':
    
    img = cv2.imread(sys.argv[1])


    #set the target to be A4 size (1188, 840)
    output_w = 840
    output_h = 1188
    pt = np.asarray([[0, 0], [output_w, 0], [0, output_h], [output_w, output_h]])
    ps = np.asarray([[153, 201], [575, 185], [21, 791], [731,769]])

    normalized_pt, T_pt = points_normalize(homogeneous_representation(pt))
    normalized_ps, T_ps = points_normalize(homogeneous_representation(ps))
    normalized_H = homography_estimate(normalized_ps, normalized_pt, pairs_num=4)
    H = np.matmul(np.linalg.inv(T_pt), np.matmul(normalized_H, T_ps))
    #H, status = cv2.findHomography(ps, pt)
    #cv2.imshow('image',img)

    rectified_img = backward_warp(img, H, (output_w, output_h))
    cv2.imwrite('rectified_p2.png', rectified_img)
    

    
    find_corner = False
    if find_corner:
        cv2.namedWindow('image')
        cv2.setMouseCallback('image',detect_location)
        while(selected_corners<4):
            cv2.imshow('image',img)
            k=cv2.waitKey(1)
            if k==ord('q'):
                break
        cv2.imwrite('mark_p2.png', img)
        cv2.destroyAllWindows()