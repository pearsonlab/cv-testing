import cv2
import numpy as np


def drawMatches(img1, kp1, img2, kp2, matches, mask):
    """
    Found this function on StackOverflow. Below is the original
    description.

    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    i = 0
    for mat in matches:
        if mask[i] == 0:
            i += 1
            continue
        i += 1
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 10
        # colour red
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 10, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 10, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour green
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 3)

    # Also return the image if you'd like a copy
    return out


def object_find(match_sift, frame, MIN_MATCH_COUNT):
    '''
    Finds an object (from precomputed sift features) within a frame.  If
    found it returns the transformation matrix from frame to still.

    This code is based on the OpenCV feature detection tutorial
    '''

    sift = cv2.SIFT()
    kp1, des1 = match_sift
    kp2, des2 = sift.detectAndCompute(frame, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        return M

    else:
        return None


# def frame_to_still_gaze(gp_x, gp_y, M):

