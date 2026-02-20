import numpy
import cv2

'''
    sx : pixels per mm in width
    sy : pixels per mm on height
    f  : focal length in mm
    w  : image width
    h  : image height

    example values:
        sx = 1920 / 6     # 320 px/mm
        sy = 1280 / 4     # 320 px/mm

        f = 4.0
        w = 1920
        h = 1280
'''
def get_camera_matrix(sx, sy, f, w, h):
    result = numpy.zeros((3, 3), dtype=numpy.float32)

    result[0][0] = f*sx
    result[1][1] = f*sy

    result[0][2] = w/2.0
    result[1][2] = h/2.0

    result[2][2] = 1.0

    return result


def simulate_projected_points(n_points, K, R, T, depth_range=(4.0, 8.0), spread=1.0):

    # orignal 3D points GT
    X = numpy.random.uniform(-spread, spread, size=(n_points, 3))
    X[:, 2] = numpy.random.uniform(*depth_range, size=n_points)


    # camera 1 projection
    # make homogeneous coordinates
    X_hom = numpy.hstack([X, numpy.ones((n_points, 1))])  

    # stack transformation matrix
    RT = numpy.hstack([R, T])  # 3×4 matrix

    # Project into both cameras
    p1_h = (K @ X.T).T                      # shape: (N, 3)
    p2_h = (K @ (RT @ X_hom.T)).T           # shape: (N, 3)

    # Normalize to 2D image points
    p1 = p1_h[:, :2] / p1_h[:, 2, numpy.newaxis]
    p2 = p2_h[:, :2] / p2_h[:, 2, numpy.newaxis]

    return X, p1, p2



def generate_random_pose():
    # Random rotation using QR decomposition
    A = numpy.random.randn(3, 3)
    Q, R = numpy.linalg.qr(A)
    if numpy.linalg.det(Q) < 0:  # Ensure right-handed coordinate system
        Q[:, 2] *= -1

    # Random translation vector
    T = numpy.random.randn(3, 1)
    return Q, T





def recover_pose_from_points(p1, p2, K):
    """
    Estimate Essential matrix and recover R, T using OpenCV.
    Assumes p1 and p2 are matched 2D points (Nx2), and K is the camera intrinsic matrix.
    Returns:
        R, T  : relative rotation and translation
        mask  : inlier mask from RANSAC
    """
    # Convert to float32 if needed
    p1 = p1.astype(numpy.float32)
    p2 = p2.astype(numpy.float32)

    # Compute essential matrix
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

      
    
    # SVD of E
    U, S, Vt = numpy.linalg.svd(E)
    
    # Enforce rank-2 constraint
    if numpy.linalg.det(U) < 0: U *= -1
    if numpy.linalg.det(Vt) < 0: Vt *= -1

    # W matrix used for rotation extraction
    W = numpy.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt   

    # Ensure proper rotation matrices (det = +1)
    if numpy.linalg.det(R1) < 0: R1 *= -1
    if numpy.linalg.det(R2) < 0: R2 *= -1

    # Translation (up to scale)
    t = U[:, 2].reshape(3, 1)

    # All 4 combinations
    poses = [
        (R1,  t),
        (R1, -t),
        (R2,  t),
        (R2, -t)
    ]

    return poses



def select_correct_pose(poses, p1, p2, K):
    """
    Given 4 pose hypotheses, select the correct one by checking
    number of points in front of both cameras.
    """
    max_positive = -1
    best_pose = None

    # Normalize points (in camera coords)
    p1_norm = cv2.undistortPoints(p1.reshape(-1,1,2), K, None).reshape(-1,2)
    p2_norm = cv2.undistortPoints(p2.reshape(-1,1,2), K, None).reshape(-1,2)

    # First camera pose: [I | 0]
    P1 = numpy.hstack((numpy.eye(3), numpy.zeros((3,1))))

    for R, T in poses:
        P2 = numpy.hstack((R, T))

        # Triangulate
        pts_4d = cv2.triangulatePoints(K @ P1, K @ P2, p1.T, p2.T)  # (4, N)
        pts_3d = pts_4d[:3] / pts_4d[3]

        # Check depths in both views
        depth1 = pts_3d[2, :]
        depth2 = (R[2, :] @ pts_3d + T[2])

        positive = numpy.sum((depth1 > 0) & (depth2 > 0))

        if positive > max_positive:
            max_positive = positive
            best_pose = (R, T)

    return best_pose[0], best_pose[1]

if __name__ == "__main__":

    n_points = 64
    
    K       = get_camera_matrix(1920/6, 1280/4, 4, 1920, 1080)
    #K       = get_camera_matrix(1.0, 1.0, 1.0, 1.0, 1.0)
    R, T    = generate_random_pose()

    print(R)
    print(T)
    print("\n\n")

    Xgt, p1, p2 = simulate_projected_points(n_points, K, R, T)

    poses = recover_pose_from_points(p1, p2, K)

    R_est, T_est = select_correct_pose(poses, p1, p2, K)

    print(R_est)
    print(T_est)

    '''
    for p in poses:
        print(p[0])
        print(p[1])
        print("\n")
    '''