import numpy as np
# https://github.com/nghiaho12/rigid_transform_3D/blob/843c4906fe2b22bec3b1c49f45683cb536fb054c/rigid_transform_3D.py#L10
# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector)
def rigid_transform_3D(A, B, center_A, center_B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.array([center_A[0], center_A[1], 0])
    centroid_B = np.array([center_B[0], center_B[1], 0])

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    assert np.isnan(H).any() == False
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
# def rigid_transform_3D(A, B, center_A, center_B, scale=False):
#     assert len(A) == len(B)
#
#     N = A.shape[0] # total points
#
#     centroid_A = np.array([center_A[0], center_A[1], 0])
#     centroid_B = np.array([center_B[0], center_B[1], 0])
#
#     # center the points
#     AA = A - np.tile(centroid_A, (N, 1))
#     BB = B - np.tile(centroid_B, (N, 1))
#
#     # dot is matrix multiplication for array
#     if scale:
#         H = np.transpose(BB) * AA / N
#     else:
#         H = np.transpose(BB) * AA
#
#     U, S, Vt = np.linalg.svd(H)
#
#     R = Vt.T * U.T
#
#     # special reflection case
#     if np.linalg.det(R) < 0:
#         print ("Reflection detected")
#         Vt[2, :] *= -1
#         R = Vt.T * U.T
#
#     if scale:
#         varA = np.var(A, axis=0).sum()
#         c = 1 / (1 / varA * np.sum(S))  # scale factor
#         t = -R * (centroid_B.T * c) + centroid_A.T
#     else:
#         c = 1
#         t = -R * centroid_B.T + centroid_A.T
#
#     return c, R, t