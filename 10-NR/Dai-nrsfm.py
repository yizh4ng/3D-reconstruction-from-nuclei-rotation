import json
import pickle

import scipy.io
import numpy as np
import cvxpy as cp
from IPython.core.debugger import set_trace
import IPython
import pandas as pd
import scipy.optimize
import math
from scipy.spatial import procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import copy
from visualize_xyz import visualize_xyz
from read_pickle import read_to_xyz
def F(gk, Mhat):
  '''
  First get a 2F*3 matrix of rotation matrices (eq 8 Dai et.al)
  Each 2x3 block of R is a rotation matrix R_i.
  R_i = [[R11, R12, R13],[R21, R22, R23]]
  namely R_i.dot(R_i.T) = [[R11^2 + R12^2+ R13^2, R11R14 + R12R15+R13R16], [R11R14 + R15R12 + R16R13, R14^2 + R15^2+ R16^2]]
  From eq 7 (Xiao), we know  each 2x3 block of Mhat.g.gT.MhatT, and we obtain the following constraints:
  (R11^2 + R12^2+ R13^2 - R14^2 + R15^2+ R16^2) = 0 and R11R14 + R12R15 + R13R16 = 0
  We normalize the return value of F by R11^2 + R12^2+ R13^2
  '''
  gk = gk.reshape(-1, 3)
  R = Mhat.dot(gk)
  R1 = R[::2]
  R2 = R[1::2]
  Z = np.sum(R1 * R2, axis=1)
  S1 = np.sum(np.square(R1), axis=1)
  S2 = np.sum(np.square(R2), axis=1)
  D = 1 - np.divide(S2, S1)
  Z = np.divide(Z, S1)
  # print(np.sum(np.square(D)) + 4*np.sum(np.square(Z)))
  return np.sum(np.square(D)) + 4 * np.sum(np.square(Z))


def retrieve_rotation(Mhat, gk, numFrames):
  '''
  eq 7 Xiao, retrieve the diagonal items. Each diagonal block is 2x2 and equal to
  Qk = [[cik^2, 0], [0, cik^2]]
  '''
  R = np.zeros((2 * numFrames, 3))
  Rdiag = np.zeros((2 * numFrames, 3 * numFrames))

  for i in range(1, numFrames + 1):
    PI = Mhat[2 * i - 2:2 * i, :]
    Qk = PI.dot(gk).dot(gk.T).dot(PI.T)

    # Eq 8 Dai et .el
    # PI.dot(gk) = cik*Ri
    R[2 * i - 2] = PI[0].dot(gk) / np.sqrt(Qk[0, 0])
    R[2 * i - 1] = PI[1].dot(gk) / np.sqrt(Qk[1, 1])

  r1 = R[0]
  r2 = R[1]
  r3 = np.cross(r1, r2)
  R_tmp = np.array([r1, r2, r3])
  # IPython.embed()
  # If det < 0 then the orthogonal transformation is a reflection
  # and not a rotation
  if np.linalg.det(R_tmp) < 0:
    R_tmp = -R_tmp
  R[0:2] = R_tmp[0:2]
  # We restrict the rotation between frames to at most 90 degrees
  for i in range(2, numFrames + 1):
    r1 = R[2 * i - 2]
    r2 = R[2 * i - 1]
    r3 = np.cross(r1, r2)
    R_i = np.array([r1, r2, r3])
    if np.linalg.det(R_i) < 0:
      R_i = -R_i
    theta = math.acos(np.maximum((np.trace((R_i.T).dot(R_tmp)) - 1), -1) / 2) * 180 / np.pi
    if theta > 90:
      R_i[0:2] = -R_i[0:2]
    R[2 * i - 2:2 * i] = R_i[0:2]
    R_tmp = R_i

  j = 1
  for i in range(1, numFrames + 1):
    Rdiag[2 * i - 2:2 * i, 3 * j - 3:3 * j] = R[2 * i - 2:2 * i]
    j += 1
  return R, Rdiag


def procrust(Rg, R):
  R = np.array(R)
  Rg = np.array(Rg)
  Y = (R.T) @ Rg
  u, s, vt = np.linalg.svd(Y, full_matrices=False)
  Q = u.dot(vt)
  R = R.dot(Q)
  return R


def compareRotations(Rg, R):
  R = procrust(Rg, R)
  F = int(R.shape[0] / 2)
  err = np.zeros(F)
  for i in range(1, F + 1):
    err[i - 1] = np.linalg.norm(R[2 * i - 2:2 * i, :] - Rg[2 * i - 2:2 * i, :])
  return err


def procrustes_alignment(Sf, Sg):
  print("Calibrating using orthogonal procrustes")
  shp = Sf.shape
  S1 = np.zeros((3, int(shp[0] * shp[1] / 3)))
  S2 = np.zeros((3, int(shp[0] * shp[1] / 3)))
  j = 0
  l = shp[1]
  for i in range(1, numFrames + 1):
    S1[0:3, j:j + l] = Sf[3 * i - 3:3 * i, :]
    S2[0:3, j:j + l] = Sg[3 * i - 3:3 * i, :]
    j = j + l
  S1 = np.array(S1)
  S2 = np.array(S2)
  Y = S1 @ (S2.T)
  u, s, vt = np.linalg.svd(Y, full_matrices=True)
  R = u.dot(vt).T

  for i in range(1, numFrames + 1):
    Sf[3 * i - 3:3 * i, :] = R.dot(Sf[3 * i - 3:3 * i, :])
  return Sf


def test(W):
  return np.random.rand(W.shape[0], W.shape[1])


def transform_to_K(S):
  '''
  This function transform a matrix of size 3F*P to size F*3p
  Basically create S# out of S
  '''
  S_sharp = np.hstack((S[0::3, :], S[1::3, :], S[2::3, :]))
  return S_sharp


def transform_to_3K(S_sharp):
  '''
  This function transform a matrix of size F*3p to size 3F*p
  Basically create S out of S#
  '''

  S_sharp_shape = S_sharp.shape
  F = S_sharp_shape[0]
  P = int(S_sharp_shape[1] / 3)
  S_k = np.zeros((3 * F, P))
  S_k[0::3, :] = S_sharp[:, 0:P]
  S_k[1::3, :] = S_sharp[:, P:2 * P]
  S_k[2::3, :] = S_sharp[:, 2 * P:]
  return S_k


def recover_S(W, R, S0, K):
  '''
  Recover S using Singular Value Thresholding
  '''
  F = int(W.shape[0] / 2)
  P = W.shape[1]

  mu_0 = 4.0  # initial continuation parameter
  tau = 0.2  # gradient step size
  mu_thr = 1e-10  # threshold on mu
  eta_mu = 0.25
  epsilon = 1e-6  # Threshold

  S_k = S0
  S_sharp = transform_to_K(S_k)

  thr_reached = False
  for i in range(1, 20):
    print("Iteration num: {}".format(i))
    if thr_reached:
      break
    mu = max(mu_thr, mu_0 * (eta_mu ** float(i)))

    if mu == mu_thr:
      thr_reached = True

    non_converged = True
    it = 1
    while (non_converged):
      # Compute the gradient 1/2 ||W-RS||^2_F
      g_S = (R.T).dot(R.dot(S_k) - W)
      # Transform g_S to g_S_sharp
      g_S_sharp = transform_to_K(g_S)

      S_sharp_k_prev = copy.deepcopy(S_sharp)

      Y_k = S_sharp - tau * g_S_sharp
      U, s, Vt = np.linalg.svd(Y_k, full_matrices=False)
      D_Y = np.diag(s)
      m = min(D_Y.shape[0], D_Y.shape[1])
      for i in range(m):
        D_Y[i, i] = D_Y[i, i] - tau * mu

        if (D_Y[i, i] < 0):
          D_Y[i, i] = 0

      S_sharp = U.dot(D_Y).dot(Vt)
      S_k = transform_to_3K(S_sharp)
      residuals = np.linalg.norm(S_sharp - S_sharp_k_prev, ord='fro') / max(1,
                                                                            np.linalg.norm(
                                                                              S_sharp_k_prev,
                                                                              ord='fro'))
      if residuals < epsilon:
        non_converged = False
      it += 1

  U, s, Vt = np.linalg.svd(S_sharp, full_matrices=False)
  s = np.diag(s)
  s[K:, K:] = 0
  S = U.dot(s).dot(Vt)
  S = transform_to_3K(S)
  return S


# Load dataset
dataset = 'yoga.mat'
mat = scipy.io.loadmat(dataset)
# K = int(mat["K"])
K =1
threshold = 0.001
# try:
# Rs = mat["Rs"]
# except Exception:
#     pass
Rs = None
Sg = mat["S"]
# Shat = mat["Shat"]
# theta = mat["theta"]
W = mat["W"]
x, y, z = [], [], []
with open('ground_truth/x.json','rb') as f:
  x = json.load(f)
with open('ground_truth/y.json','rb') as f:
  y = json.load(f)
with open('ground_truth/z.json','rb') as f:
  z = json.load(f)
x, y ,z = read_to_xyz('./data_Dai_360.pkl')
with open('data/x.json','rb') as f:
  x = json.load(f)
with open('data/y.json','rb') as f:
  y = json.load(f)
W, Sg = [], []
for i in range(len(x)):
  W.append(x[i])
  W.append(y[i])
  # Sg.append(x[i])
  # Sg.append(y[i])
  # Sg.append(z[i])
W = np.array(W)
Sg = np.array(Sg)
print("Shape of W")
print(W.shape)
numFrames = int(W.shape[0] / 2)
numPoints = W.shape[1]

# R = scipy.io.loadmat("R_Recover.mat")["R_Recover"]
# print(R)
# err = compareRotations(Rs, R)
# print(np.mean(err))
# Similar to Lucas Kanade Factorization, substract mean of 2D points
# to eliminate translation
# ===============================================================================
W = W - np.expand_dims(np.sum(W, axis=1) / numPoints, axis=1)
u, s, vt = np.linalg.svd(W, full_matrices=False)
s = np.diag(s)
S=s
V = vt
# Mhat is the first 3K columns of U and the submatrix of S of size
# (3k, 3k)
Mhat = u[:, 0:3 * K].dot(np.sqrt(s[0:3 * K, 0:3 * K]))
print("Shape of Mhat")
print(Mhat.shape)
# Construct the A matrix. To find a solution to the Gramian matrix
A = np.zeros((2 * numFrames, 9 * (K ** 2)))
for i in range(1, numFrames + 1):
  Ai = Mhat[2 * i - 2:2 * i, :]
  kr = np.kron(Ai, Ai)
  A[2 * i - 2, :] = kr[0, :] - kr[3, :]
  A[2 * i - 1, :] = kr[1, :]

# Solve the trace minimization problem using cvxpy
X = cp.Variable((3 * K, 3 * K), symmetric=True)
constraints = [X >> 0]
Xf = X.flatten()
constraints += [A @ X.flatten() == 0]
# Hack to avoid the trivial solution
#constraints += [cp.sum(X) / (9 * K ** 2) >= threshold]
prob = cp.Problem(cp.Minimize(cp.trace(X)),
                  constraints)
prob.solve()
Qt = X.value
print(np.sum(Qt))
# scipy.io.savemat('QS.mat', dict(QS=Qt))
# Get the gk matrix where Qt = gk.dot(gk.T)
u, s, vt = np.linalg.svd(Qt, full_matrices=True)
s = np.diag(s)
gk = u[:, 0:3].dot(np.sqrt(s[0:3, 0:3]))

g = scipy.optimize.minimize(F, gk, args=(Mhat,), method="BFGS", tol=1e-6,
                            options={'disp': False, 'maxiter': 10000})
sol = g.x
sol = sol.reshape(-1, 3)
# scipy.io.savemat('gk.mat', dict(gk=sol))

R, Rdiag = retrieve_rotation(Mhat, sol, numFrames)
if Rs is not None:
  err = compareRotations(Rs, R)
  Rot_err = np.mean(err)
  print("Rotation estimation error: {}".format(Rot_err))
# mtx1, mtx2, disparity = procrustes(Rs, R)

# error = []
# for i in range(numFrames+1):
#    error.append(np.linalg.norm(mtx2[2*i-2:2*i,:]-mtx1[2*i-2:2*i,:]))

# error= np.array(error)
# print("Mean error between the standardized reference rotation and the calibrated calculated rotation")
# print(np.mean(error))

# Recover the shape S using the pseudo inverse (9) from Dai el.
RdiagT = Rdiag.T
Sf = RdiagT.dot(np.linalg.inv(Rdiag.dot(RdiagT))).dot(W)

# S = procrustes_alignment(S, Sg)

Sf = recover_S(W, Rdiag, Sf, K)
# Sf = procrustes_alignment(Sf, Sg)
# ===============================================================================
# Sf = scipy.io.loadmat("Shat_BMM.mat")["Shat_BMM"]
# print(Sf)
# Sf = Qt.T @ np.sqrt(S[0:3 * K, 0:3 * K]) @ V.T[:3,:]

i = 0
da = visualize_xyz()
while (i < int(Sf.shape[0])):
  # xg = Sg[i]
  # yg = Sg[i + 1]
  # zg = Sg[i + 2]
  x = Sf[i]
  y = Sf[i + 1]
  z = Sf[i + 2]
  i += 3
  da.objects.append([x, y, z])
  #da.objects.append([xg, yg, zg])
# print(np.linalg.norm(np.array(Sf) - np.array(Sg)))
da.add_plotter(da.draw_3d)
da.show()
