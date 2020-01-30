import pybullet as pb
import cv2
import pybullet_data
import time
import numpy as np
from envs.aslaug_v12 import AslaugEnv
import pkgutil
egl = pkgutil.get_loader('eglRenderer')
print(egl.get_filename())


env = AslaugEnv(gui=0)
plugin = pb.loadPlugin(egl.get_filename(), "_tinyRendererPlugin")

# gui = False
#
# mode = pb.GUI if gui else pb.DIRECT
#
# pb.connect(mode)
# pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
# pb.configureDebugVisualizer(pb.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
# pb.configureDebugVisualizer(pb.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
# pb.configureDebugVisualizer(pb.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
# pb.setAdditionalSearchPath(pybullet_data.getDataPath())
# planeId = pb.loadURDF('plane.urdf')
# planeId = pb.loadURDF('/home/julien/aslaug3d_simulation/urdf/kallax/kallax_large.urdf', [-2, -2, 0])
# planeId = pb.loadURDF('/home/julien/aslaug3d_simulation/urdf/kallax/kallax_large.urdf', [20-1.47, -2, 0])
# planeId = pb.loadURDF('/home/julien/aslaug3d_simulation/urdf/kallax/kallax_large.urdf', [20-1.47, 7-0.37, 0])
# planeId = pb.loadURDF('/home/julien/aslaug3d_simulation/urdf/kallax/kallax_large.urdf', [-2, 7-0.37, 0])
# planeId = pb.loadURDF('/home/julien/aslaug3d_simulation/urdf/kallax/kallax_large.urdf', [5, 5, 0])
# planeId = pb.loadURDF('/home/julien/aslaug3d_simulation/urdf/kallax/kallax_large.urdf', [-3, 3, 0])
# planeId = pb.loadURDF('/home/julien/aslaug3d_simulation/urdf/kallax/kallax_large.urdf', [5, 0, 0])
# planeId = pb.loadURDF('/home/julien/aslaug3d_simulation/urdf/kallax/kallax_large.urdf', [10, 0, 0])
#


b_x = [-2, 20]
b_y = [-2, 7]
px_p_m = 10
inflation = 0.4

px_x = int(px_p_m*(b_x[1]-b_x[0]))
px_y = int(px_p_m*(b_y[1]-b_y[0]))
k_r = int((inflation*px_p_m))
kernel = np.zeros((k_r*2, k_r*2))
for x in range(k_r*2):
    for y in range(k_r*2):
        if (x-k_r)**2 + (y-k_r)**2 <= k_r**2:
            kernel[x, y] = 1
for i in range(10):
    env.soft_reset = False
    env.reset()
    pb.resetBasePositionAndOrientation(env.robotId, [-10, -10, 0],
                                       [0, 0, 0, 0], env.clientId)
    cam_pos = [0, 0, 0]
    camDistance = 1000000
    rpy = [0, -90, 0]
    viewMatrix = pb.computeViewMatrixFromYawPitchRoll(cam_pos,
                                                      camDistance,
                                                      rpy[2], rpy[1],
                                                      rpy[0], 2)

    # projectionMatrix = pb.computeProjectionMatrixFOV(60, 1, 0.1, 2001)
    pm = pb.computeProjectionMatrix(b_x[0], b_x[1], b_y[0], b_y[1], 999995, 1000001)
    # print(projectionMatrix)
    print(pm)
    sp_pos = env.sp_init_pos
    sp_idx_x = int(round((-b_x[0] + sp_pos[0])*px_p_m))
    sp_idx_y = int(round((b_y[1] - sp_pos[1])*px_p_m))
    img_arr = pb.getCameraImage(px_x,
                                px_y,
                                viewMatrix,
                                pm,
                                shadow=0,
                                lightDirection=[0.1, 0.1, 1],
                                renderer=pb.ER_BULLET_HARDWARE_OPENGL)

    img = np.array(img_arr[2])[:, :, 0:3]
    img[np.any(img != [255, 255, 255], axis=-1)] = [0, 0, 0]
    img = ~img[:, :, 0]
    # img = cv2.filter2D(img, -1, kernel)

    img_obst = img > 0
    zz = int(px_p_m * 0.3)
    img_obst[(sp_idx_y-zz):(sp_idx_y+zz), (sp_idx_x-zz):(sp_idx_x+zz)] = 0
    img_obst[int(round((b_y[1])*px_p_m)), :] = 1
    img_obst[int(round((b_y[1] - env.corridor_width)*px_p_m)), :] = 1
    img_free = img == 0
    img[img != 0] = -1
    img[img == 0] = 0
    img[sp_idx_y, sp_idx_x] = 1
    img2 = np.zeros(img.shape, dtype=float)
    img2[img_obst] = -5
    img2[sp_idx_y, sp_idx_x] = 1

    img3 = np.zeros(img.shape, dtype=float)
    img3[img_obst] = -1
    img3[sp_idx_y, sp_idx_x] = 1

    print(img.dtype)
    print(np.max(img))
    print(np.min(img))

    ts = time.time()
    print(img[50:70, 50:70])
    for i in range(20000):
        img2 = cv2.filter2D(img2, -1, np.ones((3,3))/9.0)
        img2[img_obst] = -5

        img2[sp_idx_y, sp_idx_x] = 1



    r_pos = env.robot_init_pos
    r_idx_x = int(round((-b_x[0] + r_pos[0])*px_p_m))
    r_idx_y = int(round((b_y[1] - r_pos[1])*px_p_m))

    x, y = r_idx_x, r_idx_y

    path = [(y, x)]
    i = 0
    while (x != sp_idx_x or y != sp_idx_y) and i < 2000:
        i += 1
        val = img2[y, x]
        travos = ((-1, 0), (+1, 0), (0, -1), (0, +1))
        evals = []
        for dy, dx in travos:
            evals.append(img2[y+dy, x+dx] - val)
        dr = np.argmax(evals)
        x += travos[dr][1]
        y += travos[dr][0]
        path.append((y, x))

    for y, x in path:
        img3[y, x] = 1

    # img2[r_idx_y, r_idx_x] = 1
    img2[sp_idx_y, sp_idx_x] = 1

    te = time.time()
    print("Took {}s".format(te-ts))

    print(img2)
    img_r = ((img3+1)*127.5).astype(np.uint8)
    print(img2.dtype)
    print(np.max(img2))
    print(np.min(img2))

    scale_percent = 500
    width = int(img_r.shape[1] * scale_percent / 100)
    height = int(img_r.shape[0] * scale_percent / 100)
    img_r = cv2.resize(img_r, (width, height), interpolation=cv2.INTER_NEAREST)
    img_r = cv2.cvtColor(img_r,cv2.COLOR_BGR2RGB)
    cv2.imshow("l", img_r)
    cv2.waitKey()




#
#
# cam_pos = [0, 0, 0]
# camDistance = 10
# rpy = [0, -75, 0]
# viewMatrix = pb.computeViewMatrixFromYawPitchRoll(cam_pos,
#                                                   camDistance,
#                                                   rpy[2], rpy[1],
#                                                   rpy[0], 2)
#
# projectionMatrix = pb.computeProjectionMatrixFOV(60, 1, 0.1, 30)
# # pm = pb.computeProjectionMatrix(-5, 25, -5, 15, 1995, 2001)
# # print(projectionMatrix)
# # print(pm)
# img_arr = pb.getCameraImage(22*25,
#                             9*25,
#                             viewMatrix,
#                             projectionMatrix,
#                             shadow=0,
#                             lightDirection=[0.1, 0.1, 1],
#                             renderer=pb.ER_BULLET_HARDWARE_OPENGL)
#
# img = np.array(img_arr[2])[:, :, 0:3]
# # img[np.any(img != [255, 255, 255], axis=-1)] = [0, 0, 0]
# img_r = cv2.resize(img, (1000, 1000),interpolation=cv2.INTER_NEAREST)
# cv2.imshow("l", img_r)
# cv2.waitKey()
