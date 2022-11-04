from scipy.spatial import Delaunay
import numpy as np
import open3d as o3d # for data visualization and meshing
import matplotlib.pyplot as plt
from mayavi import mlab
import MC_toy_model_funcs
import time 

# test1 = np.array([2,0,0]) # false
# test2 = np.array([0,0,0]) # true
# test3 = np.array([1,0,0]) # not sure? True!
# test4 = np.array([0.5,0,0]) # true
# test5 = np.array([1.0001,0,0]) # false
# test6 = np.array([-0.8,-0.8,-0.8]) # True
# test7 = np.array([1.5,1.5,0]) # false!
# test8 = np.array([3,0,0]) # false
# test_pts = [test1, test2, test3, test4, test5, test6, test7, test8]
num_pts = 500
test_pts = [np.random.uniform(-3,3,3) for i in range(num_pts)]
test_pts_arr = np.array(test_pts)
colors = np.empty(num_pts, dtype=str)

polygon = np.array([[1,0,0],
[-1,0,0],
[0,1,0],
[0,-1,0],
[-1,1,0],
[1,-1,0],
[-1,-1,0],
[1,1,0],
[0,1,0],
[0,0,1],
[2,2,1],
[2,-2,1],
[-2,2,1],
[-2,-2,1],
[0,0,-1],
[2,2,-1],
[2,-2,-1],
[-2,2,-1],
[-2,-2,-1]
],dtype=float)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(polygon)
o3d.visualization.draw_geometries([pcd])


tri = Delaunay(polygon)
for ind,pts in enumerate(test_pts):
    #print(tri.find_simplex(pts) >= 0)  # True if point lies within poly)
    if tri.find_simplex(pts) >= 0:
        colors[ind] = 'g'
    else: 
        colors[ind] = 'r'

#vertices = tri.points 
#faces = tri.simplices[:,0:3]
#mlab.triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], faces)
#mlab.show()

#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
#fig.show()

st = time.time()

fig = plt.figure()
ax = plt.axes(projection='3d')
MC_toy_model_funcs.plot_tri_simple(ax, polygon, tri)
ax.scatter(test_pts_arr[:,0],test_pts_arr[:,1],test_pts_arr[:,2],c=colors[:])
fig.show()

et = time.time()
elapsed_time = et - st
print('\n Execution time:', elapsed_time, 'seconds')

print(1)

st = time.time()

fig = plt.figure()
ax = plt.axes(projection='3d')
MC_toy_model_funcs.plot_tri_more_efficient(ax, polygon, tri)
ax.scatter(test_pts_arr[:,0],test_pts_arr[:,1],test_pts_arr[:,2],c=colors[:])
fig.show()

et = time.time()
elapsed_time = et - st
print('\n Execution time:', elapsed_time, 'seconds')

print(1)

