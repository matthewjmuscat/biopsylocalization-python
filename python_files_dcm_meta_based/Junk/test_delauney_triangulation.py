from scipy.spatial import Delaunay
import numpy as np
import open3d as o3d # for data visualization and meshing
import matplotlib.pyplot as plt
from mayavi import mlab

test1 = np.array([2,0,0]) # false
test2 = np.array([0,0,0]) # true
test3 = np.array([1,0,0]) # not sure? True!
test4 = np.array([0.5,0,0]) # true
test5 = np.array([1.0001,0,0]) # false
test6 = np.array([-0.8,-0.8,-0.8]) # True
test7 = np.array([1.5,1.5,0]) # false!
test8 = np.array([3,0,0]) # false
test_pts = [test1, test2, test3, test4, test5, test6, test7, test8]

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
for pts in test_pts:
    print(tri.find_simplex(pts) >= 0)  # True if point lies within poly)

vertices = tri.points 
faces = tri.simplices[:,0:3]
#mlab.triangular_mesh(vertices[:,0], vertices[:,1], vertices[:,2], faces)
#mlab.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=tri.simplices, cmap=plt.cm.Spectral)
fig.show()

plt.triplot(polygon[:,0], polygon[:,1], tri.simplices)
plt.plot(polygon[:,0], polygon[:,1], 'o')
plt.show()

#Delaunay(pcd).find_simplex(point) >= 0  # True if point lies within poly
