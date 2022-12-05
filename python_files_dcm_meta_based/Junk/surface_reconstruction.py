"""
Testing different surface reconstruction techniques
"""
# ball pivot mesh reconstruction
#ball_radii = [x for x in np.arange(0.01,2,0.001)]
#structure_trimesh = trimesh_reconstruction_ball_pivot(threeDdata_array_fully_interpolated_with_end_caps, ball_radii)
#watertight = structure_trimesh.is_watertight()
#print(watertight)
#o3d.visualization.draw_geometries([structure_trimesh], mesh_show_back_face=True)
#plotting_funcs.plot_point_cloud_and_trimesh_side_by_side(threeDdata_array_fully_interpolated_with_end_caps, structure_trimesh)

# pyvista surface reconstruction
#pyvista_point_cloud = pv.PolyData(threeDdata_array_fully_interpolated_with_end_caps)
#surface = pyvista_point_cloud.reconstruct_surface(sample_spacing = 0.4)
#pl = pv.Plotter()
#pl.add_mesh(pyvista_point_cloud, color='k', point_size=10)
#pl.add_mesh(surface)
#pl.show()

#mf = pymeshfix.MeshFix(surface)
#mf.repair
#repaired_surface = mf.mesh

#pl = pv.Plotter()
#pl.add_mesh(pyvista_point_cloud, color='k', point_size=10)
#pl.add_mesh(repaired_surface)
#pl.show()


#trimesh_reconstruction_alphashape(threeDdata_array_fully_interpolated_with_end_caps)

#structure_trimesh_poisson = trimesh_reconstruction_poisson(threeDdata_array_fully_interpolated_with_end_caps)
#watertight = structure_trimesh.is_watertight()
#print(watertight)
#plotting_funcs.plot_point_cloud_and_trimesh_side_by_side(threeDdata_array_fully_interpolated_with_end_caps, structure_trimesh_poisson)

#alpha_shape = alphashape.alphashape(threeDdata_array_fully_interpolated_with_end_caps,1)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces)
#plt.show()