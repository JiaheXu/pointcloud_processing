import open3d as o3d
coor_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
visualizer = o3d.visualization.Visualizer()
visualizer.create_window()
visualizer.add_geometry(coor_frame)
view_ctl = visualizer.get_view_control()
view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
# view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
view_ctl.set_front((-1.5, 0.0, 0.7))  # set the positive direction of the x-axis toward you
view_ctl.set_lookat((0, 0, 0.5))  # set the original point as the center point of the window
visualizer.run()
