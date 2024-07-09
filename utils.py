import open3d as o3d
import numpy as np

def project_to_rgbd(
                    width,
                    height,
                    intrinsic,
                    extrinsic,
                    depth_scale,
                    depth_max
                    ):
    depth = np.zeros((height, width, 1), dtype=np.uint16)
    color = np.zeros((height, width, 3), dtype=np.uint8)

    # The commented code here is vectorized but is missing the filtering at the end where projected points are
    # outside image bounds and depth bounds.
    # world_points = np.asarray(self.points).transpose()
    # world_points = np.append(world_points, np.ones((1, world_points.shape[1])), axis=0)
    # points_in_ccs = np.matmul(extrinsic, world_points)[:3]
    # points_in_ccs = points_in_ccs / points_in_ccs[2, :]
    # projected_points = np.matmul(intrinsic, points_in_ccs)
    # projected_points = projected_points.astype(np.int16)

    for i in range(0, self.n):
        point4d = np.append(self.points[i], 1)
        new_point4d = np.matmul(extrinsic, point4d)
        point3d = new_point4d[:-1]
        zc = point3d[2]
        new_point3d = np.matmul(intrinsic, point3d)
        new_point3d = new_point3d/new_point3d[2]
        u = int(round(new_point3d[0]))
        v = int(round(new_point3d[1]))

        # Fixed u, v checks. u should be checked for width
        if (u < 0 or u > width - 1 or v < 0 or v > height - 1 or zc <= 0 or zc > depth_max):
            continue

        d = zc * depth_scale
        depth[v, u ] = d
        color[v, u, :] = self.colors[i] * 255

    im_color = o3d.geometry.Image(color)
    im_depth = o3d.geometry.Image(depth)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        im_color, im_depth, depth_scale=1000.0, depth_trunc=depth_max, convert_rgb_to_intensity=False)
    return rgbd
