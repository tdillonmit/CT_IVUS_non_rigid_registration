"""Open3D geometry-construction and visualization helpers for CT–IVUS registration."""

import copy
import time
import numpy as np
import open3d as o3d
from scipy.interpolate import UnivariateSpline


def get_branched_skeleton(ct_skeleton_pc,closest_indices_ct,ct_centroids):
    """Construct a downsampled centerline graph with branch-centroid edges."""
    ct_skeleton_points = np.asarray(ct_skeleton_pc.points)

    print("closest_indices_ct before", closest_indices_ct)

    relevant_points = ct_skeleton_points[closest_indices_ct,:]
    i=0
    revised_points = np.empty((0,3))
    for point in ct_skeleton_points:
        if i%2==0 or i in closest_indices_ct:
            revised_points = np.vstack((revised_points, point))
        i=i+1

    print("relevant_points are", relevant_points)
    i=0
    closest_indices_ct = []
    for point in revised_points:
        print("point", point)
        if np.any(np.all(relevant_points == point, axis=1)):
            closest_indices_ct.append(i)
        i=i+1

    print("closest_indices_ct", closest_indices_ct)
    closest_indices_ct = np.asarray(closest_indices_ct)
    ct_skeleton_points = revised_points

    beforehand = np.shape(ct_skeleton_points)[0]
    graph_new=generate_graph_from_centreline(ct_skeleton_points)

    for closest_index_ct, ct_centroid in zip(closest_indices_ct,ct_centroids):
        graph_edge = np.array([ct_skeleton_points[closest_index_ct, :], ct_centroid])
        graph_new = np.vstack((graph_new, graph_edge[np.newaxis, :, :]))

    ct_skeleton_points = np.vstack((ct_skeleton_points,ct_centroids))

    ct_skeleton_pc_with_branches = o3d.geometry.PointCloud()
    ct_skeleton_pc_with_branches.points = o3d.utility.Vector3dVector(ct_skeleton_points)

    edges = []
    for edge in graph_new:
        start_idx = np.where(np.all(ct_skeleton_points == edge[0], axis=1))[0][0]
        end_idx = np.where(np.all(ct_skeleton_points == edge[1], axis=1))[0][0]
        edges.append([start_idx, end_idx])

    edges = np.array(edges)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(ct_skeleton_points)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    return ct_skeleton_pc_with_branches, line_set


def fit_3D_b_spline(centreline, numPoints=100, smoothness = 10):
    """Fit a smoothed arc-length-parameterized spline through 3D centerline points."""
    not_fitted = 1
    while not_fitted == 1:
        # Compute cumulative arc length for parameterization
        t = np.zeros(centreline.shape[0])
        t[1:] = np.cumsum(np.sqrt(np.sum(np.diff(centreline, axis=0)**2, axis=1)))

        diffs_x = np.diff(centreline[:, 0])
        diffs_y = np.diff(centreline[:, 1])
        diffs_z = np.diff(centreline[:, 2])

        # Fit individual B-splines to x, y, and z components

        spl_x = UnivariateSpline(t, centreline[:, 0], s=smoothness)
        spl_y = UnivariateSpline(t, centreline[:, 1], s=smoothness)
        spl_z = UnivariateSpline(t, centreline[:, 2], s=smoothness)

        # Generate evenly spaced values along the arc length
        t_new = np.linspace(t[0], t[-1], numPoints)
        x_new, y_new, z_new = spl_x(t_new), spl_y(t_new), spl_z(t_new)

        if np.isnan(x_new).any() or np.isnan(y_new).any() or np.isnan(z_new).any():
            smoothness = smoothness * 2
        else:
            not_fitted = 0

    b_spline = np.vstack((x_new, y_new, z_new)).T

    return b_spline


def generate_graph_from_centreline(centreline):
    """Convert an ordered centerline into consecutive point-pair edges."""
    i=0
    edges=[]
    while(i < np.shape(centreline)[0]-1):
        edge=np.vstack((centreline[i,:], centreline[i+1,:]))
        edges.append(edge)
        i=i+1

    edges=np.array(edges)

    return edges


def find_furthest_points(points):
    """Return the indices of the most widely separated pair of points."""
    distances = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)

    # Find indices of the maximum distance
    max_distance_indices = np.unravel_index(np.argmax(distances), distances.shape)

    index_1=max_distance_indices[0]
    index_2=max_distance_indices[1]

    return index_1,index_2


def plot_open3d_correspondences(source_pcd, target_pcd, correspondences, visualize_debug, color=[0,1,0], change_order=False):
    """
    Plots correspondences between two point clouds as green lines.

    Args:
    - source_pcd: Open3D PointCloud object of the source point cloud.
    - target_pcd: Open3D PointCloud object of the target point cloud.
    - correspondences: A numpy array of shape (N, 2), where each row contains
      indices of corresponding points in the source and target point clouds.
    """
    lines = []
    colors = []

    source_points =np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)

    for correspondence in correspondences:
        source_idx, target_idx = correspondence
        source_point = source_points[source_idx,:]
        target_point = target_points[target_idx,:]

        line = o3d.geometry.LineSet()

        print("source point", source_point)
        print("target point", target_point)
        line.points = o3d.utility.Vector3dVector(np.vstack((source_point, target_point)))
        line.lines = o3d.utility.Vector2iVector([np.asarray([0,1])])
        line.colors = o3d.utility.Vector3dVector([np.asarray(color)])

        lines.append(line)

    # visualize_debug the point clouds and correspondences

    if(visualize_debug == 1 and change_order== False):

        o3d.visualization.draw_geometries([source_pcd, target_pcd]+lines)
    elif(visualize_debug == 1 and change_order== True):
        source_pcd.paint_uniform_color([1,0,0])
        target_pcd.paint_uniform_color([0,0,1])
        o3d.visualization.draw_geometries(lines+[source_pcd, target_pcd])

    return lines


def create_wireframe_lineset_from_mesh(full_lumen_mesh):
    """Convert triangle-mesh edges into an Open3D line set."""
    triangles = np.asarray(full_lumen_mesh.triangles)
    edges = list({tuple(sorted(edge)) for triangle in triangles for edge in [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])]})
    edges = np.vstack(edges)
    lines = o3d.geometry.LineSet()
    lines.points = full_lumen_mesh.vertices
    lines.lines = o3d.utility.Vector2iVector(edges)
    lines.paint_uniform_color([0, 0, 0])  # Set the wireframe color

    return lines


def show_tubeset_correspondences(corres,corres_original,closest_indices_ct,closest_indices_ivus, ct_skeleton_pc,ivus_skeleton_pc,ct_centroids,ivus_centroids, ct_spheres, ivus_spheres, visualize_branches):
    """Render branch-aware CT–IVUS correspondences as tubes and spheres."""
    closest_indices_ct = []
    ct_skeleton_pc_points = np.asarray( ct_skeleton_pc.points )
    for ct_centroid in ct_centroids:
        closest_index = np.argmin(np.linalg.norm(ct_skeleton_pc_points-ct_centroid, axis=1))
        closest_indices_ct.append(closest_index)

    print("closest_indices_ct", closest_indices_ct)

    ct_skeleton_pc_with_branches, ct_lineset_branches = get_branched_skeleton(ct_skeleton_pc,closest_indices_ct,ct_centroids)
    ivus_skeleton_pc_with_branches, ivus_lineset_branches = get_branched_skeleton(ivus_skeleton_pc,closest_indices_ivus,ivus_centroids)

    ct_skeleton_branches = o3d.geometry.PointCloud()
    ct_skeleton_branches.points = o3d.utility.Vector3dVector(ct_centroids)
    ivus_skeleton_branches = o3d.geometry.PointCloud()
    ivus_skeleton_branches.points = o3d.utility.Vector3dVector(ivus_centroids)

    ivus_lineset_branches_orig = copy.deepcopy(ivus_lineset_branches)

    translation = [-0.05,0.035,-0.05]
    ivus_skeleton_pc_with_branches.translate(translation)
    ivus_lineset_branches.translate(translation)

    ivus_skeleton_pc_with_branches.paint_uniform_color([0,0,1])
    ct_skeleton_pc_with_branches.paint_uniform_color([1,0,0])
    ivus_lineset_branches.paint_uniform_color([0,0,1])
    ct_lineset_branches.paint_uniform_color([1,0,0])

    ivus_spheres_copy = copy.deepcopy(ivus_spheres)
    ivus_spheres_copy.translate(translation)

    ivus_skeleton_transformed = copy.deepcopy(ivus_skeleton_pc)
    ivus_skeleton_transformed.translate(translation)

    ivus_skeleton_branches.translate(translation)

    corres_lines_branches = plot_open3d_correspondences(ct_skeleton_branches, ivus_skeleton_branches, corres_original, 0, [1,0,1])
    corres_lines = plot_open3d_correspondences(ct_skeleton_pc, ivus_skeleton_transformed, corres, 0)

    ct_skeleton_spheres = get_sphere_cloud(np.asarray(ct_skeleton_pc.points), 0.0015, 20, [1,0,0])
    ivus_skeleton_spheres = get_sphere_cloud(np.asarray(ivus_skeleton_transformed.points), 0.0015, 20, [0,0,1])
    ivus_skeleton_spheres_orig = get_sphere_cloud(np.asarray(ivus_skeleton_pc.points), 0.0015, 20, [0,0,1])

    ivus_tubes_branches = convert_linesets_to_tubes([ivus_lineset_branches], radius =0.001, resolution=3, color=[0,0,1])
    ct_tubes_branches = convert_linesets_to_tubes([ct_lineset_branches], radius =0.001, resolution=3, color=[1,0,0])

    corres_tubes = convert_linesets_to_tubes(corres_lines, radius =0.0002, resolution=3, color=[0,1,0])

    corres_tubes_branches = convert_linesets_to_tubes(corres_lines_branches, radius =0.0006, resolution=3, color=[1,0,1])

    o3d.visualization.draw_geometries(ct_tubes_branches + ivus_tubes_branches + corres_tubes+corres_tubes_branches +[ct_spheres, ivus_spheres_copy, ct_skeleton_spheres, ivus_skeleton_spheres], mesh_show_back_face=True)

    # without displacement
    ivus_tubes_branches_orig = convert_linesets_to_tubes([ivus_lineset_branches_orig], radius =0.001, resolution=3, color=[0,0,1])

    # [ct_spheres ,ivus_spheres]
    return ct_tubes_branches, ivus_tubes_branches, corres_tubes, corres_tubes_branches, ivus_tubes_branches_orig, ct_skeleton_spheres, ivus_skeleton_spheres_orig


def create_torus_with_2d_cross(center, normal, major_radius, minor_radius,
                               resolution=80, cross_arm_frac=0.3, cross_radius_scale=0.6):
    """
    0.38 before
    Create a torus plus a 2D tubular cross in the torus plane.

    cross_arm_frac: fraction of torus major_radius for each arm length (one direction)
                    e.g. 0.25 → each arm goes 0.25*R outwards, so total arm length = 0.5R
    cross_radius_scale: relative thickness of cross tubes vs minor_radius
    """

    # --- 1. Create Torus ---
    torus = o3d.geometry.TriangleMesh.create_torus(
        torus_radius=major_radius,
        tube_radius=minor_radius,
        radial_resolution=resolution,
        tubular_resolution=resolution
    )

    normal = normal / np.linalg.norm(normal)

    # --- 2. Compute Rotation Aligning Z→normal ---
    z_axis = np.array([0, 0, 1.0])
    v = np.cross(z_axis, normal)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, normal)

    if s < 1e-8:
        R = np.eye(3)
        if c < 0:
            R[2, 2] = -1
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    torus.rotate(R, center=(0, 0, 0))
    torus.translate(center)
    torus.compute_vertex_normals()
    torus.paint_uniform_color([0.8, 0.3, 0.1])

    # --- 3. Local frame for 2D cross ---
    # choose arbitrary vector not parallel to normal
    tmp = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    u = np.cross(normal, tmp)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)

    # Compute arm length and cylinder radius
    arm_length = major_radius * cross_arm_frac * 2.0     # full cylinder length
    arm_radius = minor_radius * cross_radius_scale

    # --- 4. Create two cylinders for the cross ---
    cyl1 = o3d.geometry.TriangleMesh.create_cylinder(radius=arm_radius, height=arm_length, resolution=50)
    cyl2 = o3d.geometry.TriangleMesh.create_cylinder(radius=arm_radius, height=arm_length, resolution=50)

    # Default cylinder axis is +Z; rotate to align with u and v
    def align_cylinder_to_vector(cyl, vec):
        vec = vec / np.linalg.norm(vec)
        z = np.array([0, 0, 1.0])
        v2 = np.cross(z, vec)
        s2 = np.linalg.norm(v2)
        c2 = np.dot(z, vec)
        if s2 < 1e-8:
            R2 = np.eye(3)
            if c2 < 0:
                R2[2, 2] = -1
        else:
            vx2 = np.array([[0, -v2[2], v2[1]],
                            [v2[2], 0, -v2[0]],
                            [-v2[1], v2[0], 0]])
            R2 = np.eye(3) + vx2 + vx2 @ vx2 * ((1 - c2) / (s2 ** 2))
        cyl.rotate(R2, center=(0, 0, 0))
        return cyl

    cyl1 = align_cylinder_to_vector(cyl1, u)
    cyl2 = align_cylinder_to_vector(cyl2, v)

    # Move so cylinders intersect at center (their origin is midpoint of height)
    cyl1.translate(center)
    cyl2.translate(center)

    cyl1.paint_uniform_color([1,0,0])
    cyl2.paint_uniform_color([1,0,0])

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0007)
    sphere.translate(center)
    sphere.paint_uniform_color([1,0,0])

    torus = torus + cyl1 + cyl2
    torus.compute_vertex_normals()

    cross = cyl1+cyl2
    cross.compute_vertex_normals()

    return torus, cross


def get_ivus_rings(ivus_centroids,corres_original, clustered_pcs, canonical_flap_pc, branch_normals, visualize_debug=False):
    """Create IVUS branch rings and center markers for matched branch locations."""
    ivus_rings = o3d.geometry.TriangleMesh()
    crosses =  o3d.geometry.TriangleMesh()

    for correspondence in corres_original:

        # GETTING IVUS BRANCH
        clustered_pc = clustered_pcs[correspondence[1]]

        pts = np.asarray(clustered_pc.points)

        # --- PCA of point cloud ---
        center = np.mean(pts, axis=0)
        pts_centered = pts - center

        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        # Sort eigenvectors by descending eigenvalue
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

        # --- visualize 3 principal directions ---
        colors = [
            [1, 0, 0],   # red = principal direction 1
            [0, 1, 0],   # green = 2nd
            [0, 0, 1],   # blue = 3rd
        ]

        linesets = []
        spheres_list = []
        p1s=[]
        p2s=[]
        branch_lengths=[]
        for i in range(3):
            direction = eigvecs[:, i]
            length = 4 * np.sqrt(eigvals[i])
            branch_lengths.append(length)
            p1 = center - 0.5 * length * direction
            p2 = center + 0.5 * length * direction
            if(np.linalg.norm(p2-ivus_centroids[correspondence[1], :]) < np.linalg.norm(p2-ivus_centroids[correspondence[1], :]) ):
                p1_save = copy.deepcopy(p1)
                p1 = p2
                p2 = p1_save
            p1s.append(p1)
            p2s.append(p2)
            points = np.linspace(p1, p2, 20)
            spheres = get_sphere_cloud(points, 0.002, 10, colors[i])
            spheres_list.append(spheres)

        projections=[]
        centerline_points = np.asarray(canonical_flap_pc.points)
        closest_point= np.argmin(np.linalg.norm(centerline_points-ivus_centroids[correspondence[1]],axis=1))
        radial_vector = ivus_centroids[correspondence[1], :] - centerline_points[closest_point, :]
        radial_normalized = radial_vector / np.linalg.norm(radial_vector)
        minor_radius = 0.000275
        major_radius = 0.00425  # shrink it a little bit so it doesn't overlap mesh

        origin = ivus_centroids[correspondence[1], :]
        branch_normal = branch_normals[correspondence[0]]
        torus, cross = create_torus_with_2d_cross(origin, branch_normal, major_radius, minor_radius, 30)
        torus.paint_uniform_color([0,0,1])

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0007)
        sphere.translate(origin)
        sphere.paint_uniform_color([0,0,1]) # overrride cross
        sphere.compute_vertex_normals()
        cross = sphere

        cross.paint_uniform_color([0.2,0.2,1])
        ivus_rings= ivus_rings + torus # replace the ct_spheres
        crosses = crosses + cross

    print("ivus rings", ivus_rings)

    return ivus_rings, crosses


def get_sphere_cloud(points,radius,resolution, color=[0,1,0]):
    """Combine equal-radius spheres centered at the supplied points into one mesh."""
    spheres = o3d.geometry.TriangleMesh()
    spheres.vertices = o3d.utility.Vector3dVector([])
    spheres.triangles = o3d.utility.Vector3iVector([])

    for image_point in points:

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
        translated=sphere.translate(image_point)

        num_target_vertices = len(spheres.vertices)
        triangles_offset_np = np.asarray(sphere.triangles) + num_target_vertices
        triangles_offset = o3d.utility.Vector3iVector(triangles_offset_np)

        # Extend the vertices and triangles of the target mesh with those of the source mesh
        spheres.vertices.extend(sphere.vertices)
        spheres.triangles.extend(triangles_offset )

    spheres.paint_uniform_color(color)
    spheres.compute_vertex_normals()

    return spheres


def create_tube_mesh(start, end, radius=0.02, resolution=10, color=[0, 1, 0]):
    """
    Create a tubular mesh (cylinder) connecting two 3D points.

    Args:
        start (np.array): Starting point of the tube.
        end (np.array): Ending point of the tube.
        radius (float): Radius of the tube.
        resolution (int): Resolution of the cylinder.
        color (list): RGB color of the tube.

    Returns:
        o3d.geometry.TriangleMesh: Cylinder representing the tube.
    """
    length = np.linalg.norm(end - start)
    if length < 1e-6:
        return None  # Avoid degenerate cylinders

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
    cylinder.paint_uniform_color(color)

    cylinder.translate((0, 0, length / 2))

    # Compute the transformation to align cylinder with the line segment
    direction = (end - start) / length
    z_axis = np.array([0, 0, 1])
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, 0))  # Identity rotation

    if not np.allclose(direction, z_axis):  # If the direction is not already along Z
        axis = np.cross(z_axis, direction)
        angle = np.arccos(np.dot(z_axis, direction))
        axis /= np.linalg.norm(axis)  # Normalize axis
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    cylinder.rotate(rotation_matrix, center=(0, 0, 0))
    cylinder.translate(start)

    cylinder.compute_vertex_normals()

    return cylinder


def convert_centerline_pc_to_branched_sphere_tubeset( ct_skeleton_pc,closest_indices_ct, ct_centroids, color = [1,0,0]):
    """Create sphere and tube geometry for a branched centerline."""
    ct_skeleton_pc_with_branches, ct_lineset_branches = get_branched_skeleton(ct_skeleton_pc,closest_indices_ct,ct_centroids)
    ct_tubeset  = convert_linesets_to_tubes([ct_lineset_branches], radius =0.001, resolution=3, color=color)
    ct_skeleton_spheres = get_sphere_cloud(np.asarray(ct_skeleton_pc.points), 0.0015, 20, color)
    ct_spheres = get_sphere_cloud(ct_centroids, 0.0015, 20, color)

    return ct_skeleton_spheres, ct_tubeset, ct_spheres


def convert_linesets_to_tubes(linesets, radius=0.001, resolution=10, color=[0,1,0]):
    """
    Convert a list of Open3D LineSets into a list of tubular meshes.

    Args:
        linesets (list): List of o3d.geometry.LineSet objects.
        radius (float): Radius of the tubes.
        resolution (int): Resolution of the cylinders.

    Returns:
        list: List of o3d.geometry.TriangleMesh objects representing the tubes.
    """
    tube_meshes = []

    for lineset in linesets:
        points = np.asarray(lineset.points)
        lines = np.asarray(lineset.lines)

        for line in lines:
            pt1, pt2 = points[line[0]], points[line[1]]
            tube = create_tube_mesh(np.array(pt1), np.array(pt2), radius=radius, resolution=resolution, color=color)
            if tube:
                tube_meshes.append(tube)

    return tube_meshes
