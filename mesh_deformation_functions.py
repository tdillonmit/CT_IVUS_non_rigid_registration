"""Non-rigid centerline and mesh-deformation helpers for CT–IVUS registration."""

import copy
import time
from typing import List, Tuple
import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sksparse.cholmod import cholesky_AAt

from basic_correspondence_matching import *
from visualization_functions import *


def choleskySolve(M, b):
    """Solve sparse least-squares normal equations with CHOLMOD."""
    factor = cholesky_AAt(M.T)
    x = factor(M.T.dot(b))
    return x.toarray() if hasattr(x, "toarray") else x


def lineset_nrcip_extended(ct_skeleton_pc,ivus_skeleton_pc,corres=None, closest_indices_ct=None, closest_indices_ivus=None, visualize_debug =0, beta = 10.0, alphas =  np.linspace(200, 1, 20), gamma = 1000000.0):
    """Non-rigidly register CT and IVUS centerlines while enforcing landmark correspondences."""
    o3d.io.write_point_cloud("/home/tdillon/ct_skeleton_pc_test.ply", ct_skeleton_pc)
    o3d.io.write_point_cloud("/home/tdillon/ivus_skeleton_pc_test.ply", ivus_skeleton_pc)
    np.save("/home/tdillon/closest_indices_ct_test.npy", closest_indices_ct)
    np.save("/home/tdillon/closest_indices_ivus_test.npy", closest_indices_ivus)
    np.save("/home/tdillon/corres_test.npy", corres)

    # find cropping region for after non rigid icp
    if(corres is not None):
        corres = corres[corres[:, 0].argsort()]
        min_corres = np.min(corres[:,0])
        max_corres = np.max(corres[:,0])

        intersected_vertices = np.asarray(ivus_skeleton_pc.points)[corres[:,1]]
        source_landmarks = corres[:,0]
    else:
        intersected_vertices = None
        source_landmarks = None

    ct_points = np.array(np.asarray(ct_skeleton_pc.points))  # Convert to a numpy array if not already

    ct_lineset = o3d.geometry.LineSet()
    ct_lineset.points = o3d.utility.Vector3dVector(ct_points)
    edges = [[i, i+1] for i in range(len(ct_points)-1)]
    ct_lineset.lines = o3d.utility.Vector2iVector(edges)

    ivus_points = np.array(np.asarray(ivus_skeleton_pc.points))  # Convert to a numpy array if not already

    ivus_lineset = o3d.geometry.LineSet()
    ivus_lineset.points = o3d.utility.Vector3dVector(ivus_points)
    edges = [[i, i+1] for i in range(len(ivus_points)-1)]
    ivus_lineset.lines = o3d.utility.Vector2iVector(edges)

    start_time = time.time()

    near_deformed, vertsTransformed_full, D, X, matches, wVec, transform_list = nonrigidIcp_lineset(ct_lineset, ivus_lineset, source_landmarks=source_landmarks, target_landmarks=intersected_vertices, beta = beta, alphas = alphas, gamma = gamma)
    near_deformed.paint_uniform_color([0,1,0])

    if(visualize_debug==1):
        o3d.visualization.draw_geometries([near_deformed, ivus_lineset, ct_lineset])

    end_time = time.time()
    difference = end_time - start_time

    print("in the deformation itself", difference)

    outside_pc = o3d.geometry.PointCloud()
    outside_pc.points = o3d.utility.Vector3dVector(vertsTransformed_full)
    outside_pc.paint_uniform_color([1,0,0])

    start_vector = (vertsTransformed_full[min_corres] - ct_points[min_corres])
    end_vector = (vertsTransformed_full[max_corres] - ct_points[max_corres])
    vertsTransformed_full[:min_corres+1] =  start_vector + ct_points[:min_corres+1]
    vertsTransformed_full[max_corres:] = end_vector + ct_points[max_corres:]

    print("calling new propagation")

    verts_pc =o3d.geometry.PointCloud()
    verts_pc.points = o3d.utility.Vector3dVector(vertsTransformed_full)
    verts_pc.paint_uniform_color([0,0,1])

    if(visualize_debug==1):
        o3d.visualization.draw_geometries([verts_pc, outside_pc])

        # --------- ANIMATE POINT CLOUD DEFORMATION ------ #

    time_points = 20

    sourcelineset = copy.deepcopy(ct_skeleton_pc)  # Starting pc
    targetlineset = copy.deepcopy(ivus_skeleton_pc)  # Target pc

    if(visualize_debug==1):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True

    interp_lineset = copy.deepcopy(sourcelineset)

    center_ct_closest = np.asarray(ct_skeleton_pc.points)[closest_indices_ct]
    centerline_ct_spheres = get_sphere_cloud(center_ct_closest, 0.004, 10, [1,0,0])

    center_ivus_closest = np.asarray(ivus_skeleton_pc.points)[closest_indices_ivus]
    centerline_ivus_spheres = get_sphere_cloud(center_ivus_closest, 0.004, 10, [0,0,1])

    if(visualize_debug==1):
        vis.add_geometry(interp_lineset)
        vis.add_geometry(targetlineset)
        vis.add_geometry(centerline_ct_spheres)
        vis.add_geometry(centerline_ivus_spheres)

        vis.run()

    start_time = time.time()

    source_points = np.array(sourcelineset.points)  # Original source points
    for i in range(time_points):
        t = i / time_points

        # Interpolate between source points and deformed points
        interpolated_points = (1 - t) * source_points + t * vertsTransformed_full

        interpolated_centers = (1 - t) * center_ct_closest + t * (vertsTransformed_full[closest_indices_ct])
        placeholder_spheres = get_sphere_cloud(interpolated_centers, 0.004, 10, [1,0,0])

        interp_lineset.points = o3d.utility.Vector3dVector(interpolated_points)

        centerline_ct_spheres.vertices = placeholder_spheres.vertices
        centerline_ct_spheres.paint_uniform_color([1,0,0])

        if(visualize_debug==1):
            vis.update_geometry(interp_lineset)
            vis.update_geometry(centerline_ct_spheres)

            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)

    if(visualize_debug==1):
        o3d.visualization.draw_geometries([interp_lineset, targetlineset])

    end_time = time.time()
    difference = end_time - start_time
    print("difference animation", difference)

    return ct_lineset, vertsTransformed_full, interp_lineset


def robust_kernel(distances, delta=0.02):
    """Compute robust weights using the Huber kernel."""
    return 1 / (1 + (distances / delta) ** 2)


def nonrigidIcp_lineset(source_lineset, target_lineset, source_landmarks=None, target_landmarks=None,
                         beta=0.1, alphas=np.linspace(200, 1, 20), gamma=1, delta=0.02):
    """Perform affine-per-vertex non-rigid ICP between two line sets."""
    refined_lineset = copy.deepcopy(source_lineset)
    target_vertices = np.array(target_lineset.points)
    source_vertices = np.array(refined_lineset.points)
    n_source_verts = source_vertices.shape[0]

    knnsearch = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_vertices)
    source_edges = np.array(refined_lineset.lines)
    n_source_edges = len(source_edges)

    M = sparse.lil_matrix((n_source_edges, n_source_verts), dtype=np.float32)
    for i, edge in enumerate(source_edges):
        M[i, edge[0]] = -1
        M[i, edge[1]] = 1

    G = np.diag([1, 1, 1, gamma]).astype(np.float32)
    kron_M_G = sparse.kron(M, G)

    D = sparse.lil_matrix((n_source_verts, n_source_verts*4), dtype=np.float32)
    j_ = 0
    for i in range(n_source_verts):
        D[i, j_:j_+3] = source_vertices[i, :]
        D[i, j_+3] = 1
        j_ += 4

    X_ = np.concatenate((np.eye(3), np.array([[0, 0, 0]])), axis=0)
    X = np.tile(X_, (n_source_verts, 1))

    if source_landmarks is not None and target_landmarks is not None:
        assert len(source_landmarks) == len(target_landmarks), "Source and target landmarks must have the same length."
        n_landmarks = len(source_landmarks)
        DL = sparse.lil_matrix((n_landmarks, n_source_verts*4), dtype=np.float32)

        for i, lm in enumerate(source_landmarks):
            DL[i, 4*lm:4*lm+4] = np.append(source_vertices[lm], 1)

        UL = target_landmarks
    else:
        DL = None
        UL = None

    for num_, alpha_stiffness in enumerate(alphas):
        print(f"Step {num_ + 1}/20")

        # more efficient
        stiff_block = alpha_stiffness * kron_M_G
        stiff_rows = 4 * n_source_edges
        data_rows = n_source_verts

        if DL is not None and UL is not None:
            lm_block = beta * DL
            total_rows = stiff_rows + data_rows + n_landmarks
            UL_beta = beta * UL
        else:
            total_rows = stiff_rows + data_rows

        for _ in range(3):
            vertsTransformed = D @ X

            distances, indices = knnsearch.kneighbors(vertsTransformed)
            indices = indices.ravel()
            matches = target_vertices[indices]

            wVec = robust_kernel(distances, delta).reshape(-1, 1).astype(np.float32)
            U = wVec * matches

            Dw = D.multiply(wVec)

            if DL is not None and UL is not None:
                A = sparse.vstack([stiff_block, Dw, lm_block], format="csr")

                B = np.zeros((total_rows, 3), dtype=np.float32)
                B[stiff_rows:stiff_rows + data_rows, :] = U
                B[stiff_rows + data_rows:, :] = UL_beta
            else:
                A = sparse.vstack([stiff_block, Dw], format="csr")

                B = np.zeros((total_rows, 3), dtype=np.float32)
                B[stiff_rows:stiff_rows + data_rows, :] = U

            X = choleskySolve(A, B)

    print("X is", X)
    print("D is", D)

    n_source_verts = source_vertices.shape[0]
    transform_list = extract_4x4_transforms_with_condition(X, n_source_verts, wVec)
    vertsTransformed = D * X
    refined_lineset.points = o3d.utility.Vector3dVector(vertsTransformed)

    return refined_lineset, vertsTransformed, D, X, matches, wVec, transform_list


def extract_4x4_transforms_with_condition(X, n_source_verts, wVec):
    """
    Extracts 4x4 transformation matrices from X for each vertex in the LineSet,
    assigning identity matrices where the wVec condition is not met.

    Args:
        X: The 4n x 3 matrix of affine transformations.
        n_source_verts: Number of vertices in the source LineSet.
        wVec: Weight vector indicating valid correspondences (1 for valid, 0 for invalid).

    Returns:
        transforms: List of 4x4 numpy arrays representing the transformations.
    """
    transforms = []
    for i in range(n_source_verts):

            # Extract the 4x3 block corresponding to vertex i
        transform_4x3 = X[4 * i:4 * i + 4, :]

        transform_4x4 = np.eye(4)
        transform_4x4[:3, :3] = transform_4x3[:3, :].T  # Transpose the 3x3 rotation/scale
        transform_4x4[:3, 3] = transform_4x3[3, :]       # Translation vector
        #     # Set to identity matrix if correspondence is invalid

        transforms.append(transform_4x4)

    return transforms


def quantify_proportion_pc_inside_mesh(original_pc, original_mesh):
    """Measure the fraction of point-cloud samples inside a triangle mesh."""
    colored_pc = copy.deepcopy(original_pc)

    to_deform_legacy = o3d.t.geometry.TriangleMesh.from_legacy(original_mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(to_deform_legacy)  # we do not need the geometry ID for mesh

    query_points = np.asarray(colored_pc.points)

    query_points_tensor = o3d.core.Tensor(query_points, dtype=o3d.core.Dtype.Float32)

    # Compute the signed distance for N random points
    signed_distance = scene.compute_signed_distance(query_points_tensor)

    # Convert the signed distance tensor to a NumPy array for easier indexing
    signed_distance_np = signed_distance.numpy()  # Convert to NumPy array

    colors = np.zeros((len(query_points), 3))  # Create an array for colors (N x 3)
    colors[signed_distance_np < 0] = [0, 1, 0]  # Green for points inside (negative distance)
    colors[signed_distance_np > 0] = [1, 0, 0]  # Red for points outside (positive distance)

    colored_pc.colors = o3d.utility.Vector3dVector(colors)

    points_inside = np.sum(signed_distance_np < 0)  # Count points with negative signed distance
    proportion_inside = points_inside / len(query_points)  # Proportion of points inside

    return proportion_inside, colored_pc


def quantify_inlier_points_clustered(to_deform, mapped_pcs, relevant_nodes_list, semantic_corres_ransac_sub, closest_indices_ct, closest_indices_ivus, aneurysm = 0, ct_half_index=None):
    """Score clustered IVUS branch points against corresponding regions of the deformed CT mesh."""
    # need to find relevant nodes and mapped pc for given semantic corres_ransac_sub

    colored_pcs = []
    points_inside_total = 0
    captures = 0

    ls_check = create_wireframe_lineset_from_mesh(to_deform)

    for corres_pair in semantic_corres_ransac_sub:

        if(aneurysm==1):
            if(corres_pair[0]>ct_half_index):
                continue

        ct_centroid_index = np.argwhere(closest_indices_ct == corres_pair[0]).squeeze()
        ivus_centroid_index = np.argwhere(closest_indices_ivus == corres_pair[1]).squeeze()

        relevant_nodes = relevant_nodes_list[ct_centroid_index]
        mapped_pc = mapped_pcs[ivus_centroid_index]

        query_points = np.asarray(mapped_pc.points)
        no_query_points = np.shape(query_points)[0]

        # # proportion close enough
        relevant_points = np.asarray(to_deform.vertices)
        relevant_points = relevant_points[relevant_nodes,:]

        tree = cKDTree(relevant_points)
        dists, idx = tree.query(query_points, k=1, workers=-1)  # nearest neighbor for each query point

        closest_pts = relevant_points[idx]              # (n, 3) the matched nearest points
        offset_vecs = closest_pts - query_points       # (n, 3) vectors from query -> nearest

        inlier_threshold = 0.005

        mask_2 = np.linalg.norm(offset_vecs, axis=1) < inlier_threshold

        mask = mask_2 # NB THAT THIS IS HERE!
        points_inside = np.sum(mask)  # number of inliers

        colored_pc = copy.deepcopy(mapped_pc)

        colors = np.zeros((len(query_points), 3))  # Create an array for colors (N x 3)
        colors[mask == True] = [0, 1, 0]  # Green for points inside (negative distance)
        colors[mask == False] = [1, 0, 0]  # Red for points outside (positive distance)

        colored_pc.colors = o3d.utility.Vector3dVector(colors)

        colored_pcs.append(colored_pc)

        relevant_pc = o3d.geometry.PointCloud()
        relevant_pc.points = o3d.utility.Vector3dVector(relevant_points)

        percentage_inliers = (np.sum(mask) / no_query_points)

        capture_threshold = 0.4
        if(percentage_inliers> capture_threshold):
            captures = captures + no_query_points

        points_inside_total = points_inside_total + points_inside

    return points_inside_total, captures, colored_pcs


def fast_aortascope_deformation(
    mesh_smp: o3d.geometry.TriangleMesh,
    ct_lineset: o3d.geometry.LineSet,
    vertsTransformed_full: np.ndarray,
    ivus_skeleton_pc: o3d.geometry.PointCloud,
    ivus_spheres: o3d.geometry.TriangleMesh,
    ct_side_branch_pc: o3d.geometry.PointCloud,
    projection_data: List[Tuple[int, int, float, float]],
    mesh_downsample: int = 1,
    visualize_debug: bool = False,
    time_points: int = 30

) -> tuple[
    o3d.geometry.TriangleMesh,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud
]:
    """
    Creates a simplified mesh from ct_slicer_mesh, builds a red 'to_deform'
    mesh+lineset, sets up canonical_flap_pc, then animates a smooth projection
    from the canonical to the full-transformed vertices. At each time step it
    deforms both the main mesh and the side-branch pointclouds.

    Returns:
      to_deform:                the red simplified mesh (last deformed state)
      canonical_flap_pc:        the flap pointcloud (last interpolated state)
      ct_side_branch_pc:        the side-branch pointcloud (last deformed state)
      side_branch_centrelines_pc: the side-branch centrelines (last deformed state)
    """

    # 2) prepare outputs
    to_deform = copy.deepcopy(mesh_smp)
    to_deform.paint_uniform_color([1, 0, 0])
    to_deform.compute_vertex_normals()
    to_deform_lineset = create_wireframe_lineset_from_mesh(to_deform)
    to_deform_lineset.paint_uniform_color([1, 0, 0])

    canonical_flap_pc = o3d.geometry.PointCloud()
    canonical_flap_pc.points = ct_lineset.points

    if visualize_debug:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True
        vis.add_geometry(to_deform_lineset)
        ivus_skeleton_pc.paint_uniform_color([0, 0, 1])
        vis.add_geometry(ivus_skeleton_pc)
        canonical_flap_pc.paint_uniform_color([1, 0, 0])
        vis.add_geometry(canonical_flap_pc)
        vis.add_geometry(ivus_spheres)
        vis.run()

    # 3) prepare interpolation targets
    base_pts = np.asarray(ct_lineset.points)
    target_pts = vertsTransformed_full
    desired_locations = [
        base_pts + (i / time_points) * (target_pts - base_pts)
        for i in range(time_points + 1)
    ]

    start_time = time.time()

    # 4) animation loop
    for i, desired in enumerate(desired_locations):
        disp = desired - np.asarray(canonical_flap_pc.points)

        sub_mesh = deform_with_cached_projection(to_deform, disp, projection_data)

        canonical_flap_pc.points = o3d.utility.Vector3dVector(desired)

        new_ls = create_wireframe_lineset_from_mesh(sub_mesh)
        to_deform_lineset.points = new_ls.points
        to_deform.vertices      = sub_mesh.vertices

        if visualize_debug:
            vis.update_geometry(to_deform_lineset)
            vis.update_geometry(canonical_flap_pc)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)

    return to_deform


def improved_animate_aortascope_deformation(
    mesh_smp: o3d.geometry.TriangleMesh,
    ct_lineset: o3d.geometry.LineSet,
    vertsTransformed_full: np.ndarray,
    ivus_skeleton_pc: o3d.geometry.PointCloud,
    ivus_spheres: o3d.geometry.TriangleMesh,
    ct_spheres: o3d.geometry.TriangleMesh,
    ct_side_branch_pc: o3d.geometry.PointCloud,
    side_branch_centrelines_pc: o3d.geometry.PointCloud,
    mesh_downsample: int = 1,
    visualize_debug: bool = False,
    time_points: int = 30,

    corres=np.ndarray,
    corres_original=np.ndarray,
    closest_indices_ct=np.ndarray,
    closest_indices_ivus=np.ndarray,
    ivus_centroids = np.ndarray,
    near_mesh = o3d.geometry.TriangleMesh,
    far_pc = o3d.geometry.PointCloud

) -> tuple[
    o3d.geometry.TriangleMesh,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud
]:
    """
    Creates a simplified mesh from ct_slicer_mesh, builds a red 'to_deform'
    mesh+lineset, sets up canonical_flap_pc, then animates a smooth projection
    from the canonical to the full-transformed vertices. At each time step it
    deforms both the main mesh and the side-branch pointclouds.

    Returns:
      to_deform:                the red simplified mesh (last deformed state)
      canonical_flap_pc:        the flap pointcloud (last interpolated state)
      ct_side_branch_pc:        the side-branch pointcloud (last deformed state)
      side_branch_centrelines_pc: the side-branch centrelines (last deformed state)
    """

    # 2) prepare outputs
    to_deform = copy.deepcopy(mesh_smp)
    to_deform.paint_uniform_color([1, 0, 0])
    to_deform.compute_vertex_normals()
    to_deform_lineset = create_wireframe_lineset_from_mesh(to_deform)
    to_deform_lineset.paint_uniform_color([1, 0, 0])

    canonical_flap_pc = o3d.geometry.PointCloud()
    canonical_flap_pc.points = ct_lineset.points

    if visualize_debug:

        ct_centroids = np.asarray(ct_side_branch_pc.points)
        ct_spheres = get_sphere_cloud(ct_centroids, 0.0015, 10, [1,0,0])
        ivus_spheres = get_sphere_cloud(ivus_centroids, 0.0015, 10, [0,0,1])
        ct_tubes_branches, ivus_tubes_branches, corres_tubes, corres_tubes_branches, ivus_tubes_branches_orig, ct_skeleton_spheres, ivus_skeleton_spheres_orig = show_tubeset_correspondences( corres,corres_original, closest_indices_ct,closest_indices_ivus, ct_lineset,ivus_skeleton_pc,ct_centroids,ivus_centroids,ct_spheres, ivus_spheres, 1)

        # add near and far pc
        near_pc = o3d.geometry.PointCloud()
        near_pc.points = near_mesh.vertices
        near_pc.paint_uniform_color([0,0,1])
        far_pc.paint_uniform_color([0,0,1])

        # add tubesets for ct and ivus

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True
        vis.add_geometry(to_deform_lineset)
        vis.add_geometry(near_pc) # for figures
        vis.add_geometry(far_pc)
        ivus_skeleton_pc.paint_uniform_color([0, 0, 1])
        vis.add_geometry(ivus_skeleton_pc)
        canonical_flap_pc.paint_uniform_color([1, 0, 0])
        vis.add_geometry(canonical_flap_pc)
        vis.add_geometry(ivus_spheres)
        vis.add_geometry(ct_spheres)
        vis.run()

    # 3) prepare interpolation targets
    base_pts = np.asarray(ct_lineset.points)
    target_pts = vertsTransformed_full
    desired_locations = [
        base_pts + (i / time_points) * (target_pts - base_pts)
        for i in range(time_points + 1)
    ]

    start_time = time.time()

    # 4) animation loop
    for i, desired in enumerate(desired_locations):
        disp = desired - np.asarray(canonical_flap_pc.points)

        # deform main mesh + spheres + side branches

        sub_mesh       = deform_mesh_with_projection(to_deform, disp, canonical_flap_pc)

        sub_spheres    = deform_mesh_with_projection(ct_spheres, disp, canonical_flap_pc)
        ct_side_branch_pc = deform_side_branches_with_projection(ct_side_branch_pc, disp, canonical_flap_pc)
        side_branch_centrelines_pc = deform_side_branches_with_projection(
            side_branch_centrelines_pc, disp, canonical_flap_pc
        )

        # update canonical flap to the new interpolated points
        canonical_flap_pc.points = o3d.utility.Vector3dVector(desired)

        new_ls = create_wireframe_lineset_from_mesh(sub_mesh)
        to_deform_lineset.points = new_ls.points
        to_deform.vertices      = sub_mesh.vertices

        p1 = np.asarray(ct_side_branch_pc.points).squeeze()
        temp_ct_spheres = get_sphere_cloud(p1, 0.004, 10, [1,0,0])
        ct_spheres.vertices = temp_ct_spheres.vertices
        ct_spheres.triangles = temp_ct_spheres.triangles

        if visualize_debug:
            vis.update_geometry(to_deform_lineset)
            vis.update_geometry(canonical_flap_pc)
            vis.update_geometry(ct_spheres)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)

    time.sleep(1.0)

    end_time = time.time()
    difference_time = end_time-start_time
    print("total deformation time", difference_time)

    return to_deform, canonical_flap_pc, ct_side_branch_pc, side_branch_centrelines_pc, desired_locations


def animate_aortascope_deformation(
    mesh_smp: o3d.geometry.TriangleMesh,
    ct_lineset: o3d.geometry.LineSet,
    vertsTransformed_full: np.ndarray,
    ivus_skeleton_pc: o3d.geometry.PointCloud,
    ivus_spheres: o3d.geometry.TriangleMesh,
    ct_spheres: o3d.geometry.TriangleMesh,
    ct_side_branch_pc: o3d.geometry.PointCloud,
    side_branch_centrelines_pc: o3d.geometry.PointCloud,
    mesh_downsample: int = 1,
    visualize_debug: bool = False,
    time_points: int = 30,
) -> tuple[
    o3d.geometry.TriangleMesh,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud
]:
    """
    Creates a simplified mesh from ct_slicer_mesh, builds a red 'to_deform'
    mesh+lineset, sets up canonical_flap_pc, then animates a smooth projection
    from the canonical to the full-transformed vertices. At each time step it
    deforms both the main mesh and the side-branch pointclouds.

    Returns:
      to_deform:                the red simplified mesh (last deformed state)
      canonical_flap_pc:        the flap pointcloud (last interpolated state)
      ct_side_branch_pc:        the side-branch pointcloud (last deformed state)
      side_branch_centrelines_pc: the side-branch centrelines (last deformed state)
    """

    # 2) prepare outputs
    to_deform = copy.deepcopy(mesh_smp)
    to_deform.paint_uniform_color([1, 0, 0])
    to_deform.compute_vertex_normals()
    to_deform_lineset = create_wireframe_lineset_from_mesh(to_deform)
    to_deform_lineset.paint_uniform_color([1, 0, 0])

    canonical_flap_pc = o3d.geometry.PointCloud()
    canonical_flap_pc.points = ct_lineset.points

    if visualize_debug:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True
        vis.add_geometry(to_deform_lineset)
        ivus_skeleton_pc.paint_uniform_color([0, 0, 1])
        vis.add_geometry(ivus_skeleton_pc)
        canonical_flap_pc.paint_uniform_color([1, 0, 0])
        vis.add_geometry(canonical_flap_pc)
        vis.add_geometry(ivus_spheres)
        vis.add_geometry(ct_spheres)
        vis.run()

    # 3) prepare interpolation targets
    base_pts = np.asarray(ct_lineset.points)
    target_pts = vertsTransformed_full
    desired_locations = [
        base_pts + (i / time_points) * (target_pts - base_pts)
        for i in range(time_points + 1)
    ]

    start_time = time.time()

    # 4) animation loop
    for i, desired in enumerate(desired_locations):
        disp = desired - np.asarray(canonical_flap_pc.points)

        # deform main mesh + spheres + side branches
        sub_mesh       = deform_mesh_with_projection(to_deform, disp, canonical_flap_pc)

        sub_spheres    = deform_mesh_with_projection(ct_spheres, disp, canonical_flap_pc)
        ct_side_branch_pc = deform_side_branches_with_projection(ct_side_branch_pc, disp, canonical_flap_pc)
        side_branch_centrelines_pc = deform_side_branches_with_projection(
            side_branch_centrelines_pc, disp, canonical_flap_pc
        )

        # update canonical flap to the new interpolated points
        canonical_flap_pc.points = o3d.utility.Vector3dVector(desired)

        new_ls = create_wireframe_lineset_from_mesh(sub_mesh)
        to_deform_lineset.points = new_ls.points
        to_deform.vertices      = sub_mesh.vertices

        p1 = np.asarray(ct_side_branch_pc.points).squeeze()
        temp_ct_spheres = get_sphere_cloud(p1, 0.004, 10, [1,0,0])
        ct_spheres.vertices = temp_ct_spheres.vertices
        ct_spheres.triangles = temp_ct_spheres.triangles

        if visualize_debug:
            vis.update_geometry(to_deform_lineset)
            vis.update_geometry(canonical_flap_pc)
            vis.update_geometry(ct_spheres)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.01)

    end_time = time.time()
    difference_time = end_time-start_time
    print("total deformation time", difference_time)

    return to_deform, canonical_flap_pc, ct_side_branch_pc, side_branch_centrelines_pc, desired_locations


def slide_and_twist_branches(mesh, ct_centroids,ivus_centroids,corres_original,canonical_flap_pc,side_branch_centrelines_pc,slide_increment, twist_increment, side_branch_centrelines_indices, orig_branch_pc,visualize_debug=False, node_pool_distance=0.0095):
    """Refine registered branch locations by sliding and twisting branch neighborhoods."""
    max_slide = 0.010

    max_twist = 4*3.14/10

    slide_numbers = []
    ct_centroids_slided=[]
    slide_signages = []

    # first fix the canonical centreline
    smoothingPoints = 70
    smoothness_spline = 0.0001
    canonical_flap_pc_points = np.asarray(canonical_flap_pc.points)

    b_spline=fit_3D_b_spline(canonical_flap_pc_points, numPoints=smoothingPoints, smoothness = smoothness_spline)
    canonical_flap_pc.points = o3d.utility.Vector3dVector(b_spline)

    twist_numbers = []
    ct_centroids_rotated = []
    twist_signages = []

    side_branch_centrelines = np.asarray(side_branch_centrelines_pc.points)
    side_branch_centrelines_indices = np.squeeze(side_branch_centrelines_indices)

    # pull out which branch corresponds to which centroid (before considering ivus or correspondences at all)
    unique_branches = np.unique(side_branch_centrelines_indices)
    branch_corres = np.empty((0,4))
    for j in unique_branches:
        relevant_args = np.argwhere(side_branch_centrelines_indices == j)
        relevant_args = np.squeeze(relevant_args)
        side_branch_centreline = side_branch_centrelines[relevant_args,:]
        first_point = side_branch_centreline[0,:]
        row = np.hstack((j, first_point))
        print("first_point", first_point)
        branch_corres = np.vstack((branch_corres,row))

    print("branch_corres[:,1:3]", branch_corres[:,1:3])
    centroid_branches = []

    for ct_centroid in ct_centroids:
        print("branch corres 1 -4", branch_corres[:,1:4])
        print("ct_centroid", ct_centroid)
        arg_minimizes= np.argmin(np.linalg.norm(branch_corres[:,1:4]-ct_centroid, axis=1))
        centroid_branch = branch_corres[arg_minimizes,0]
        centroid_branches.append(centroid_branch)

    centroid_branches = np.asarray(centroid_branches)

    # Precompute
    side_branch_centrelines_pc_points = np.asarray(side_branch_centrelines_pc.points)

    print("pooling distance is:", node_pool_distance)
    # 1. Collect all sublists
    all_nodes_list = []
    branch_points_list = []
    for check_index in np.unique(side_branch_centrelines_indices):
        relevant_args = np.argwhere(side_branch_centrelines_indices == check_index)[:, 0]
        relevant_branch_pts = side_branch_centrelines_pc_points[relevant_args, :]
        relevant_nodes_sub = get_all_nodes_inside_radius(
            relevant_branch_pts, node_pool_distance, mesh
        )
        all_nodes_list.append(np.array(relevant_nodes_sub))
        branch_points_list.append(relevant_branch_pts)

    # 2. Flatten to find duplicates
    all_nodes_flat = np.concatenate(all_nodes_list)
    unique_nodes, counts = np.unique(all_nodes_flat, return_counts=True)
    overlapping_nodes = unique_nodes[counts > 1]

    # 3. Build KD-trees for each branch to compute distances
    branch_trees = [cKDTree(pts) for pts in branch_points_list]

    # 4. Assign overlaps to closest branch only
    final_nodes_list = []
    for i, nodes in enumerate(all_nodes_list):
        # Keep nodes unique to this branch for now
        keep_nodes = set(nodes) - set(overlapping_nodes)
        final_nodes_list.append(keep_nodes)

    # Resolve overlaps
    for node in overlapping_nodes:
        # find closest branch centerline
        node_xyz = mesh.vertices[node]  # assuming mesh.vertices is an (N,3) array
        dists = [tree.query(node_xyz)[0] for tree in branch_trees]
        best_idx = np.argmin(dists)
        final_nodes_list[best_idx].add(node)

    # 5. Convert back to list format (if you need arrays)
    relevant_nodes_list = [np.array(sorted(list(nodes))) for nodes in final_nodes_list]

    euclidean_errors = []

    for correspondence in corres_original:

        p1 = ct_centroids[correspondence[0],:]
        p2 = ivus_centroids[correspondence[1],:]

        min_intra_axial = (get_intra_axial_distance( p1, p2, canonical_flap_pc))

        slide_signage = np.sign(min_intra_axial)

        print("min intra axial", min_intra_axial)

        num_slides = 0
        p1_slided = p1
        while((slide_increment * (num_slides+1)) < max_slide):

            p1 = slide_point_along_centreline(p1, canonical_flap_pc, slide_signage*slide_increment)
            intra_axial = get_intra_axial_distance( p1, p2,canonical_flap_pc)
            print("intra axial", intra_axial)

            if(abs(intra_axial) < abs(min_intra_axial)):
                min_intra_axial = intra_axial
                num_slides=num_slides+1
                p1_slided = p1

            else:
                break

        if((slide_increment * (num_slides+1)) >= max_slide):
            print("exceeded limit!")

        slide_numbers.append(num_slides)
        slide_signages.append(slide_signage)

        print("SLIDE NUMBERS", slide_numbers)

        p1 = p1_slided

        min_intra_circum = (get_intra_circum_distance( p1, p2, canonical_flap_pc))
        twist_signage = np.sign(min_intra_circum)
        num_twists = 0
        p1_rotated = p1
        while((twist_increment * (num_twists+1)) < max_twist):

            p1 = twist_point_about_centreline( p1, canonical_flap_pc, twist_signage*twist_increment )
            intra_circum = get_intra_circum_distance( p1, p2,canonical_flap_pc)

            if(abs(intra_circum) < abs(min_intra_circum)):
                min_intra_circum = intra_circum

                euclidean_error = np.linalg.norm(p1 - p2)
                num_twists = num_twists+1
                p1_rotated = p1

            else:

                break

        if((twist_increment * (num_twists+1)) >= max_twist):
            print("exceeded limit!")

        twist_numbers.append(num_twists)
        ct_centroids_rotated.append(p1_rotated)
        twist_signages.append(twist_signage)
        euclidean_errors.append(euclidean_error)

    ct_centroids_rotated = np.asarray(ct_centroids_rotated)

    mesh.compute_vertex_normals()

    mesh_lineset = create_wireframe_lineset_from_mesh(mesh)

    ivus_spheres = get_sphere_cloud(ivus_centroids, 0.004, 10, [0,0,1])
    ct_spheres = get_sphere_cloud(ct_centroids, 0.004, 10, [1,0,0])

    if visualize_debug:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().mesh_show_back_face = True
        vis.add_geometry(mesh_lineset)
        vis.add_geometry(side_branch_centrelines_pc)
        vis.add_geometry(canonical_flap_pc)
        vis.add_geometry(ivus_spheres)
        vis.add_geometry(ct_spheres)
        vis.run()

    sub_mesh = o3d.geometry.TriangleMesh()
    sub_mesh.vertices = copy.deepcopy(mesh.vertices)
    sub_mesh.triangles = copy.deepcopy(mesh.triangles)
    sub_mesh.paint_uniform_color([1,0,0])

    sub_side_branch_centerlines = o3d.geometry.TriangleMesh()
    sub_side_branch_centerlines.vertices = copy.deepcopy(side_branch_centrelines_pc.points)

    side_branch_centrelines_indices = np.squeeze(side_branch_centrelines_indices)

    new_ct_centroids = copy.deepcopy(ct_centroids)

    for correspondence, num_slides,num_twists, twist_signage, slide_signage in zip(corres_original, slide_numbers,twist_numbers, twist_signages, slide_signages):

        # skip certain branches

        centroid_branch = centroid_branches[correspondence[0]]

        # pull out relevant centerline
        relevant_args = np.argwhere(side_branch_centrelines_indices == centroid_branch)
        relevant_args = np.squeeze(relevant_args)
        side_branch_centreline = side_branch_centrelines[relevant_args,:]

        # old way of getting relevant mesh nodes

        # better to precompute Oct 25
        relevant_nodes = relevant_nodes_list[correspondence[0]]

        p1 = ct_centroids[correspondence[0],:]

        p2 = ivus_centroids[correspondence[1],:]
        sub_ct_point = o3d.geometry.TriangleMesh()
        sub_ct_point.vertices = o3d.utility.Vector3dVector([p1])

        sub_ivus_point = o3d.geometry.TriangleMesh()
        sub_ivus_point.vertices = o3d.utility.Vector3dVector([p2])

        slides = 0

        # new Aug 25
        proportion_inside_before, colored_adjust_pc = quantify_proportion_pc_inside_mesh(orig_branch_pc, sub_mesh)
        sub_side_branch_centerlines_before = copy.deepcopy(sub_side_branch_centerlines)
        p1_before = copy.deepcopy(p1)
        sub_mesh_before = copy.deepcopy(sub_mesh)

        # apply slide
        while slides < num_slides:

            # need to deform side_branch_centerlines with it
            sub_mesh = slide_mesh_along_centreline(sub_mesh, canonical_flap_pc, slide_increment*slide_signage, relevant_nodes)
            sub_side_branch_centerlines = slide_mesh_along_centreline(sub_side_branch_centerlines, canonical_flap_pc, slide_increment*slide_signage, relevant_args)

            sub_ct_point = slide_mesh_along_centreline(sub_ct_point, canonical_flap_pc, slide_increment*slide_signage)

            slides = slides + 1

            p1 = np.asarray(sub_ct_point.vertices).squeeze()
            temp_ct_spheres = get_sphere_cloud([p1], 0.004, 10, [1,0,0])
            ct_spheres.vertices = temp_ct_spheres.vertices
            ct_spheres.triangles = temp_ct_spheres.triangles
            ct_spheres.paint_uniform_color([1,0,0])
            ct_spheres.compute_vertex_normals()

            temp_ivus_spheres = get_sphere_cloud([p2], 0.004, 10, [1,0,0])
            ivus_spheres.vertices = temp_ivus_spheres.vertices
            ivus_spheres.triangles = temp_ivus_spheres.triangles
            ivus_spheres.paint_uniform_color([0,0,1])
            ivus_spheres.compute_vertex_normals()

            mesh_lineset.points = sub_mesh.vertices
            side_branch_centrelines_pc.points = sub_side_branch_centerlines.vertices

            if(visualize_debug == True):
                vis.update_geometry(mesh_lineset)
                vis.update_geometry(side_branch_centrelines_pc)
                vis.update_geometry(ct_spheres)
                vis.update_geometry(ivus_spheres)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)

        twists = 0

        # apply twist
        while twists < num_twists:

            sub_mesh = rotate_mesh_about_centreline(sub_mesh, canonical_flap_pc, twist_increment*twist_signage, relevant_nodes)
            sub_side_branch_centerlines = rotate_mesh_about_centreline(sub_side_branch_centerlines, canonical_flap_pc, twist_increment*twist_signage, relevant_args)
            twists = twists + 1

            sub_ct_point = rotate_mesh_about_centreline(sub_ct_point, canonical_flap_pc, twist_increment*twist_signage)

            mesh_lineset.points = sub_mesh.vertices
            side_branch_centrelines_pc.points = sub_side_branch_centerlines.vertices

            p1 = np.asarray(sub_ct_point.vertices).squeeze()
            temp_ct_spheres = get_sphere_cloud([p1], 0.004, 10, [1,0,0])
            ct_spheres.vertices = temp_ct_spheres.vertices
            ct_spheres.triangles = temp_ct_spheres.triangles
            ct_spheres.paint_uniform_color([1,0,0])
            ct_spheres.compute_vertex_normals()

            if(visualize_debug == True):
                vis.update_geometry(mesh_lineset)
                vis.update_geometry(side_branch_centrelines_pc)
                vis.update_geometry(ct_spheres)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.05)

        # did that branch adjustment actually help? if not, skip it
        proportion_inside_after, colored_adjust_pc = quantify_proportion_pc_inside_mesh(orig_branch_pc, sub_mesh)

        if(proportion_inside_after > proportion_inside_before):

            new_ct_centroids[correspondence[0], :] = p1

        else:

            sub_side_branch_centerlines = sub_side_branch_centerlines_before
            side_branch_centrelines_pc.points = sub_side_branch_centerlines.vertices
            sub_mesh = sub_mesh_before

            # undo everything

    new_ct_centroids = np.asarray(new_ct_centroids)

    # add original centroids

    return sub_mesh, new_ct_centroids, side_branch_centrelines_pc, euclidean_errors


def compute_triangle_stretch(mesh: o3d.geometry.TriangleMesh,
                             ref_mesh: o3d.geometry.TriangleMesh,
                             eps: float = 1e-12) -> np.ndarray:
    """
    Compute per-triangle stretch by comparing triangle edge lengths
    between two meshes.

    Stretch for each triangle is:
        || [e0/ref_e0, e1/ref_e1, e2/ref_e2] ||_2

    Args:
        mesh: deformed mesh
        ref_mesh: reference mesh with same topology
        eps: small value to avoid division by zero

    Returns:
        stretch_factors: shape (num_triangles,)
    """
    triangles = np.asarray(mesh.triangles)          # (T, 3)
    vertices = np.asarray(mesh.vertices)            # (V, 3)
    ref_vertices = np.asarray(ref_mesh.vertices)    # (V, 3)

    # Gather triangle vertex coordinates: (T, 3, 3)
    tri_pts = vertices[triangles]
    ref_tri_pts = ref_vertices[triangles]

    # Edge vectors for each triangle
    edge_vecs = np.stack([
        tri_pts[:, 1] - tri_pts[:, 0],
        tri_pts[:, 2] - tri_pts[:, 1],
        tri_pts[:, 0] - tri_pts[:, 2],
    ], axis=1)   # (T, 3, 3)

    ref_edge_vecs = np.stack([
        ref_tri_pts[:, 1] - ref_tri_pts[:, 0],
        ref_tri_pts[:, 2] - ref_tri_pts[:, 1],
        ref_tri_pts[:, 0] - ref_tri_pts[:, 2],
    ], axis=1)   # (T, 3, 3)

    # Edge lengths: (T, 3)
    edge_lengths = np.linalg.norm(edge_vecs, axis=2)
    ref_edge_lengths = np.linalg.norm(ref_edge_vecs, axis=2)

    # Ratios and per-triangle L2 norm
    ratios = edge_lengths / np.maximum(ref_edge_lengths, eps)
    stretch_factors = np.linalg.norm(ratios, axis=1)

    return stretch_factors


def get_all_nodes_inside_radius(centroids, radius, mesh):
    """
    Finds all nodes on a mesh that are within a specified radius of given centroids.

    Parameters:
        centroids (numpy.ndarray): Array of shape (n, 3) containing the centroid coordinates.
        radius (float): Radius within which to search for nodes.
        mesh (o3d.geometry.TriangleMesh): The registered mesh to search nodes in.

    Returns:
        dict: A dictionary where keys are centroid indices, and values are lists of mesh node indices within the radius.
    """
    vertices = np.asarray(mesh.vertices)

    # Build a KDTree for the mesh vertices
    kdtree = o3d.geometry.KDTreeFlann(mesh)

    # Dictionary to store the result
    result = {}

    for i, centroid in enumerate(centroids):
        # Query the KDTree for all points within the radius
        [_, idxs, _] = kdtree.search_radius_vector_3d(centroid, radius)

        # Store the indices in the result dictionary
        result[i] = idxs  # idxs is a list of indices of vertices within the radius

    combined_list = []
    for key, int_vector in result.items():
        combined_list.extend(list(int_vector))  # Convert IntVector to list and extend
    result = np.array(combined_list)

    return result


def get_intra_axial_distance( p1,p2, canonical_flap_pc):
    """Measure signed displacement along a centerline between two projected locations."""
    # Extract centerline points
    centerline_points = np.asarray(canonical_flap_pc.points)

    centerline_directions = np.diff(centerline_points, axis=0)
    centerline_directions = np.vstack([centerline_directions, centerline_directions[-1]])  # Extend last direction for boundary
    centerline_directions = np.array([d / np.linalg.norm(d) for d in centerline_directions])  # Normalize all directions

    p_list = [p1,p2]

    geodesics = compute_geodesic_distance_on_point_cloud(canonical_flap_pc)

    z_distances = []

    for p in p_list:

        print("p is", p)

        # Calculate distances to centerline points and find the nearest segment
        diff = centerline_points - p
        dist = np.linalg.norm(diff, axis=1)
        nearest_idx = np.argmin(dist)

        # Determine the two nearest points on the centerline to define the segment
        if nearest_idx == 0:
            next_idx = 1
        elif nearest_idx == len(centerline_points) - 1:
            next_idx = nearest_idx - 1
        else:
            next_idx = nearest_idx + 1 if dist[nearest_idx + 1] < dist[nearest_idx - 1] else nearest_idx - 1

        geodesic_start = geodesics[nearest_idx]
        geodesic_end = geodesics[next_idx]

        # Define the segment start and end
        segment_start = centerline_points[nearest_idx]
        segment_end = centerline_points[next_idx]
        segment_direction = segment_end - segment_start
        segment_length = np.linalg.norm(segment_direction)
        segment_direction /= segment_length  # Normalize the direction

        # Ensure the sliding direction aligns consistently with the precomputed centerline directions

        # Project the vertex onto the segment and compute the interpolation factor
        projection = np.dot(p - segment_start, segment_direction)
        projection_factor = projection / segment_length

        # Interpolate the sliding direction based on the projection factor
        interpolated_geodesic = (1 - projection_factor) * geodesic_start + projection_factor * geodesic_end

        z_distances.append(interpolated_geodesic)

    print("z distance 1",z_distances[1] )
    print("z distance 0",z_distances[0] )

    intra_axial = z_distances[1]-z_distances[0]

    return intra_axial


def get_intra_circum_distance( p1,p2, canonical_flap_pc):
    """Measure angular displacement around the local centerline axis."""
    # Extract centerline points
    centerline_points = np.asarray(canonical_flap_pc.points)

    centerline_directions = np.diff(centerline_points, axis=0)
    centerline_directions = np.vstack([centerline_directions, centerline_directions[-1]])  # Extend last direction for boundary
    centerline_directions = np.array([d / np.linalg.norm(d) for d in centerline_directions])  # Normalize all directions

    p_list = [p1,p2]

    theta_distances = []

    # Calculate distances to centerline points and find the nearest segment
    diff = centerline_points - p1
    dist = np.linalg.norm(diff, axis=1)
    nearest_idx = np.argmin(dist)

    # Determine the two nearest points on the centerline to define the segment
    if nearest_idx == 0:
        next_idx = 1
    elif nearest_idx == len(centerline_points) - 1:
        next_idx = nearest_idx - 1
    else:
        next_idx = nearest_idx + 1 if dist[nearest_idx + 1] < dist[nearest_idx - 1] else nearest_idx - 1

    # Define the axis of rotation using the centerline segment direction
    rotation_axis = centerline_points[next_idx] - centerline_points[nearest_idx]
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the axis

    if np.dot(rotation_axis, centerline_directions[nearest_idx]) < 0:
        rotation_axis = -rotation_axis

    # Find the closest point on the axis to the vertex
    segment_start = centerline_points[nearest_idx]
    projection_factor = np.dot(p1 - segment_start, rotation_axis)
    closest_point_on_axis = segment_start + projection_factor * rotation_axis

    # Define the vector from the axis to the vertex
    rejection_vector_1 = p1 - closest_point_on_axis
    rejection_vector_2 = p2 - closest_point_on_axis

    # Calculate the angle between rejection_vector_1 and rejection_vector_2 using dot product
    dot_product = np.dot(rejection_vector_1, rejection_vector_2)
    magnitude_1 = np.linalg.norm(rejection_vector_1)
    magnitude_2 = np.linalg.norm(rejection_vector_2)

    cos_theta = dot_product / (magnitude_1 * magnitude_2)

    theta = np.arccos(cos_theta)

    # Compute the cross product of rejection vectors to determine the direction
    cross_prod = np.cross(rejection_vector_1, rejection_vector_2)
    cross_prod_magnitude = np.linalg.norm(cross_prod)

    # Determine the direction of the angle using the cross product relative to the rotation axis
    if np.dot(cross_prod, rotation_axis) < 0:
        theta = -theta  # Invert the angle if the cross product points in the opposite direction

    # Optionally, convert to degrees if needed

    intra_circum = theta

    return intra_circum


def rotate_mesh_about_centreline(mesh, canonical_flap_pc, theta, relevant_nodes = None):
    """
    Rotate a mesh around a centerline by applying a rotation to mesh vertices
    about the axis defined by the nearest centerline segment.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        Input mesh to be rotated.
    canonical_flap_pc : open3d.geometry.PointCloud
        Canonical centerline point cloud.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    deformed_mesh : open3d.geometry.TriangleMesh
        Rotated mesh.
    """
    # Extract centerline points
    centerline_points = np.asarray(canonical_flap_pc.points)

    centerline_directions = np.diff(centerline_points, axis=0)
    centerline_directions = np.vstack([centerline_directions, centerline_directions[-1]])  # Extend last direction for boundary

    # Prepare mesh vertices for deformation
    vertices = np.asarray(mesh.vertices)
    num_vertices = vertices.shape[0]

    if relevant_nodes is None:
        relevant_indices = range(num_vertices)

    else:
        relevant_indices = relevant_nodes
        num_vertices = relevant_indices.shape[0]

    # Array to hold deformed vertices

    # will modify the vertices from here rather than building all from scratch Jul 25
    deformed_vertices = vertices

    for i in relevant_indices:
        vertex = vertices[i]

        # Calculate distances to centerline points and find the nearest segment
        diff = centerline_points - vertex
        dist = np.linalg.norm(diff, axis=1)
        nearest_idx = np.argmin(dist)

        # Determine the two nearest points on the centerline to define the segment
        if nearest_idx == 0:
            next_idx = 1
        elif nearest_idx == len(centerline_points) - 1:
            next_idx = nearest_idx - 1
        else:
            next_idx = nearest_idx + 1 if dist[nearest_idx + 1] < dist[nearest_idx - 1] else nearest_idx - 1

        # Define the axis of rotation using the centerline segment direction
        rotation_axis = centerline_points[next_idx] - centerline_points[nearest_idx]
        rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the axis

        if np.dot(rotation_axis, centerline_directions[nearest_idx]) < 0:
            rotation_axis = -rotation_axis

        # Find the closest point on the axis to the vertex
        segment_start = centerline_points[nearest_idx]
        projection_factor = np.dot(vertex - segment_start, rotation_axis)
        closest_point_on_axis = segment_start + projection_factor * rotation_axis

        # Define the vector from the axis to the vertex
        rejection_vector = vertex - closest_point_on_axis

        # Apply the rotation using Rodrigues' rotation formula
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotated_vector = (
            rejection_vector * cos_theta +
            np.cross(rotation_axis, rejection_vector) * sin_theta +
            rotation_axis * np.dot(rotation_axis, rejection_vector) * (1 - cos_theta)
        )

        # Compute the new vertex position
        deformed_position = closest_point_on_axis + rotated_vector
        deformed_vertices[i] = deformed_position

    deformed_mesh = copy.deepcopy(mesh)
    deformed_mesh.vertices = o3d.utility.Vector3dVector(deformed_vertices)

    return deformed_mesh


def slide_mesh_along_centreline(mesh, canonical_flap_pc, slide_distance, relevant_nodes = None):
    """
    Slide a mesh along a centerline by interpolating the sliding direction
    based on the projection of each vertex onto the nearest centerline segment.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        Input mesh to be translated.
    canonical_flap_pc : open3d.geometry.PointCloud
        Canonical centerline point cloud.
    slide_distance : float
        Distance to slide the mesh along the centerline.

    Returns
    -------
    deformed_mesh : open3d.geometry.TriangleMesh
        Translated mesh.
    """
    # Extract centerline points
    centerline_points = np.asarray(canonical_flap_pc.points)

    centerline_directions = np.diff(centerline_points, axis=0)
    centerline_directions = np.vstack([centerline_directions, centerline_directions[-1]])  # Extend last direction for boundary
    centerline_directions = np.array([d / np.linalg.norm(d) for d in centerline_directions])  # Normalize all directions

    # Prepare mesh vertices for deformation
    vertices = np.asarray(mesh.vertices)
    num_vertices = vertices.shape[0]

    # Array to hold deformed vertices

    if relevant_nodes is None:
        relevant_indices = range(num_vertices)

    else:
        relevant_indices = relevant_nodes
        num_vertices = relevant_indices.shape[0]

    # Array to hold deformed vertices

    # will modify the vertices from here rather than building all from scratch Jul 25
    deformed_vertices = vertices

    for i in relevant_indices:
        vertex = vertices[i]

        # Calculate distances to centerline points and find the nearest segment
        diff = centerline_points - vertex
        dist = np.linalg.norm(diff, axis=1)
        nearest_idx = np.argmin(dist)

        # Determine the two nearest points on the centerline to define the segment
        if nearest_idx == 0:
            next_idx = 1
        elif nearest_idx == len(centerline_points) - 1:
            next_idx = nearest_idx - 1
        else:
            next_idx = nearest_idx + 1 if dist[nearest_idx + 1] < dist[nearest_idx - 1] else nearest_idx - 1

        # Define the segment start and end
        segment_start = centerline_points[nearest_idx]
        segment_end = centerline_points[next_idx]
        segment_direction = segment_end - segment_start
        segment_length = np.linalg.norm(segment_direction)
        segment_direction /= segment_length  # Normalize the direction

        # Ensure the sliding direction aligns consistently with the precomputed centerline directions
        if np.dot(segment_direction, centerline_directions[nearest_idx]) < 0:
            segment_direction = -segment_direction

        # Project the vertex onto the segment and compute the interpolation factor
        projection = np.dot(vertex - segment_start, segment_direction)
        projection_factor = projection / segment_length

        # Interpolate the sliding direction based on the projection factor
        interpolated_position = (1 - projection_factor) * segment_start + projection_factor * segment_end
        sliding_direction = segment_direction  # Already normalized

        # Slide the vertex along the interpolated sliding direction
        deformed_position = vertex + slide_distance * sliding_direction
        deformed_vertices[i] = deformed_position

    deformed_mesh = copy.deepcopy(mesh)
    deformed_mesh.vertices = o3d.utility.Vector3dVector(deformed_vertices)

    return deformed_mesh


def slide_point_along_centreline(point, canonical_flap_pc, slide_distance):
    """Move a point along the centerline while preserving its local radial offset."""
    # Extract centerline points
    centerline_points = np.asarray(canonical_flap_pc.points)

    centerline_directions = np.diff(centerline_points, axis=0)
    centerline_directions = np.vstack([centerline_directions, centerline_directions[-1]])  # Extend last direction for boundary
    centerline_directions = np.array([d / np.linalg.norm(d) for d in centerline_directions])  # Normalize all directions

    # Calculate distances to centerline points and find the nearest segment
    diff = centerline_points - point
    dist = np.linalg.norm(diff, axis=1)
    nearest_idx = np.argmin(dist)

    # Determine the two nearest points on the centerline to define the segment
    if nearest_idx == 0:
        next_idx = 1
    elif nearest_idx == len(centerline_points) - 1:
        next_idx = nearest_idx - 1
    else:
        next_idx = nearest_idx + 1 if dist[nearest_idx + 1] < dist[nearest_idx - 1] else nearest_idx - 1

    # Define the segment start and end
    segment_start = centerline_points[nearest_idx]
    segment_end = centerline_points[next_idx]
    segment_direction = segment_end - segment_start
    segment_length = np.linalg.norm(segment_direction)
    segment_direction /= segment_length  # Normalize the direction

    # Ensure the sliding direction aligns consistently with the precomputed centerline directions
    if np.dot(segment_direction, centerline_directions[nearest_idx]) < 0:
        segment_direction = -segment_direction

    # Project the vertex onto the segment and compute the interpolation factor
    projection = np.dot(point - segment_start, segment_direction)
    projection_factor = np.clip(projection / segment_length, 0.0, 1.0)

    # Interpolate the sliding direction based on the projection factor
    interpolated_position = (1 - projection_factor) * segment_start + projection_factor * segment_end
    sliding_direction = segment_direction  # Already normalized

    # Slide the vertex along the interpolated sliding direction
    deformed_position = point+ slide_distance * sliding_direction

    return deformed_position


def twist_point_about_centreline(point, canonical_flap_pc, theta):
    """Rotate a point around the local centerline axis."""
    # Extract centerline points
    centerline_points = np.asarray(canonical_flap_pc.points)

    centerline_directions = np.diff(centerline_points, axis=0)
    centerline_directions = np.vstack([centerline_directions, centerline_directions[-1]])  # Extend last direction for boundary

    # Calculate distances to centerline points and find the nearest segment
    diff = centerline_points - point
    dist = np.linalg.norm(diff, axis=1)
    nearest_idx = np.argmin(dist)

    # Determine the two nearest points on the centerline to define the segment
    if nearest_idx == 0:
        next_idx = 1
    elif nearest_idx == len(centerline_points) - 1:
        next_idx = nearest_idx - 1
    else:
        next_idx = nearest_idx + 1 if dist[nearest_idx + 1] < dist[nearest_idx - 1] else nearest_idx - 1

    # Define the axis of rotation using the centerline segment direction
    rotation_axis = centerline_points[next_idx] - centerline_points[nearest_idx]
    rotation_axis /= np.linalg.norm(rotation_axis)  # Normalize the axis

    if np.dot(rotation_axis, centerline_directions[nearest_idx]) < 0:
        rotation_axis = -rotation_axis

    # Find the closest point on the axis to the vertex
    segment_start = centerline_points[nearest_idx]
    projection_factor = np.dot(point - segment_start, rotation_axis)
    closest_point_on_axis = segment_start + projection_factor * rotation_axis

    # Define the vector from the axis to the vertex
    rejection_vector = point - closest_point_on_axis

    # Apply the rotation using Rodrigues' rotation formula
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotated_vector = (
        rejection_vector * cos_theta +
        np.cross(rotation_axis, rejection_vector) * sin_theta +
        rotation_axis * np.dot(rotation_axis, rejection_vector) * (1 - cos_theta)
    )

    # Compute the new vertex position
    deformed_position = closest_point_on_axis + rotated_vector

    return deformed_position


def compute_geodesic_distance_on_point_cloud(point_cloud):
    """Compute cumulative distance along an ordered point cloud."""
    points = np.asarray(point_cloud.points)
    N = len(points)
    GD = np.zeros([N, 1])  # Initialize a 1D array with zeros

    # Compute cumulative Euclidean distances
    for i in range(1, N):
        gdist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
        GD[i] = GD[i-1] + gdist  # Add the distance to the cumulative sum

    return GD


def precompute_projection_weights(mesh, canonical_flap_pc, k=5):
    """Cache mesh-to-centerline segment projections and interpolation weights."""
    P1 = np.asarray(mesh.vertices)
    canonical_points = np.asarray(canonical_flap_pc.points)

    tree = cKDTree(canonical_points)
    _, n1_indices = tree.query(P1, k=1)
    _, neighbor_indices = tree.query(P1, k=k+1)

    projection_data = []

    for i in range(P1.shape[0]):
        v = P1[i]
        n1 = n1_indices[i]
        n_neighbors = neighbor_indices[i][1:]  # exclude self

        best_d = np.inf
        best_n2, best_f2 = None, None
        p1 = canonical_points[n1]

        for n2 in n_neighbors[:4]:
            p2 = canonical_points[n2]
            line = p2 - p1
            line_len2 = np.dot(line, line)
            if line_len2 < 1e-8:
                continue

            t = np.dot(v - p1, line) / line_len2
            d = np.linalg.norm((p1 + t * line) - v)

            if 0 <= t <= 1 and d < best_d:
                best_n2 = n2
                best_f2 = t
                best_d = d

        if best_n2 is not None:
            f2 = best_f2
            n2 = best_n2
        else:
            n2 = n_neighbors[0]
            f2 = 0 if np.linalg.norm(v - p1) < np.linalg.norm(v - canonical_points[n2]) else 1

        f1 = 1 - f2
        projection_data.append((n1, n2, f1, f2))

    return projection_data


def deform_with_cached_projection(mesh, deformation_vectors, projection_data):
    """
    Apply cached 2-neighbor projection deformation.

    Args:
        mesh: Open3D TriangleMesh
        deformation_vectors: array of shape (K, 3)
        projection_data: iterable of length N with entries (n1, n2, f1, f2)

    Returns:
        mesh_deformed: deep-copied mesh with updated vertices
    """
    P1 = np.asarray(mesh.vertices)   # (N, 3)
    proj = np.asarray(projection_data)

    n1 = proj[:, 0].astype(np.intp)
    n2 = proj[:, 1].astype(np.intp)
    f1 = proj[:, 2].astype(P1.dtype)
    f2 = proj[:, 3].astype(P1.dtype)

    disp = (
        f1[:, None] * deformation_vectors[n1] +
        f2[:, None] * deformation_vectors[n2]
    )

    P1_deformed = P1 + disp

    mesh_deformed = copy.deepcopy(mesh)
    mesh_deformed.vertices = o3d.utility.Vector3dVector(P1_deformed)
    return mesh_deformed


def deform_mesh(mesh1, deformation_vector_1, deformation_indices_1, constraint_indices=None, smoothing=0.1, number_iter=10):
    """
    Deforms the given mesh using ARAP energy minimization with optional additional constraints.

    Parameters:
        mesh1 (o3d.geometry.TriangleMesh): Input mesh to be deformed.
        deformation_vector_1 (numpy.ndarray): Array of deformation vectors (n, 3).
        deformation_indices_1 (list[int]): List of vertex indices to apply deformation.
        constraint_indices (list[int] or None): Optional list of vertex indices to keep stationary as constraints.
        smoothing (float): Smoothing alpha value for ARAP optimization.

    Returns:
        tuple: (deformed_mesh, handle_positions_pc)
               - deformed_mesh: The optimized deformed mesh.
               - handle_positions_pc: PointCloud of the deformed vertices.
    """
    # Prepare deformation handle indices and positions
    picked_1 = np.squeeze(np.array(deformation_indices_1))
    handle_ids = list(deformation_indices_1)

    handle_pos = []

    if picked_1.ndim == 0:
        picked_1 = [int(picked_1)]

    for i in range(len(picked_1)):
        picked = picked_1[i]
        deformation_vector = deformation_vector_1[i, :]
        handle_pos.append(np.asarray(mesh1.vertices)[picked, :] + deformation_vector)

    handle_pos = np.vstack(handle_pos)
    handle_pos_pc = o3d.geometry.PointCloud()
    handle_pos_pc.points = o3d.utility.Vector3dVector(handle_pos)

    # Include constraint indices if provided
    if constraint_indices is not None:
        # Add stationary constraints to the full set of constraint IDs and positions
        constraint_vertices = np.asarray(mesh1.vertices)[constraint_indices, :]
        handle_ids.extend(constraint_indices)
        handle_pos = np.vstack([handle_pos, constraint_vertices])

    full_ids = [int(x) for x in handle_ids]
    constraint_ids = o3d.utility.IntVector(full_ids)
    constraint_pos = o3d.utility.Vector3dVector(handle_pos)

    # Perform ARAP optimization
    mesh_prime = mesh1.deform_as_rigid_as_possible(
        constraint_ids,
        constraint_pos,
        max_iter=number_iter,
        smoothed_alpha=smoothing,
        energy=o3d.geometry.DeformAsRigidAsPossibleEnergy.Smoothed,
    )

    return mesh_prime, handle_pos_pc


def full_aorta_arap(source_keypoints, target_keypoints, ct_mesh_roughly_aligned, deformed_nodes, aneurysm_constraint_nodes, smoothness = 5000000.0):
    """Apply an as-rigid-as-possible deformation using constrained centerline motion."""
    deformed_registered_ct = copy.deepcopy(ct_mesh_roughly_aligned)

    intersected_vertex_indices = deformed_nodes #deformed nodes
    total_def= target_keypoints - source_keypoints #deformation applied to deformed nodes

    final_deformed_ct, handle_pc = deform_mesh(
                deformed_registered_ct,
                total_def,
                intersected_vertex_indices,
                aneurysm_constraint_nodes,
                smoothness,
                100)

    return final_deformed_ct


def get_centerline_span_for_mesh(
    mesh: o3d.geometry.TriangleMesh,
    centerline: np.ndarray,
    percentile_clip: float = 1.0,
):
    """
    Compute the earliest and latest centerline indices spanned by a mesh.

    Parameters
    ----------
    mesh : o3d.geometry.TriangleMesh
        Mesh that follows a vessel segment.
    centerline : (M,3) np.ndarray
        Ordered vessel centerline (proximal -> distal).
    percentile_clip : float
        Percentile for outlier rejection (default = 1.0).
        Uses [p, 100-p] percentiles.

    Returns
    -------
    start_idx : int
        Earliest centerline index corresponding to the mesh.
    end_idx : int
        Latest centerline index corresponding to the mesh.
    indices : (N,) np.ndarray
        Centerline index for each mesh vertex (for debugging / plotting).
    distances : (N,) np.ndarray
        Distance from each mesh vertex to the centerline.
    """

    # --- Safety checks ---
    centerline = np.asarray(centerline)
    if centerline.ndim != 2 or centerline.shape[1] != 3:
        raise ValueError("centerline must be of shape (M, 3)")

    vertices = np.asarray(mesh.vertices)
    if vertices.size == 0:
        raise ValueError("mesh has no vertices")

    # --- Build KD-tree on centerline ---
    tree = cKDTree(centerline)

    # --- Project each vertex to nearest centerline point ---
    distances, indices = tree.query(vertices, k=1)

    # --- Robust trimming to remove outliers ---
    if percentile_clip > 0:
        lo = np.percentile(indices, percentile_clip)
        hi = np.percentile(indices, 100 - percentile_clip)
        start_idx = int(np.floor(lo))
        end_idx   = int(np.ceil(hi))
    else:
        start_idx = int(indices.min())
        end_idx   = int(indices.max())

    # --- Clamp indices to valid range ---
    start_idx = max(start_idx, 0)
    end_idx   = min(end_idx, len(centerline) - 1)

    return start_idx, end_idx, indices, distances


def find_calculable_nodes_nricp(vertsTransformed_full_stacked, start_idx, end_idx, ct_slicer_mesh):
    """Identify mesh vertices whose deformation can be inferred from valid centerline transforms."""
    s1_points = vertsTransformed_full_stacked

    ct_slicer_mesh_temp = copy.deepcopy(ct_slicer_mesh)
    vertices = np.asarray(ct_slicer_mesh_temp.vertices)

    # --- STEP 1: Determine which vertices to keep ---
    mesh = copy.deepcopy(ct_slicer_mesh)

    # Original vertex + triangle arrays
    verts = np.asarray(mesh.vertices)
    tris  = np.asarray(mesh.triangles)
    kept_vertices = []
    kept_original_indices = []
    removed_nodes=[]

    for orig_idx, v in enumerate(verts):
        closest_index = np.argmin(np.linalg.norm(s1_points - v, axis=1))
        if closest_index <= end_idx and closest_index >= start_idx:
            kept_vertices.append(v)
            kept_original_indices.append(orig_idx)

        else:
            removed_nodes.append(orig_idx)

    return kept_original_indices, removed_nodes


def deform_mesh_with_projection(mesh, deformation_vectors, canonical_flap_pc, k=5, ivus_skeleton_pc = None):
    """
    Deform a mesh using projection-based interpolation with deformation vectors and
    a canonical flap point cloud.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        Input mesh with vertices to be deformed.
    deformation_vectors : numpy array (Mx3)
        Deformation vectors for each point in the canonical flap point cloud.
    canonical_flap_pc : open3d.geometry.PointCloud
        Canonical flap point cloud providing reference positions for deformation.
    k : int
        Number of nearest neighbors to use for local projection (default 5).

    Returns
    -------
    mesh_deformed : open3d.geometry.TriangleMesh
        Deformed mesh with updated vertex positions.
    """

    # Vertex arrays
    P1 = np.asarray(mesh.vertices)
    canonical_points = np.asarray(canonical_flap_pc.points)
    P1_deformed = P1.copy()

    # Build KDTree for efficient neighbor search
    tree = cKDTree(canonical_points)

    # For each vertex, find the closest point and its neighbors
    _, n1_indices = tree.query(P1, k=1)
    _, neighbor_indices = tree.query(P1, k=k+1)  # +1 because n1 is also included

    indices_of_deformable_nodes = []
    original = []
    deformed= []

    calculable_nodes = []
    if(ivus_skeleton_pc is not None):
        start_idx, end_idx, placeholder, placeholder_2 = get_centerline_span_for_mesh(
        mesh,
        np.asarray(ivus_skeleton_pc.points),
        percentile_clip=1.0)
        calculable_nodes, placeholder = find_calculable_nodes_nricp(np.asarray(canonical_flap_pc.points), start_idx, end_idx, copy.deepcopy(mesh))

    for i in range(P1.shape[0]):
        v = P1[i]
        n1 = n1_indices[i]
        n_neighbors = neighbor_indices[i][1:]  # exclude self (n1)

        best_projection = None
        best_d = np.inf
        best_n2 = None
        best_f2 = None

        p1 = canonical_points[n1]

        for n2 in n_neighbors[:4]:  # limit for efficiency
            p2 = canonical_points[n2]
            line = p2 - p1
            line_len2 = np.dot(line, line)
            if line_len2 < 1e-8:
                continue  # degenerate segment

            t = np.dot(v - p1, line) / line_len2
            projection = p1 + t * line
            d = np.linalg.norm(projection - v)

            if t < 0:
                f2 = -1
            elif t > 1:
                f2 = -2
            else:
                f2 = t

            if 0 <= f2 <= 1 and d < best_d:
                best_projection = projection
                best_d = d
                best_n2 = n2
                best_f2 = f2

        if best_projection is not None:
            n2 = best_n2
            f2 = best_f2
        else:
            n2 = n_neighbors[0]
            f2 = 0 if np.linalg.norm(v - canonical_points[n1]) < np.linalg.norm(v - canonical_points[n2]) else 1

        f1 = 1 - f2
        disp1 = deformation_vectors[n1]
        disp2 = deformation_vectors[n2]

        if i in calculable_nodes and ivus_skeleton_pc is not None:
            original.append(v)
            new_pos = f1 * disp1 + f2 * disp2
            deformed.append(v+ new_pos)
            indices_of_deformable_nodes.append(i)

        elif ivus_skeleton_pc is None:
            P1_deformed[i] +=f1 * disp1 + f2 * disp2

    if ivus_skeleton_pc is None:
        mesh_deformed = copy.deepcopy(mesh)
        mesh_deformed.vertices = o3d.utility.Vector3dVector(P1_deformed)

    else:
        original = np.vstack(original)
        deformed = np.vstack(deformed)
        constraint_indices = None

        deformed_pc = o3d.geometry.PointCloud()
        deformed_pc.points = o3d.utility.Vector3dVector(deformed)
        deformed_pc.paint_uniform_color([1,0,0])
        orig_pc = o3d.geometry.PointCloud()
        orig_pc.points = o3d.utility.Vector3dVector(original)
        o3d.visualization.draw_geometries([orig_pc, deformed_pc])

        mesh_temp = full_aorta_arap( original,  deformed, copy.deepcopy(mesh), indices_of_deformable_nodes, constraint_indices, smoothness = 100000.0)

        mesh_temp.compute_vertex_normals()

        mesh_deformed = copy.deepcopy(mesh)
        mesh_deformed.vertices = copy.deepcopy(mesh_temp.vertices)

    return mesh_deformed


def deform_side_branches_with_projection(pc, deformation_vectors, canonical_flap_pc):
    """
    Deform a mesh using projection-based interpolation with deformation vectors and
    a canonical flap point cloud.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        Input mesh with vertices to be deformed.
    deformation_vectors : numpy array (Mx3)
        Deformation vectors for each point in the canonical flap point cloud.
    canonical_flap_pc : open3d.geometry.PointCloud
        Canonical flap point cloud providing reference positions for deformation.

    Returns
    -------
    mesh_deformed : open3d.geometry.TriangleMesh
        Deformed mesh with updated vertex positions.
    """
    import numpy as np
    import copy

    P1 = np.asarray(pc.points)
    num_points = P1.shape[0]

    canonical_points = np.asarray(canonical_flap_pc.points)
    num_canonical_points = canonical_points.shape[0]

    P1_deformed = P1.copy()

    # Iterate over all vertices in the mesh
    for i in range(num_points):
        # Compute distances to all canonical points
        diff = canonical_points - P1[i, :]
        dist = np.linalg.norm(diff, axis=1)
        n1 = np.argmin(dist)  # Nearest canonical point index

        # Find neighbors of the nearest canonical point
        diff_neighbors = canonical_points - canonical_points[n1, :]
        dist_neighbors = np.linalg.norm(diff_neighbors, axis=1)
        n2_candidates = np.argsort(dist_neighbors)[1:]  # Nearest neighbors after n1

        best_projection = None
        best_d = np.inf
        best_n2 = None
        best_f2 = None

        # Perform projection-based interpolation
        for n2 in n2_candidates[:4]:  # Consider up to 4 nearest neighbors for efficiency
            line_direction = canonical_points[n2, :] - canonical_points[n1, :]
            projection = canonical_points[n1, :] + (
                ((P1[i, :] - canonical_points[n1, :]) @ line_direction)
                / (line_direction @ line_direction)
            ) * line_direction
            d = np.linalg.norm(projection - P1[i, :])

            f2 = np.linalg.norm(projection - canonical_points[n1, :]) / np.linalg.norm(
                line_direction
            )

            if np.dot(projection - canonical_points[n1, :], line_direction) < 0:
                f2 = -1  # Outside on n1 side
            elif f2 > 1:
                f2 = -2  # Outside on n2 side

            if 0 <= f2 <= 1 and d < best_d:
                best_projection = projection
                best_d = d
                best_n2 = n2
                best_f2 = f2

        # Use best candidate or fall back to nearest point
        if best_projection is not None:
            n2 = best_n2
            f2 = best_f2
        else:
            n2 = n2_candidates[0]
            f2 = 0 if dist[n1] < dist[n2] else 1

        # Interpolation weights
        f1 = 1 - f2

        # Apply deformation vectors
        displacement1 = deformation_vectors[n1, :]
        displacement2 = deformation_vectors[n2, :]
        P1_deformed[i, :] += f1 * displacement1 + f2 * displacement2

    mesh_deformed = copy.deepcopy(pc)
    mesh_deformed.points = o3d.utility.Vector3dVector(P1_deformed)

    return mesh_deformed
