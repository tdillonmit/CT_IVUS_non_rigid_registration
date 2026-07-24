"""Correspondence estimation and branch-detection helpers for CT–IVUS registration."""

import copy
import itertools
import time
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import viterbi_correspondence_matching as skm
from visualization_functions import *


def find_nearest_neighbour_correspondences(ct_centroids, ivus_centroids,
                                           closest_indices_ct, closest_indices_ivus):
    """Match CT branch centroids to unique nearest IVUS centroids and return centerline-index pairs."""
    centroid_corres = []
    for i, ct_centroid in enumerate(ct_centroids):
        # nearest IVUS index for this CT centroid
        distances = np.linalg.norm(ivus_centroids - ct_centroid, axis=1)
        closest_index_ivus = np.argmin(distances)
        centroid_corres.append([i, closest_index_ivus, distances[closest_index_ivus]])

    centroid_corres = np.asarray(centroid_corres)  # shape (N, 3): [ct_idx, ivus_idx, dist]

    # sort by distance so closest CT–IVUS pairs come first
    centroid_corres = centroid_corres[np.argsort(centroid_corres[:, 2])]

    # enforce one-to-one: keep only first (closest) CT for each IVUS index
    _, unique_indices = np.unique(centroid_corres[:, 1], return_index=True)
    centroid_corres = centroid_corres[unique_indices]

    # now build semantic correspondences (map back to original indices)
    semantic_corres_nn = []
    for ct_idx, ivus_idx, _ in centroid_corres:
        semantic_corres_nn.append([
            closest_indices_ct[int(ct_idx)],
            closest_indices_ivus[int(ivus_idx)]
        ])

    return np.asarray(semantic_corres_nn)


def show_branched_correspondences(corres,semantic_corres,closest_indices_ct,closest_indices_ivus,ct_skeleton_pc,ivus_skeleton_pc,ct_centroids,ivus_centroids, visualize_branches):
    """Build branch-aware correspondence geometry and optionally visualize offset skeletons."""
    # convert from centreline correspondences to orifice (centroid) correspondences
    corres_original = np.empty((0,2))
    closest_indices_ct = np.asarray(closest_indices_ct)
    closest_indices_ivus = np.asarray(closest_indices_ivus)

    print("semantic_corres", semantic_corres)
    print("closest_indices_ct", closest_indices_ct)
    print("closest indices ivus", closest_indices_ivus)
    for row in semantic_corres:
        matches_a = np.argwhere(closest_indices_ct == row[0]).flatten()
        matches_b = np.argwhere(closest_indices_ivus == row[1]).flatten()

        if matches_a.size > 0 and matches_b.size > 0:
            corres_a = matches_a[0]
            corres_b = matches_b[0]
            corres_original = np.vstack((corres_original, [corres_a, corres_b]))
        else:
            print(f"Skipping row {row} — match not found in CT or IVUS indices")

    corres_original = corres_original.astype(int)

    ct_skeleton_pc_with_branches, ct_lineset_branches = get_branched_skeleton(ct_skeleton_pc,closest_indices_ct,ct_centroids)

    ivus_skeleton_pc_with_branches, ivus_lineset_branches = get_branched_skeleton(ivus_skeleton_pc,closest_indices_ivus,ivus_centroids)

    ct_skeleton_branches = o3d.geometry.PointCloud()
    ct_skeleton_branches.points = o3d.utility.Vector3dVector(ct_centroids)
    ivus_skeleton_branches = o3d.geometry.PointCloud()
    ivus_skeleton_branches.points = o3d.utility.Vector3dVector(ivus_centroids)
    corres_lines_branches_to_save = plot_open3d_correspondences(ct_skeleton_branches, ivus_skeleton_branches, corres_original, 0, [1,0,1])
    corres_lines_to_save = plot_open3d_correspondences(ct_skeleton_pc, ivus_skeleton_pc, corres, 0)

    ivus_lineset_branches_to_save = copy.deepcopy(ivus_lineset_branches)
    ct_lineset_branches_to_save = copy.deepcopy(ct_lineset_branches)

    translation = [0.03,0,0.03]
    ivus_skeleton_pc_with_branches.translate(translation)
    ivus_lineset_branches.translate(translation)

    ivus_skeleton_pc_with_branches.paint_uniform_color([0,0,1])
    ct_skeleton_pc_with_branches.paint_uniform_color([1,0,0])
    ivus_lineset_branches.paint_uniform_color([0,0,1])
    ct_lineset_branches.paint_uniform_color([1,0,0])

    ivus_skeleton_transformed = copy.deepcopy(ivus_skeleton_pc)
    ivus_skeleton_transformed.translate(translation)

    ivus_skeleton_branches.translate(translation)
    corres_lines_branches = plot_open3d_correspondences(ct_skeleton_branches, ivus_skeleton_branches, corres_original, 0, [1,0,1])
    corres_lines = plot_open3d_correspondences(ct_skeleton_pc, ivus_skeleton_transformed, corres, 0)

    if(visualize_branches == 1):
        o3d.visualization.draw_geometries([ct_skeleton_pc_with_branches,ct_lineset_branches, ivus_skeleton_pc_with_branches, ivus_lineset_branches] + corres_lines + corres_lines_branches)

    return corres_original, ct_lineset_branches_to_save, ivus_lineset_branches_to_save, corres_lines_to_save, corres_lines_branches_to_save








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


def cluster_pc_based_on_centroids(orig_branch_pc, ivus_centroids, distance_threshold=np.inf, colormap_name='tab20'):
    """
    Clusters a point cloud based on proximity to IVUS centroids and assigns a unique color
    to each cluster using a matplotlib colormap. Points farther than the distance threshold
    from any centroid are not included in any cluster.

    Parameters:
    - orig_branch_pc: open3d.geometry.PointCloud
    - ivus_centroids: (n, 3) numpy array
    - colormap_name: name of a matplotlib colormap (e.g., 'tab20', 'viridis')
    - distance_threshold: maximum distance for a point to be assigned to a cluster

    Returns:
    - clustered_pcs: list of open3d.geometry.PointCloud objects, colored by cluster
    """
    orig_branch_points = np.asarray(orig_branch_pc.points)
    n_centroids = ivus_centroids.shape[0]

    grouped_points = [[] for _ in range(n_centroids)]

    for point in orig_branch_points:
        distances = np.linalg.norm(ivus_centroids - point, axis=1)
        min_dist = np.min(distances)
        closest_index = np.argmin(distances)

        if min_dist <= distance_threshold:
            grouped_points[closest_index].append(point)

    cmap = plt.get_cmap(colormap_name)
    colors = [cmap(i / max(n_centroids - 1, 1))[:3] for i in range(n_centroids)]

    clustered_pcs = []
    for i, group in enumerate(grouped_points):
        pc = o3d.geometry.PointCloud()
        if group:
            pc.points = o3d.utility.Vector3dVector(np.array(group))
            pc.paint_uniform_color(colors[i])
        clustered_pcs.append(pc)

    return clustered_pcs


def filter_spurious_points_3(ivus_skeleton_pc, ct_skeleton_pc, ivus_centroids,ct_centroids, orifice_pc, visualize_debug, closest_indices_ct, orig_branch_pc=None, ivus_funsr_mesh=None, ct_scan_pc = None):
    """Filter IVUS branch detections using CT proximity, branch orientation, and spatial clustering."""
    # find distance from each ivus centroid to nearest transformed ct_centroid
    ct_centroid_pc = o3d.geometry.PointCloud()
    ct_centroid_pc.points = o3d.utility.Vector3dVector(ct_centroids)
    transformed_ct_centroids = np.asarray(ct_centroid_pc.points)

    ivus_temp_spheres = get_sphere_cloud(ivus_centroids, 0.004, 10, [0,0,1])
    ct_temp_spheres = get_sphere_cloud(transformed_ct_centroids, 0.004, 10, [1,0,0])

    centerline_points = np.asarray(ivus_skeleton_pc.points)
    centerline_normals = np.diff(centerline_points, axis=0)
    centerline_normals = np.vstack([centerline_normals, centerline_normals[-1]])  # Extend last normal for boundary

    ivus_centroids_filtered = []
    ivus_centroids_filtered_indices = []

    j = 0

    # ------ STEP 1 - INITIAL FILTER ------- #
    # "for each IVUS centroid, if any CT scan points are close enough in both space angle, keep that IVUS centroid"
    for centroid in ivus_centroids:

        # find euclidean distance
        diff = transformed_ct_centroids - centroid
        dist = np.linalg.norm(diff, axis=1)
        euclidean_distance_threshold = 0.05
        relevant_ct_centroids_indices = np.argwhere(dist < euclidean_distance_threshold)

        # find angular distance
        relevant_ct_centroids = transformed_ct_centroids[relevant_ct_centroids_indices,:]

        # Calculate distances to centerline points and find the nearest segment
        AD_degrees = []
        s = np.asarray([1,0,0])  #later should replace this with the vector spanning the minimum dimension of the part

        # of those CT scan points that are close enough in space, check angles
        for branch in relevant_ct_centroids:

            vertex = branch
            diff = vertex - centerline_points
            dist = np.linalg.norm(diff, axis=1)
            nearest_idx = np.argmin(dist)

            # Determine the two nearest points on the centerline to define the segment
            if nearest_idx == 0:
                next_idx = 1
            elif nearest_idx == len(centerline_points) - 1:
                next_idx = nearest_idx - 1
            else:
                next_idx = nearest_idx + 1 if dist[nearest_idx + 1] < dist[nearest_idx - 1] else nearest_idx - 1

            # Define the direction of the segment and calculate the projection of the vertex onto this segment
            segment_direction = centerline_points[next_idx] - centerline_points[nearest_idx]
            projection_factor = np.dot(vertex - centerline_points[nearest_idx], segment_direction) / np.dot(segment_direction, segment_direction)

            # average n
            interpolated_normal = ((1 - projection_factor) * centerline_normals[nearest_idx]) + (projection_factor * centerline_normals[next_idx])
            interpolated_normal /= np.linalg.norm(interpolated_normal)  # Normalize the interpolated normal

            interpolated_point = ((1 - projection_factor) * centerline_points[nearest_idx]) + (projection_factor * centerline_points[next_idx])
            closest_point = interpolated_point

            reference_vector = centroid - closest_point
            reference_vector /= np.linalg.norm(reference_vector)
            reference_vector = np.array(reference_vector).flatten()
            b_1 = reference_vector - ((np.dot(reference_vector,interpolated_normal)/np.dot(interpolated_normal,interpolated_normal))*interpolated_normal)
            b_1 /= np.linalg.norm(b_1)
            b_1 = np.array(b_1).flatten()

            rejection_vector = vertex - closest_point
            rejection_vector /= np.linalg.norm(rejection_vector)

            rejection_vector = np.array(rejection_vector).flatten()

            # this does not return the signed angle
            angle = np.arccos(np.dot(b_1,rejection_vector)/np.dot(rejection_vector,rejection_vector))

            angle = np.degrees(abs(angle))
            AD_degrees.append(angle)

        AD_degrees = np.asarray(AD_degrees)
        angular_distance_threshold = 360
        if(np.any( AD_degrees < angular_distance_threshold)):
            ivus_centroids_filtered.append(centroid)
            ivus_centroids_filtered_indices.append(j)

        j=j+1

    ivus_centroids_filtered_indices = np.asarray(ivus_centroids_filtered_indices)

    ivus_temp_spheres = get_sphere_cloud(ivus_centroids_filtered, 0.004, 10, [0,0,1])
    ct_temp_spheres = get_sphere_cloud(transformed_ct_centroids, 0.004, 10, [1,0,0])

    ivus_centroids_filtered = np.asarray(ivus_centroids_filtered)

    centroids, s1_pcd, closest_indices = cluster_abdominal_ivus_centroids(ivus_skeleton_pc, ivus_centroids, orifice_pc, visualize_debug,  orig_branch_pc, ivus_centroids_filtered, ivus_centroids_filtered_indices, transformed_ct_centroids, ct_skeleton_pc, closest_indices_ct, ct_centroids)

    return centroids, s1_pcd, closest_indices


def cluster_abdominal_ivus_centroids(ivus_skeleton_pc, ivus_centroids, orifice_pc, visualize_debug, orig_branch_pc=None,  ivus_centroids_filtered=None, ivus_centroids_filtered_indices=None, transformed_ct_centroids = None, ct_skeleton_pc = None, closest_indices_ct=None, ct_centroids=None):
    """Cluster abdominal IVUS branch detections using branch-pass labels and spatial consistency."""
    branch_passes = []
    labels_temp = np.asarray(orifice_pc.colors)
    branch_pass_temp = (labels_temp[:, 0] * 255).astype(int)
    labels_check = np.asarray(orifice_pc.colors)
    weights = (labels_check[:, 1] * 2000).astype(int)

    # eps = 0.0015
    eps = 0.005
    # eps = 0.0015
    min_points = 5

    ivus_centroids_pc = o3d.geometry.PointCloud()
    ivus_centroids_pc.points = o3d.utility.Vector3dVector(ivus_centroids)
    dbscan_labels = np.array(ivus_centroids_pc.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    unique_dbscan_labels = np.unique(dbscan_labels)

    branch_passes = np.zeros_like(branch_pass_temp)

    for dbscan_label in unique_dbscan_labels:

        # find indices in that label
        indices_in_dbscan = np.argwhere(dbscan_labels==dbscan_label).flatten()

        # find the branch pass values of those clusters
        branch_pass_values = branch_pass_temp[indices_in_dbscan]

        # what are the unique branch pass values of that cluster label
        unique_branch_pass_values = np.unique(branch_pass_values)

        # where are values in branch_pass_temp that are equal to any of the unique beanch pass values
        indices_to_change = np.isin(branch_pass_temp,unique_branch_pass_values)

        branch_passes[indices_to_change] = dbscan_label

    if(visualize_debug==1):
        print("db scan visualize")
        colors = plt.get_cmap("tab20")(branch_passes % 20)[:, :3]  # RGB colors, cycle through 20 colors
        colors[branch_passes == -1] = [0, 0, 0]  # Color noise (label -1) as black
        ivus_centroids_pc.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([ivus_centroids_pc], window_name="IVUS branch pass Clusters")

    # --- VISUALIZATION (OPTIONAL) ----- #
    unique_branch_passes, inverse_indices = np.unique(branch_passes, return_inverse = True )

    all_spheres=[]

    colormap = plt.cm.get_cmap('viridis', len(unique_branch_passes))
    colors = colormap(np.linspace(0, 1, len(unique_branch_passes)))
    colors = (colormap(np.linspace(0, 1, len(unique_branch_passes)))[:, :3] * 255).astype(int)
    colors = colors / 255.0  # Convert back to Open3D format

    cluster_centroids = []
    all_cluster_spheres = []

    if(ivus_centroids_filtered is not None):
        ct_temp_spheres = get_sphere_cloud(transformed_ct_centroids, 0.004, 10, [1,0,0])
    else:
        ct_temp_spheres = None

    for unique_branch_pass, color in zip(unique_branch_passes, colors):

        branch_color = color
        relevant_centroids_args = np.argwhere( branch_passes == unique_branch_pass ).flatten()
        relevant_centroids = ivus_centroids[relevant_centroids_args,:]
        filtered_weight_arrays = weights[relevant_centroids_args]

        all_points=np.empty((0,3))

        for one_centroid, weight in zip(relevant_centroids, filtered_weight_arrays):

            duplicated_point = np.tile(one_centroid, (weight, 1))
            all_points = np.vstack((all_points,duplicated_point))

        if all_points.shape[0] == 0:  # Check if the array is not empty along axis 0 (rows)
            print("warning invalid weights (0) encountered during clustering")
            continue

        centroid = np.mean(all_points, axis=0, keepdims=True)
        cluster_centroids.append(centroid)

        branch_spheres = get_sphere_cloud( relevant_centroids , 0.004, 10, branch_color)
        cluster_sphere = get_sphere_cloud( centroid , 0.004, 10, branch_color)
        all_spheres.append(branch_spheres)
        all_cluster_spheres.append(cluster_sphere)

    if(visualize_debug == 1):

            print("visualizing clusters")
            if(ivus_centroids_filtered is not None):
                o3d.visualization.draw_geometries(all_cluster_spheres + [ct_temp_spheres, ct_skeleton_pc, ivus_skeleton_pc])
            else:
                o3d.visualization.draw_geometries(all_cluster_spheres + [ivus_skeleton_pc, orig_branch_pc, orifice_pc])
            print("done visualizing clusters")

    # ------- STEP 3 - FIND FINAL CLUSTERS ------- #
    # combined clustering based on branch pass with filtering based on euclidean and angular distance

    orifice_center_points = np.asarray(orifice_pc.points)

    if(ivus_centroids_filtered_indices is not None):
        orifice_center_points = orifice_center_points[ivus_centroids_filtered_indices,:]
        branch_passes = branch_passes[ivus_centroids_filtered_indices]
        weights = weights[ivus_centroids_filtered_indices]

    unique_entries, inverse_indices = np.unique(branch_passes, return_inverse=True)

    # Split points by clusters into multiple arrays
    split_centerpoint_arrays = [orifice_center_points[inverse_indices == i] for i in range(len(unique_entries))]
    split_weight_arrays = [weights[inverse_indices == i] for i in range(len(unique_entries))]


    min_weighted_points_threshold = 400
    # min_weighted_points_threshold = 1000
    filtered_centerpoint_arrays, filtered_weight_arrays = zip(*[
        (array, weight) for array, weight in zip(split_centerpoint_arrays, split_weight_arrays)
        if np.sum(weight) >= min_weighted_points_threshold
    ])

    # Filter further based on number of points (i.e. images)
    min_points_threshold = 3
    filtered_centerpoint_arrays, filtered_weight_arrays = zip(*[
        (array, weight) for array, weight in zip(filtered_centerpoint_arrays, filtered_weight_arrays)
        if len(array) >= min_points_threshold
    ])

    # Find final clusters
    centroids = []
    for center_array, weight_array in zip(filtered_centerpoint_arrays, filtered_weight_arrays):
        all_points = np.repeat(center_array, weight_array, axis=0)  # Vectorized weight duplication
        centroids.append(np.mean(all_points, axis=0))
    centroids = np.asarray(centroids)

    # ----- STEP 5 - VISUALIZE AND RECREATE IVUS SKELETON PC FOLLOWING FILTERING ---- #

    if(visualize_debug ==1):

        pointclouds = []
        colors = (colormap(np.linspace(0, 1, len(filtered_centerpoint_arrays)))[:, :3] * 255).astype(int)
        colors = colors / 255.0  # Convert back to Open3D format
        for array, color in zip(filtered_centerpoint_arrays, colors):

            test_pc = o3d.geometry.PointCloud()
            test_pc.points = o3d.utility.Vector3dVector(np.asarray(array))
            test_pc.paint_uniform_color(color)
            pointclouds.append(test_pc)

        spheres = get_sphere_cloud(centroids, 0.004, 10)
        spheres.paint_uniform_color([0,0,1])

        if(ivus_centroids_filtered is not None):
            red_pc = copy.deepcopy(ct_skeleton_pc)
            red_pc.paint_uniform_color([1,0,0])
        blue_pc = copy.deepcopy(ivus_skeleton_pc)
        blue_pc.paint_uniform_color([0,0,1])

        spheres_lineset = create_wireframe_lineset_from_mesh(spheres)
        o3d.visualization.draw_geometries([ivus_skeleton_pc,orig_branch_pc, orifice_pc, spheres_lineset]+pointclouds)

        if(ivus_centroids_filtered is not None):
            o3d.visualization.draw_geometries([spheres, ct_temp_spheres, red_pc, blue_pc])

    s1_points = np.asarray(ivus_skeleton_pc.points)
    closest_points = np.empty((0,3))

    closest_indices = []

    # Step 1: Track all inserted points
    inserted_points = []

    for orifice in centroids:
        interpolated_point, insert_index = get_closest_projected_point(orifice, s1_points)
        inserted_points.append(interpolated_point)

        # Insert the new point
        s1_points = np.vstack((s1_points[:insert_index], interpolated_point, s1_points[insert_index:]))

    # Step 2: Convert to array
    inserted_points = np.array(inserted_points)

    # Step 3: After the loop, find indices of inserted points in final s1_points
    def find_matching_indices(inserted_points, full_points):
        indices = []
        for pt in inserted_points:
            dists = np.linalg.norm(full_points - pt, axis=1)
            idx = np.argmin(dists)
            indices.append(idx)
        return np.array(indices)

    inserted_indices = find_matching_indices(inserted_points, s1_points)

    # Step 4: Sort the indices and reorder centroids accordingly
    sorted_order = np.argsort(inserted_indices)
    closest_indices = inserted_indices[sorted_order]
    centroids = centroids[sorted_order]

    s1_pcd = o3d.geometry.PointCloud()
    s1_pcd.points = o3d.utility.Vector3dVector(s1_points)

    if(ivus_centroids_filtered is not None):
        ct_skeleton_spheres, ct_tubeset, ct_spheres = convert_centerline_pc_to_branched_sphere_tubeset( ct_skeleton_pc,closest_indices_ct, ct_centroids, [1,0,0])

    ivus_skeleton_spheres, ivus_tubeset, ivus_spheres = convert_centerline_pc_to_branched_sphere_tubeset( s1_pcd,closest_indices, centroids, [0,0,1])

    ivus_skeleton_pc_with_branches, ivus_lineset_branches = get_branched_skeleton(s1_pcd,closest_indices,centroids)
    ivus_skeleton_pc_with_branches.paint_uniform_color([0,0,1])
    ivus_lineset_branches.paint_uniform_color([0,0,1])
    if(visualize_debug==1):
        o3d.visualization.draw_geometries([ivus_skeleton_pc,orig_branch_pc, orifice_pc, spheres_lineset, ivus_skeleton_pc_with_branches, ivus_lineset_branches]+pointclouds)

    colors = np.zeros_like(s1_points)
    colors[ closest_indices, : ] = [0,0,1]
    s1_pcd.colors = o3d.utility.Vector3dVector(colors)

    spheres =  get_sphere_cloud(centroids, 0.004, 10, [0,0,1])

    spheres =  get_sphere_cloud(centroids, 0.004, 10, [0,0,1])

    return centroids, s1_pcd, closest_indices


def generate_full_injective_matchings(
    closest_indices_ct: np.ndarray,
    closest_indices_ivus: np.ndarray
    ) -> list[np.ndarray]:
    """
    Generate all strictly increasing one-to-one matchings from CT to IVUS,
    assuming all CT points must be matched and IVUS may have extras.

    Returns a list of (M, 2) correspondence arrays.
    """
    M = len(closest_indices_ct)
    N = len(closest_indices_ivus)

    correspondences = []

    print("closest_indices_ivus check", closest_indices_ivus)

    if(M<N):
        print("IVUS has more")
        for ivus_subset in itertools.combinations(closest_indices_ivus, M):

            pair_array = np.stack((closest_indices_ct, ivus_subset), axis=1)

            correspondences.append(pair_array)

    else:

        for k in range(2, N + 1):
            print("CT has more")
            for ct_subset in itertools.combinations(closest_indices_ct, k):
                for ivus_subset in itertools.combinations(closest_indices_ivus, k):
                    pair_array = np.stack((ct_subset, ivus_subset), axis=1)

                    correspondences.append(pair_array)

                    # use this if its not generating any abd candidates due to renal order swap

    if M == 1:
        c = closest_indices_ct[0]
        for v in closest_indices_ivus:
            correspondences.append(np.array([[c, v]]))

    if N == 1:
        v = closest_indices_ivus[0]
        for c in closest_indices_ct:
            correspondences.append(np.array([[c, v]]))

    return correspondences


def get_ransac_correspondence_sets(ct_skeleton_pc, ivus_skeleton_pc, closest_indices_ct, closest_indices_ivus, ct_centroids, ivus_centroids, ct_side_branch_pc, abdominal_reg, visualize_debug, chopped=0):
    """Generate anatomically plausible head and abdominal correspondence candidate sets."""
    # split up into head and abd corres
    # split into head branches and visceral branches
    s1_points = np.asarray(ct_skeleton_pc.points)
    half_index_s1 = np.shape(s1_points)[0] // 2

    s2_points = np.asarray(ivus_skeleton_pc.points)
    half_index_s2 = np.shape(s2_points)[0] // 2

    closest_indices_ct = np.asarray(closest_indices_ct)
    closest_indices_ivus = np.asarray(closest_indices_ivus)

    if(abdominal_reg == 0):
        closest_indices_ct_head = closest_indices_ct[closest_indices_ct < half_index_s1]
        ct_centroids_head = ct_centroids[closest_indices_ct < half_index_s1,:]
        closest_indices_ivus_head = closest_indices_ivus[closest_indices_ivus < half_index_s2]
        ivus_centroids_head = ivus_centroids[closest_indices_ivus < half_index_s2, :]
        closest_indices_ct_abd = closest_indices_ct[closest_indices_ct >= half_index_s1]
        ct_centroids_abd = ct_centroids[closest_indices_ct >= half_index_s1,:]
        closest_indices_ivus_abd = closest_indices_ivus[closest_indices_ivus >= half_index_s2]
        ivus_centroids_abd = ivus_centroids[closest_indices_ivus >= half_index_s2,:]

    # all points on ivus centerline are abdominal for abdominal reg
    if(abdominal_reg ==1):

        if(chopped!=1):
            closest_indices_ct_head = closest_indices_ct[closest_indices_ct < half_index_s1]
            ct_centroids_head = ct_centroids[closest_indices_ct < half_index_s1,:]
            closest_indices_ivus_head = []
            ivus_centroids_head = []

            closest_indices_ct_abd = closest_indices_ct[closest_indices_ct >= half_index_s1]
            ct_centroids_abd = ct_centroids[closest_indices_ct >= half_index_s1,:]
            closest_indices_ivus_abd = closest_indices_ivus
            ivus_centroids_abd = ivus_centroids

        else:
            closest_indices_ct_head = []
            ct_centroids_head = []
            closest_indices_ivus_head = []
            ivus_centroids_head = []

            closest_indices_ct_abd = closest_indices_ct
            ct_centroids_abd = ct_centroids
            closest_indices_ivus_abd = closest_indices_ivus
            ivus_centroids_abd = ivus_centroids

    # do this for head corres first

    head_semantic_corres_sets = generate_full_injective_matchings(closest_indices_ct_head,closest_indices_ivus_head)

    abd_semantic_corres_sets = generate_full_injective_matchings(closest_indices_ct_abd,closest_indices_ivus_abd)

    # prune out any correspondences with wildly different angles between CT and IVUS

    head_semantic_corres_sets_filtered = []

    angle_threshold = 100

    for head_semantic_corres_set in head_semantic_corres_sets:

        angles = []

        for corres_pair in head_semantic_corres_set:

            # find relevant ct_centroid and ivus centroid first
            ct_centroid_index = np.argwhere(closest_indices_ct == corres_pair[0]).squeeze()
            ivus_centroid_index = np.argwhere(closest_indices_ivus == corres_pair[1]).squeeze()

            print("ct_centroid_index", ct_centroid_index)
            print("ivus_centroid_index", ivus_centroid_index)

            ct_fen_vector = ct_centroids[ct_centroid_index,:] - s1_points[corres_pair[0],:]
            ct_fen_vector = ct_fen_vector / np.linalg.norm(ct_fen_vector)
            ivus_fen_vector = ivus_centroids[ivus_centroid_index,:] - s2_points[corres_pair[1],:]
            ivus_fen_vector = ivus_fen_vector / np.linalg.norm(ivus_fen_vector)

            dot_product = np.dot(ct_fen_vector, ivus_fen_vector)
            norm_a = np.linalg.norm(ct_fen_vector)
            norm_b = np.linalg.norm(ivus_fen_vector)

            # Angle in radians
            angle_rad = np.arccos(dot_product / (norm_a * norm_b))
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)

        angles = np.asarray(np.abs(angles))

        if(np.all(angles < angle_threshold)):
            print("accepted set head", angles)
            head_semantic_corres_sets_filtered.append(head_semantic_corres_set)

        else:
            print("rejected set head", head_semantic_corres_set)

    abd_semantic_corres_sets_filtered = []

    for abd_semantic_corres_set in abd_semantic_corres_sets:

        angles = []

        # pulling out wrong correspondence pairs

        ct_fen_vector_lines = []
        ivus_fen_vector_lines = []

        for corres_pair in abd_semantic_corres_set:

            # find relevant ct_centroid and ivus centroid first
            ct_centroid_index = np.argwhere(closest_indices_ct == corres_pair[0]).squeeze()
            ivus_centroid_index = np.argwhere(closest_indices_ivus == corres_pair[1]).squeeze()

            ct_fen_vector = ct_centroids[ct_centroid_index,:] - s1_points[corres_pair[0],:]
            ct_fen_vector = ct_fen_vector / np.linalg.norm(ct_fen_vector)
            ivus_fen_vector = ivus_centroids[ivus_centroid_index,:] - s2_points[corres_pair[1],:]
            ivus_fen_vector = ivus_fen_vector / np.linalg.norm(ivus_fen_vector)

            dot_product = np.dot(ct_fen_vector, ivus_fen_vector)
            norm_a = np.linalg.norm(ct_fen_vector)
            norm_b = np.linalg.norm(ivus_fen_vector)

            # Angle in radians
            angle_rad = np.arccos(dot_product / (norm_a * norm_b))
            angle_deg = np.degrees(angle_rad)
            angles.append(angle_deg)

        angles = np.asarray(np.abs(angles))

        if(np.all(angles < angle_threshold) or abdominal_reg==1):

            abd_semantic_corres_sets_filtered.append(abd_semantic_corres_set)

        else:
            pass

    head_semantic_corres_sets = head_semantic_corres_sets_filtered
    abd_semantic_corres_sets = abd_semantic_corres_sets_filtered

    return  head_semantic_corres_sets,  abd_semantic_corres_sets


def initial_vessel_rotation(ct_skeleton_pc, ivus_skeleton_pc, axis_ct, axis_ivus, visualize_debug, ct_spheres, ivus_spheres):
    """Rigidly align the CT vessel axis and centerline to the IVUS vessel frame."""
    ct_skeleton_pc_orig = copy.deepcopy(ct_skeleton_pc)

    points = np.asarray(ct_skeleton_pc.points)

    # CT scan
    # assumes longest axis is in the superior-inferior direction
    index_1,index_2 = find_furthest_points(points)

    point_1 = points[index_1,:]
    point_2_ct = points[index_2,:]

    point_2_index_ct = index_2

    if(np.linalg.norm(point_2_ct - points[-1,:]) < np.linalg.norm(point_1 - points[-1,:])):
        placeholder = copy.deepcopy(point_2_ct)
        placeholder_index = copy.deepcopy(index_1)
        point_2_ct = point_1
        point_1 = placeholder
        point_2_index_ct = placeholder_index

    points = np.asarray(ivus_skeleton_pc.points)

    # IVUS map
    # assumes longest axis is in the superior-inferior direction
    index_1,index_2 = find_furthest_points(points)

    point_1 = points[index_1,:]
    point_2_ivus = points[index_2,:]
    point_2_index_ivus = index_2

    if(np.linalg.norm(point_2_ivus - points[-1,:]) < np.linalg.norm(point_1 - points[-1,:])):
        placeholder = copy.deepcopy(point_2_ivus)
        placeholder_index = copy.deepcopy(index_1)
        point_2_ivus = point_1
        point_1 = placeholder
        point_2_index_ivus = placeholder_index

    # the translation vector to be applied them now that we've determined which end is which
    difference = point_2_ivus - point_2_ct

    # transformation_matrix_2[:3, 3] = difference

    corres = np.asarray([[point_2_index_ct, point_2_index_ivus]])
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    correspondence_set = o3d.utility.Vector2iVector(corres)
    transformation_p2p = estimation.compute_transformation(ct_skeleton_pc, ivus_skeleton_pc, correspondence_set)

    ct_skeleton_pc.transform(transformation_p2p)

    point_ct = get_sphere_cloud([point_2_ct], 0.004, 10, [0,0,1])
    point_ivus = get_sphere_cloud([point_2_ivus], 0.004, 10, [0,0,1])

    corres = get_rigid_geodesic_correspondences_with_max_point(ct_skeleton_pc, ivus_skeleton_pc, point_2_index_ct, point_2_index_ivus, point_ct)
    row_index = np.where(corres[:, 0] == point_2_index_ct)[0]
    corres[row_index,:] = np.asarray([[point_2_index_ct, point_2_index_ivus]])

    correspondence_set = o3d.utility.Vector2iVector(corres)
    transformation_geodesic = estimation.compute_transformation(ct_skeleton_pc, ivus_skeleton_pc, correspondence_set)

    ct_skeleton_pc.transform(transformation_geodesic)

    combined_transformation = transformation_geodesic @ transformation_p2p

    if(visualize_debug == 1):
        # ----- visualize_debug RIGID ROTATION ------- #

        ct_spheres_orig = copy.deepcopy(ct_spheres)
        ct_spheres_def = copy.deepcopy(ct_spheres)
        ct_spheres_def.transform(combined_transformation)
        target_positions_spheres = ct_spheres_def.vertices

        red_pc = copy.deepcopy(ct_skeleton_pc_orig)
        blue_pc = copy.deepcopy(ivus_skeleton_pc)

        time_points = 20

        red_pc.paint_uniform_color([1,0,0])
        blue_pc.paint_uniform_color([0,0,1])

        if(visualize_debug ==1):
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.get_render_option().mesh_show_back_face = True
            vis.add_geometry(red_pc)
            vis.add_geometry(blue_pc)
            vis.add_geometry(ct_spheres_orig)
            vis.add_geometry(ivus_spheres)

            vis.run()

        source_position = np.asarray(red_pc.points)
        target_position = np.asarray(ct_skeleton_pc.points)

        source_position_spheres = np.asarray(ct_spheres_orig.vertices)
        target_position_spheres = np.asarray(target_positions_spheres)

        for i in range(time_points):
            t = i / time_points

            # Interpolate between source points and deformed points
            interpolated_points = (1 - t) * source_position + t * target_position
            red_pc.points = o3d.utility.Vector3dVector(interpolated_points)
            red_pc.paint_uniform_color([1,0,0])

            interpolated_points_spheres = (1 - t) * source_position_spheres + t * target_position_spheres
            ct_spheres_orig.vertices = o3d.utility.Vector3dVector(interpolated_points_spheres)
            ct_spheres_orig.compute_vertex_normals()

            if(visualize_debug == 1):
                vis.update_geometry(red_pc)

                vis.update_geometry(ct_spheres_orig)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.1)

    return combined_transformation, corres


def get_rigid_geodesic_correspondences_with_max_point(ct_skeleton,ivus_skeleton, point_2_index_ct, point_2_index_ivus, point_ct):
    """Build ordered geodesic centerline correspondences anchored at a selected point."""
    ct_skeleton_points = np.asarray(ct_skeleton.points)
    ivus_skeleton_points = np.asarray(ivus_skeleton.points)

    geodesic_ivus = compute_geodesic_distance_on_point_cloud(ivus_skeleton)
    geodesic_ct = compute_geodesic_distance_on_point_cloud(ct_skeleton)

    relevant_corres_ct = np.arange(0,point_2_index_ct)
    relevant_corres_ivus =np.arange(0,point_2_index_ivus)

    # before max point
    segment_geodesic_ct = -(geodesic_ct[0:point_2_index_ct] - geodesic_ct[point_2_index_ct])
    segment_geodesic_ivus = -(geodesic_ivus[0:point_2_index_ivus] - geodesic_ivus[point_2_index_ivus])

    full_corres = []
    # add the top point anyway
    full_corres.append(np.asarray([point_2_index_ct,point_2_index_ivus]))

    range_array = np.arange(0, len(relevant_corres_ivus) )
    reversed_range = range_array[::-1]

    for i in reversed_range:

        ivus_index = relevant_corres_ivus[i]
        ivus_distance = segment_geodesic_ivus[i]

        closest_index = np.abs(segment_geodesic_ct - (ivus_distance)).argmin()
        ct_index = relevant_corres_ct[closest_index]

        full_corres.append([ct_index,ivus_index])

        if(ivus_index == relevant_corres_ivus[0]):
            break

    # after max point
    relevant_corres_ct = np.arange(point_2_index_ct,np.shape(ct_skeleton_points)[0])
    relevant_corres_ivus = np.arange(point_2_index_ivus,np.shape(ivus_skeleton_points)[0])

    segment_geodesic_ct = (geodesic_ct[point_2_index_ct:np.shape(ct_skeleton_points)[0]] - geodesic_ct[point_2_index_ct])
    segment_geodesic_ivus = (geodesic_ivus[point_2_index_ivus:np.shape(ivus_skeleton_points)[0]] - geodesic_ivus[point_2_index_ivus])

    for i in np.arange(1,len(relevant_corres_ivus)):

        ivus_index = relevant_corres_ivus[i]
        ivus_distance = segment_geodesic_ivus[i]

        closest_index = np.abs(segment_geodesic_ct - (ivus_distance)).argmin()
        ct_index = relevant_corres_ct[closest_index]

        full_corres.append([ct_index,ivus_index])

        if(ivus_index == relevant_corres_ivus[-1]):
            break

    corres = np.asarray(full_corres)

    return corres


def get_geodesic_correspondences_2(ivus_skeleton,ct_skeleton,semantic_corres, chopped=0):
    """Expand sparse semantic branch matches into ordered centerline correspondences."""
    # IMPORTANT - IVUS IS FIRST, NOT CT

    # find the geodesic along the skeletons
    geodesic_ivus = compute_geodesic_distance_on_point_cloud(ivus_skeleton)
    geodesic_ct = compute_geodesic_distance_on_point_cloud(ct_skeleton)

    ivus_skeleton_points = np.asarray(ivus_skeleton.points)
    ct_skeleton_points = np.asarray(ct_skeleton.points)

    # order semantic corres in ct skeleton
    semantic_corres = semantic_corres[semantic_corres[:, 0].argsort()]

    full_corres = []
    scaling_factors =[]
    corres_no = []

    for i in np.arange(0,np.shape(semantic_corres)[0]-1):

        # find the relevant segment indices for that correspondence pair
        relevant_corres_ct = np.arange(semantic_corres[i,0],semantic_corres[i+1,0]+1)
        relevant_corres_ivus = np.arange(semantic_corres[i,1],semantic_corres[i+1,1]+1)

        print("semantic_corres[i,1]", semantic_corres[i,1])
        print("semantic_corres[i+1,1]+1", semantic_corres[i+1,1]+1)

        # find geodesic distances relative to the segment start point, and then normalize those
        segment_geodesic_ct = geodesic_ct[semantic_corres[i,0]:semantic_corres[i+1,0]+1] - geodesic_ct[semantic_corres[i,0]]
        normalized_geodesic_ct = segment_geodesic_ct / segment_geodesic_ct[-1]
        segment_geodesic_ivus = geodesic_ivus[semantic_corres[i,1]:semantic_corres[i+1,1]+1] - geodesic_ivus[semantic_corres[i,1]]

        # so long as we're not at the end of the vessel
        if(len(segment_geodesic_ivus)!=0):

            normalized_geodesic_ivus = segment_geodesic_ivus / segment_geodesic_ivus[-1]

            for i in np.arange(1,len(normalized_geodesic_ct)-1):

                # pull out the index that corresponds to in the original unsegmented centrelines
                ct_index = relevant_corres_ct[i]
                ct_distance = normalized_geodesic_ct[i]

                # find the closest normalized geodesic distance on the ivus map
                closest_index = np.abs(normalized_geodesic_ivus - ct_distance).argmin()
                ivus_index = relevant_corres_ivus[closest_index]

                full_corres.append([ct_index,ivus_index])

    # ----- vessel ends ----- #
    # vessel top

    # also need to account for situations where all segments were already covered above (i.e. closest_indices_ct or ivus are actually endpoints of the vessel)
    # it should be - for each remaining IVUS point find a CT point, only for the vessel ends!

    relevant_corres_ct = np.arange(0,semantic_corres[0,0])
    relevant_corres_ivus = np.arange(0,semantic_corres[0,1])

    # use scaling factor rather than normalization because we can't guarantee completeness
    segment_geodesic_ct = -(geodesic_ct[0:semantic_corres[0,0]] - geodesic_ct[semantic_corres[0,0]])
    segment_geodesic_ivus = -(geodesic_ivus[0:semantic_corres[0,1]] - geodesic_ivus[semantic_corres[0,1]])

    for i in np.arange(1,len(segment_geodesic_ivus)-1):

        ivus_index = relevant_corres_ivus[i]
        ivus_distance = segment_geodesic_ivus[i]

        closest_index = np.abs(segment_geodesic_ct - (ivus_distance)).argmin()
        ct_index = relevant_corres_ct[closest_index]

        full_corres.append([ct_index,ivus_index])

    # vessel bottom

    relevant_corres_ct = np.arange(semantic_corres[-1,0],np.shape(ct_skeleton_points)[0])
    relevant_corres_ivus = np.arange(semantic_corres[-1,1],np.shape(ivus_skeleton_points)[0])

    # use scaling factor rather than normalization because we can't guarantee completeness
    segment_geodesic_ct = (geodesic_ct[semantic_corres[-1,0]:np.shape(ct_skeleton_points)[0]] - geodesic_ct[semantic_corres[-1,0]])
    segment_geodesic_ivus = (geodesic_ivus[semantic_corres[-1,1]:np.shape(ivus_skeleton_points)[0]] - geodesic_ivus[semantic_corres[-1,1]])

    # find the closest geodesic normalized distances for each free corres in ct scan

    for i in np.arange(1,len(segment_geodesic_ivus)-1):

        ivus_index = relevant_corres_ivus[i]
        ivus_distance = segment_geodesic_ivus[i]

        closest_index = np.abs(segment_geodesic_ct - (ivus_distance)).argmin()
        ct_index = relevant_corres_ct[closest_index]

        full_corres.append([ct_index,ivus_index])

    full_corres = np.asarray(full_corres)
    full_corres = np.vstack((full_corres,semantic_corres))
    print("full_corres", full_corres)

    if(chopped==1):
        full_corres = prune_endpoint_correspondences(full_corres,np.shape(ct_skeleton_points)[0],np.shape(ivus_skeleton_points)[0])

    return full_corres


def prune_endpoint_correspondences(full_corres, ct_len, ivus_len):
    """
    full_corres: Nx2 array [ct_index, ivus_index]
    ct_len: number of points in CT skeleton
    ivus_len: number of points in IVUS skeleton
    """

    full_corres = np.asarray(full_corres)

    # Find all correspondences to CT top (0) and CT bottom (ct_len-1)
    top_matches    = full_corres[full_corres[:,0] == 0]
    bottom_matches = full_corres[full_corres[:,0] == ct_len - 1]

    keep = []

    # ----- HANDLE TOP END -----
    if len(top_matches) > 0:
        # Keep the IVUS index that is farthest along IVUS (max index)
        best_ivus = top_matches[:,1].max()
        keep.append([0, best_ivus])

    # ----- HANDLE BOTTOM END -----
    if len(bottom_matches) > 0:
        # Keep the IVUS index closest to bottom (min index or max depending direction)
        best_ivus = bottom_matches[:,1].min()
        keep.append([ct_len - 1, best_ivus])

    # Remove all endpoint correspondences
    mask_mid = (full_corres[:,0] != 0) & (full_corres[:,0] != ct_len - 1)
    mid_corres = full_corres[mask_mid]

    # Add cleaned endpoint correspondences back
    cleaned = np.vstack([mid_corres, np.array(keep)])

    # Sort by IVUS or CT index depending on what you want
    cleaned = cleaned[cleaned[:,1].argsort()]

    return cleaned


def compute_geodesic_distance_on_point_cloud(point_cloud):
    """Compute cumulative centerline distance from the first point."""
    points = np.asarray(point_cloud.points)
    N = len(points)
    GD = np.zeros([N, 1])  # Initialize a 1D array with zeros

    # Compute cumulative Euclidean distances
    for i in range(1, N):
        gdist = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
        GD[i] = GD[i-1] + gdist  # Add the distance to the cumulative sum

    return GD


def get_closest_projected_point(vertex, centerline_points):
    """
    Finds the closest point on the centerline to the vertex.

    Parameters:
    vertex (numpy.ndarray): 1x3 array representing the vertex.
    centerline_points (numpy.ndarray): nx3 array of centerline points.

    Returns:
    interpolated_point (numpy.ndarray): Closest point on the centerline segment.
    insert_index (int): Index where the interpolated point lies.
    """
    # Calculate distances to centerline points and find the nearest point
    diff = centerline_points - vertex
    dist = np.linalg.norm(diff, axis=1)
    nearest_idx = np.argmin(dist)

    # Determine the next index for the segment
    if nearest_idx == 0:
        next_idx = 1
    elif nearest_idx == len(centerline_points) - 1:
        nearest_idx = len(centerline_points) - 2
        next_idx = len(centerline_points) - 1
    else:
        if(dist[nearest_idx + 1] < dist[nearest_idx - 1]):
            next_idx = nearest_idx + 1
        else:
            placeholder = nearest_idx
            nearest_idx = placeholder - 1
            next_idx = placeholder

    # Define segment start and end
    segment_start = centerline_points[nearest_idx]
    segment_end = centerline_points[next_idx]

    segment_direction = segment_end - segment_start
    segment_length = np.linalg.norm(segment_direction)

    # Ensure nonzero segment length

    segment_direction /= segment_length  # Normalize direction vector

    # Project vertex onto the segment
    projection = np.dot(vertex - segment_start, segment_direction)
    projection_factor = projection / segment_length


    # Compute the interpolated point
    interpolated_point = segment_start + projection_factor * segment_direction * segment_length

    # Insert index is always after nearest_idx
    insert_index = nearest_idx + 1

    return interpolated_point, insert_index
