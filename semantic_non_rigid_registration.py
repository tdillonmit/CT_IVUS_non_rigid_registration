import numpy as np
import open3d as o3d
from joblib import load
import os
import time
import glob
import copy
import sys


from mesh_deformation_functions import *
from viterbi_correspondence_matching import *



def register_branches(dataset, visualize_debug):

    # ----- LOAD CT DATA ------ #
    # centerline data was obtained from VMTK centerline extraction module in 3D Slicer #
    ct_skeleton_pc = o3d.io.read_point_cloud(
        dataset + '/ct_data/ct_skeleton_pc.ply'  # CT data aortic centerline 
    )

    ct_side_branch_pc = o3d.io.read_point_cloud(
        dataset + '/ct_data/ct_side_branch_pc.ply'  # CT data orifice locations 
    )

    ct_slicer_mesh = o3d.io.read_triangle_mesh(
        dataset + '/ct_data/ct_slicer_mesh.ply'  # CT data mesh surface 
    )

    closest_indices_ct = np.load(
        dataset + '/ct_data/closest_indices_ct.npy'  # indices of closest point along ct_skeleton_pc to side branch segmentations
    ).astype(int)

    side_branch_centrelines_pc = o3d.io.read_point_cloud(
        dataset + '/ct_data/side_branch_centrelines_pc.ply'  # CT data side branch centerlines 
    )

    side_branch_centrelines_indices = np.load(
        dataset + '/ct_data/side_branch_centrelines_indices.npy'  # labels indicating branch number for each point in side_branch_centerlines_pc
    ).astype(int)

    ct_axis = o3d.io.read_line_set(
        dataset + '/ct_data/ct_axis.ply'  # 3D line of best fit through centerline data to obtain vessel main axis
    )


    # -------- LOAD IVUS DATA ------- #
    ivus_skeleton_pc = o3d.io.read_point_cloud(
        dataset + '/ivus_data/ivus_skeleton_pc.ply'  # IVUS aortic centerline
    )

    ivus_side_branch_pc = o3d.io.read_point_cloud(
        dataset + '/ivus_data/ivus_side_branch_pc.ply'  # IVUS data orifice locations
    )

    ivus_funsr_mesh = o3d.io.read_triangle_mesh(
        dataset + '/ivus_data/ivus_funsr_mesh.ply'  # IVUS segmented surface from SDF marching cubes
    )

    closest_indices_ivus = np.load(
        dataset + '/ivus_data/closest_indices_ivus.npy'  # indices of closest point along ivus_skeleton_pc to side branch segmentations
    ).astype(int)

    ivus_axis = o3d.io.read_line_set(
        dataset + '/ivus_data/ivus_axis.ply'  # 3D line of best fit through centerline data to obtain vessel main axis
    )

    orifice_pc = o3d.io.read_point_cloud(
        dataset + '/ivus_data/orifice_pc.ply'  # IVUS data orifice locations
    )

    orig_branch_pc = o3d.io.read_point_cloud(
        dataset + '/ivus_data/orig_branch_pc.ply'  # raw branch segmentation as 3D point cloud
    )

    


    # obtain simple dervied objects from the loaded objects
    ct_centroids = np.asarray(ct_side_branch_pc.points)
    ct_spheres = get_sphere_cloud(ct_centroids, 0.004, 10, [1,0,0])
    ivus_centroids = np.asarray(ivus_side_branch_pc.points)
    ivus_spheres = get_sphere_cloud(
        ivus_centroids, 0.004, 10, [0, 0, 1]
    )
    ivus_funsr_lineset = create_wireframe_lineset_from_mesh(
        ivus_funsr_mesh
    )

    # registration parameters
    method = 'ransac'  # 'viterbi', 'ransac','nearest_neighbour'
    visualize_debug = 0
    mesh_downsample =1 # for computational efficiency
    downsample_size = 250
    node_pool_distance = 0.0095

    # # rotate CT scan towards IVUS map
    rotation_transformation, corres = initial_vessel_rotation(ct_skeleton_pc, ivus_skeleton_pc, ct_axis, ivus_axis, 1, ct_spheres, ivus_spheres)


    # downsample mesh
    ct_slicer_mesh.compute_vertex_normals()
    voxel_size = max(ct_slicer_mesh.get_max_bound() - ct_slicer_mesh.get_min_bound()) / downsample_size
    mesh_copy = ct_slicer_mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    
    if(mesh_downsample == 1):
        voxel_size = max(ct_slicer_mesh.get_max_bound() - ct_slicer_mesh.get_min_bound()) / 100
        mesh_smp = ct_slicer_mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
    else:
        voxel_size = max(ct_slicer_mesh.get_max_bound() - ct_slicer_mesh.get_min_bound()) / downsample_size
        mesh_smp = ct_slicer_mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)

    if(mesh_downsample == 1):
        voxel_size = max(ct_slicer_mesh.get_max_bound() - ct_slicer_mesh.get_min_bound()) / 100
        mesh_smp = ct_slicer_mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)
    else:
        voxel_size = max(ct_slicer_mesh.get_max_bound() - ct_slicer_mesh.get_min_bound()) / downsample_size
        mesh_smp = ct_slicer_mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average)

    

    # apply initial rigid vessel transformation
    ct_scan_pc = create_wireframe_lineset_from_mesh(mesh_copy)
    ct_scan_pc.transform(rotation_transformation)
    ct_slicer_mesh.transform(rotation_transformation)
    mesh_smp.transform(rotation_transformation)
    ct_mesh_baseline = copy.deepcopy(mesh_smp)
    ct_side_branch_pc.transform(rotation_transformation)
    side_branch_centrelines_pc.transform(rotation_transformation)


    # generate CT branch detections as spheres
    ct_centroids = np.asarray(ct_side_branch_pc.points)
    ct_spheres = get_sphere_cloud(ct_centroids, 0.004, 10, [1,0,0])
  
    if(visualize_debug == 1):
        print("visualize branch detections and centerlines...")
        o3d.visualization.draw_geometries([ivus_skeleton_pc, ct_skeleton_pc, ct_spheres, ivus_spheres])
    

    # filter obvious branch detection outliers
    ivus_centroids, ivus_skeleton_pc, closest_indices_ivus = filter_spurious_points_3(ivus_skeleton_pc, ct_skeleton_pc, ivus_centroids,ct_centroids, orifice_pc, visualize_debug,closest_indices_ct, orig_branch_pc, ivus_funsr_mesh, ct_scan_pc)

    # generate IVUS branch detections as spheres
    ivus_spheres = get_sphere_cloud(ivus_centroids, 0.004, 10, [0,0,1])


    # ---- CLUSTER SIDE BRANCH POINT CLOUDS ---- #
    
    # find the relevant vertices here
    side_branch_centrelines_pc_points = np.asarray(side_branch_centrelines_pc.points)

    relevant_nodes_list = []
    for check_index in np.unique(side_branch_centrelines_indices):

        relevant_args = np.argwhere(side_branch_centrelines_indices == check_index)[:,0]
        relevant_side_branch_centrelines_pc_points = side_branch_centrelines_pc_points[relevant_args,:]
        relevant_nodes_sub = get_all_nodes_inside_radius(relevant_side_branch_centrelines_pc_points, 0.0085, mesh_smp) # note relevant to original mesh - side_branch_centrelines_indices
        relevant_nodes_list.append(relevant_nodes_sub)

    clustered_pcs = cluster_pc_based_on_centroids(orig_branch_pc, ivus_centroids, distance_threshold = 0.015) 
    

    # --------- CORRESPONDENCE ESTIMATION ----------- #

    # 3 options are viterbi (HMM), RANSAC, or nearest neighbour

    # ------------ VITERBI --------------- #


    abdominal_reg = 0
    
    def run_viterbi_stage():
        semantic_corres_viterbi = get_viterbi_correspondences(
            ct_skeleton_pc, ivus_skeleton_pc, closest_indices_ct, closest_indices_ivus,
            ct_centroids, ivus_centroids)

        corres_viterbi = get_geodesic_correspondences_2(ivus_skeleton_pc, ct_skeleton_pc, semantic_corres_viterbi)

        
        corres_original_viterbi, ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches = show_branched_correspondences(
            corres_viterbi, semantic_corres_viterbi, closest_indices_ct, closest_indices_ivus,
            ct_skeleton_pc, ivus_skeleton_pc, ct_centroids, ivus_centroids, 0)

        ct_lineset, vertsTransformed_full, deformed_lineset = lineset_nrcip_extended(
            ct_skeleton_pc, ivus_skeleton_pc, corres_viterbi, closest_indices_ct, closest_indices_ivus)

        ct_spheres_copied = copy.deepcopy(ct_spheres)
        ct_side_branch_pc_copied = copy.deepcopy(ct_side_branch_pc)
        side_branch_centrelines_pc_copied = copy.deepcopy(side_branch_centrelines_pc)

        a, b, c, d, e = animate_aortascope_deformation(
        mesh_smp=mesh_smp,
        ct_lineset=ct_lineset,
        vertsTransformed_full=vertsTransformed_full,
        ivus_skeleton_pc=ivus_skeleton_pc,
        ivus_spheres=ivus_spheres,
        ct_spheres=ct_spheres_copied,
        ct_side_branch_pc=ct_side_branch_pc_copied,
        side_branch_centrelines_pc=side_branch_centrelines_pc_copied,
        mesh_downsample=mesh_downsample,
        visualize_debug=visualize_debug,
        time_points=30
        )

        # just for visualization in evaluation script
        deformed_ct_centroids = np.asarray(c.points)
        placeholder, ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches = show_branched_correspondences(
        corres_viterbi, semantic_corres_viterbi, closest_indices_ct, closest_indices_ivus, b, ivus_skeleton_pc, deformed_ct_centroids, ivus_centroids, 1)

        return a, b, c, d, e, corres_original_viterbi, ct_lineset, deformed_lineset, ct_lineset_branches, ivus_lineset_branches, corres_lines, corres_lines_branches, semantic_corres_viterbi



    if(method=='viterbi'):
        
        to_deform, canonical_flap_pc, ct_side_branch_pc, \
        side_branch_centrelines_pc, desired_locations, corres_original, ct_lineset, deformed_lineset, \
        ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches, semantic_corres= run_viterbi_stage()
        proportion_inside, colored_pc = quantify_proportion_pc_inside_mesh(orifice_pc, to_deform)
        


    # ------ RANSAC -------- #

    def run_ransac_stage():


        
        head_semantic_corres_sets,  abd_semantic_corres_sets = get_ransac_correspondence_sets(
        ct_skeleton_pc, ivus_skeleton_pc, closest_indices_ct, closest_indices_ivus, ct_centroids, ivus_centroids, ct_side_branch_pc, abdominal_reg, visualize_debug)


        projection_data = precompute_projection_weights(mesh_smp, ct_skeleton_pc)


        all_semantic_corres_ransac_sets = [head_semantic_corres_sets,  abd_semantic_corres_sets]

        
        start_time = time.time()

        best_semantic_corres_ransac = np.empty((0, 2), dtype=int) 



        for semantic_corres_ransac_sets in all_semantic_corres_ransac_sets:

            best_inliers = 0
            best_captures = 0
            semantic_corres_ransac = np.empty((0, 2), dtype=int) 
            
            # RANSAC
            for semantic_corres_ransac_sub in semantic_corres_ransac_sets:



                # get geodesic
                corres_ransac_sub = get_geodesic_correspondences_2(ivus_skeleton_pc, ct_skeleton_pc, semantic_corres_ransac_sub)

                # register centrelines
                ct_lineset, vertsTransformed_full, deformed_lineset = lineset_nrcip_extended(
                ct_skeleton_pc, ivus_skeleton_pc, corres_ransac_sub, closest_indices_ct, closest_indices_ivus)

                ct_side_branch_pc_copied = copy.deepcopy(ct_side_branch_pc)

                # deform mesh
                check_deform= fast_aortascope_deformation(
                mesh_smp=ct_mesh_baseline,
                ct_lineset=ct_lineset,
                vertsTransformed_full=vertsTransformed_full,
                ivus_skeleton_pc=ivus_skeleton_pc,
                ivus_spheres=ivus_spheres,
                ct_side_branch_pc = ct_side_branch_pc_copied,
                projection_data = projection_data,
                mesh_downsample=mesh_downsample,
                visualize_debug=0,
                time_points=30,
                
                )

                # print("relevant_nodes_list", relevant_nodes_list)
                # o3d.visualization.draw_geometries([check_deform]+ clustered_pcs)
                inliers, captures, colored_pcs = quantify_inlier_points_clustered(check_deform, clustered_pcs, relevant_nodes_list, semantic_corres_ransac_sub, closest_indices_ct, closest_indices_ivus)
        
                if(inliers>0):
                    print("total inliers (if any)", inliers)

                check_deform_lineset = create_wireframe_lineset_from_mesh(check_deform)
    
                max_stretch_threshold = 5.0
  
                stretch_factors = compute_triangle_stretch(check_deform, mesh_smp)

                if(np.max(stretch_factors)>max_stretch_threshold and visualize_debug==True):
                    print("detected high stretch!")
 
            
                
                if(np.max(stretch_factors)<max_stretch_threshold):
                    if(captures > best_captures):
                        best_inliers = inliers
                        best_captures = captures
                        semantic_corres_ransac = semantic_corres_ransac_sub
                        # corres_ransac = corres_ransac_sub

                    elif(captures == best_captures):
                        if(inliers > best_inliers):
                            best_inliers = inliers
                            best_captures = captures
                            semantic_corres_ransac = semantic_corres_ransac_sub

            
            best_semantic_corres_ransac = np.vstack((best_semantic_corres_ransac, semantic_corres_ransac))

        
        print("best semantic correspondences were: ", best_semantic_corres_ransac)
        corres_ransac = get_geodesic_correspondences_2(ivus_skeleton_pc, ct_skeleton_pc, best_semantic_corres_ransac)
        semantic_corres_ransac = best_semantic_corres_ransac
       


        ct_lineset, vertsTransformed_full, deformed_lineset = lineset_nrcip_extended(
        ct_skeleton_pc, ivus_skeleton_pc, corres_ransac, closest_indices_ct, closest_indices_ivus)



        corres_original_ransac, ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches = show_branched_correspondences(
        corres_ransac, semantic_corres_ransac, closest_indices_ct, closest_indices_ivus,
        ct_skeleton_pc, ivus_skeleton_pc, ct_centroids, ivus_centroids, 0) #visualize

        ct_spheres_copied = copy.deepcopy(ct_spheres)
        ct_side_branch_pc_copied = copy.deepcopy(ct_side_branch_pc)
        side_branch_centrelines_pc_copied = copy.deepcopy(side_branch_centrelines_pc)


        a, b, c, d, e = improved_animate_aortascope_deformation(
        mesh_smp=mesh_smp,
        ct_lineset=ct_lineset,
        vertsTransformed_full=vertsTransformed_full,
        ivus_skeleton_pc=ivus_skeleton_pc,
        ivus_spheres=ivus_spheres,
        ct_spheres=ct_spheres_copied,
        ct_side_branch_pc=ct_side_branch_pc_copied,
        side_branch_centrelines_pc=side_branch_centrelines_pc_copied,
        mesh_downsample=mesh_downsample,
        visualize_debug=1,
        time_points=30,
        corres=corres_ransac,
        corres_original=corres_original_ransac,
        closest_indices_ct=closest_indices_ct,
        closest_indices_ivus=closest_indices_ivus,
        ivus_centroids = ivus_centroids,
        near_mesh = ivus_funsr_mesh,
        far_pc = orig_branch_pc
        )

    

        inliers, captures,colored_pcs = quantify_inlier_points_clustered(a, clustered_pcs, relevant_nodes_list, best_semantic_corres_ransac, closest_indices_ct, closest_indices_ivus)


        a_lineset = create_wireframe_lineset_from_mesh(a)
        a_lineset.paint_uniform_color([1,0,0])
        orig_branch_pc.paint_uniform_color([0,0,1])

        if(visualize_debug==True):
            o3d.visualization.draw_geometries([a_lineset, orig_branch_pc])

        
        # # just for visualization in evaluation script
        deformed_ct_centroids = np.asarray(c.points)
        placeholder, ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches = show_branched_correspondences(
        corres_ransac, semantic_corres_ransac, closest_indices_ct, closest_indices_ivus, b, ivus_skeleton_pc, deformed_ct_centroids, ivus_centroids, 0) #don't visualize

        return a, b, c, d, e, corres_original_ransac, ct_lineset, deformed_lineset, ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches, semantic_corres_ransac, inliers, corres_ransac


    if(method == 'ransac'):
       
        to_deform, canonical_flap_pc, ct_side_branch_pc, \
        side_branch_centrelines_pc, desired_locations, corres_original, ct_lineset, deformed_lineset, \
        ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches, semantic_corres, proportion_inside, corres = run_ransac_stage()
        proportion_inside, colored_pc = quantify_proportion_pc_inside_mesh(orig_branch_pc, to_deform)
      

            


    # ------------ NEAREST NEIGHBOUR --------------- #

    def run_nn_stage():

        semantic_corres_nn = find_nearest_neighbour_correspondences(ct_centroids, ivus_centroids, closest_indices_ct, closest_indices_ivus)
        corres_nn = get_geodesic_correspondences_2(ivus_skeleton_pc, ct_skeleton_pc, semantic_corres_nn)

        
        corres_original_nn, ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches = show_branched_correspondences(
            corres_nn, semantic_corres_nn, closest_indices_ct, closest_indices_ivus,
            ct_skeleton_pc, ivus_skeleton_pc, ct_centroids, ivus_centroids, 0)

        ct_lineset, vertsTransformed_full, deformed_lineset = lineset_nrcip_extended(
            ct_skeleton_pc, ivus_skeleton_pc, corres_nn, closest_indices_ct, closest_indices_ivus)

        ct_spheres_copied = copy.deepcopy(ct_spheres)
        ct_side_branch_pc_copied = copy.deepcopy(ct_side_branch_pc)
        side_branch_centrelines_pc_copied = copy.deepcopy(side_branch_centrelines_pc)

        a, b, c, d, e = animate_aortascope_deformation(
        mesh_smp=mesh_smp,
        ct_lineset=ct_lineset,
        vertsTransformed_full=vertsTransformed_full,
        ivus_skeleton_pc=ivus_skeleton_pc,
        ivus_spheres=ivus_spheres,
        ct_spheres=ct_spheres_copied,
        ct_side_branch_pc=ct_side_branch_pc_copied,
        side_branch_centrelines_pc=side_branch_centrelines_pc_copied,
        mesh_downsample=mesh_downsample,
        visualize_debug=0,
        time_points=30
        )

        # just for visualization in evaluation script
        deformed_ct_centroids = np.asarray(c.points)
        placeholder, ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches = show_branched_correspondences(
        corres_nn, semantic_corres_nn, closest_indices_ct, closest_indices_ivus, b, ivus_skeleton_pc, deformed_ct_centroids, ivus_centroids, 0)

        return a, b, c, d, e, corres_original_nn, ct_lineset, deformed_lineset, ct_lineset_branches, ivus_lineset_branches, corres_lines, corres_lines_branches, semantic_corres_nn

    if(method == 'nearest_neighbour'):
        to_deform, canonical_flap_pc, ct_side_branch_pc, \
        side_branch_centrelines_pc, desired_locations, corres_original, ct_lineset, deformed_lineset, \
        ct_lineset_branches, ivus_lineset_branches, corres_lines,corres_lines_branches, semantic_corres = run_nn_stage()
        
        proportion_inside, colored_pc = quantify_proportion_pc_inside_mesh(orig_branch_pc, to_deform)
                


    

    ct_centroids = np.asarray(ct_side_branch_pc.points)
    ct_spheres = get_sphere_cloud(ct_centroids, 0.004, 10, [1,0,0])

    


    # ----- FINAL ADJUSTMENTS ----- #


    slide_increment = 0.00015
    twist_increment = 2*np.pi / 300
    nricp_wireframe_before = create_wireframe_lineset_from_mesh(to_deform)
    nricp_wireframe_before.paint_uniform_color([1,0,0])

    to_deform_adjust, ct_centroids_adjust, side_branch_centrelines_pc_adjust, euclidean_errors = slide_and_twist_branches(to_deform, ct_centroids,ivus_centroids,corres_original,canonical_flap_pc,side_branch_centrelines_pc,slide_increment, twist_increment, side_branch_centrelines_indices, orig_branch_pc, 0, node_pool_distance=node_pool_distance)


    # update after adjustment
    nricp_wireframe = create_wireframe_lineset_from_mesh(to_deform_adjust)
    to_deform = to_deform_adjust
    nricp_wireframe.paint_uniform_color([0,0,1])
    ct_centroids = ct_centroids_adjust
    side_branch_centrelines_pc = side_branch_centrelines_pc_adjust
    ct_spheres = get_sphere_cloud(ct_centroids, 0.004, 10, [1,0,0])
   

    if(visualize_debug==True):       
        # render adjustment deformation
        o3d.visualization.draw_geometries([nricp_wireframe,nricp_wireframe_before,orig_branch_pc])
    nricp_wireframe.paint_uniform_color([1,0,0])


    # ---- VISUALIZE RESULTS SIMILAR TO FIGURE 3 ----- #  

    # group the filtered IVUS branch points
    constraint_locations = np.asarray(side_branch_centrelines_pc.points)
    side_branch_points_grouped = []
    for check_index in np.unique(side_branch_centrelines_indices):
        relevant_args = np.argwhere(side_branch_centrelines_indices == check_index)[:,0]
        relevant_side_branch_centrelines_pc_points = constraint_locations[relevant_args,:]
        side_branch_points_grouped.append(relevant_side_branch_centrelines_pc_points)

    # generate the crosses similar to figure 3
    ct_rings = o3d.geometry.TriangleMesh()
    branch_normals = []
    for centroid, relevant_side_branch_centrelines_pc_points  in zip(ct_centroids, side_branch_points_grouped):
        minor_radius = 0.000275
        major_radius = 0.00425 
        origin = relevant_side_branch_centrelines_pc_points[0,:]
        normal = relevant_side_branch_centrelines_pc_points[ 5, :] - origin
        normal = normal / np.linalg.norm(normal)
        branch_normals.append(normal)
        torus, cross = create_torus_with_2d_cross(origin, normal, major_radius, minor_radius, 60)
        torus.paint_uniform_color([1,0,0])
        ct_rings = ct_rings + torus 
    
    ivus_rings, ivus_crosses = get_ivus_rings(ivus_centroids,corres_original, clustered_pcs,canonical_flap_pc, branch_normals, visualize_debug)
    

    ivus_funsr_mesh_pc = o3d.geometry.PointCloud()
    ivus_funsr_mesh_pc.points = ivus_funsr_mesh.vertices
    ivus_funsr_mesh_pc.paint_uniform_color([0,0,1])
    orig_branch_pc.paint_uniform_color([0,0,1])
    ivus_funsr_mesh_pc = ivus_funsr_mesh.sample_points_uniformly(number_of_points=8000) #override
    ivus_funsr_mesh_pc.paint_uniform_color([0,0,1])
    to_deform_tubeset = convert_linesets_to_tubes([nricp_wireframe], radius =0.00003, resolution=3, color=[1,0,0])
    o3d.visualization.draw_geometries([ct_rings, ivus_funsr_mesh_pc, orig_branch_pc, ivus_crosses, nricp_wireframe]) 

    print("registration complete")

    

if __name__ == "__main__":

    dataset_path = sys.argv[1]
    dataset = 'datasets/' + dataset_path
    visualize_debug = 0
    
    if len(sys.argv) > 2:
        
        visualize_debug = sys.argv[2]
    else:
        pass
    
    

    print("calling register branches")
    register_branches(dataset, visualize_debug)

    




