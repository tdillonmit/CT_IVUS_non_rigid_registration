"""Viterbi skeleton-matching helpers used by semantic registration."""
"""Adapted from https://github.com/PRBonn/4d_plant_registration Chebrolu et al, UBonn"""

import numpy as np
from dijkstar import Graph, find_path
import open3d as o3d


def get_viterbi_correspondences(ct_skeleton_pc,ivus_skeleton_pc,closest_indices_ct,closest_indices_ivus,ct_centroids,ivus_centroids):
  """Run labeled skeleton matching and convert matched branch nodes to centerline indices."""
  S1 = get_branched_viterbi_skeleton(ct_skeleton_pc,closest_indices_ct,ct_centroids)

  S2 = get_branched_viterbi_skeleton(ivus_skeleton_pc,closest_indices_ivus,ivus_centroids)

  unconnected = get_unconnected_point_indices(np.asarray(S1._XYZ),
                                          np.asarray(S1._edges))

  if len(unconnected) > 0:
      print(f"There are {len(unconnected)} unconnected points: {unconnected}")
  else:
      print("All points are connected to at least one line.")

  unconnected = get_unconnected_point_indices(np.asarray(S2._XYZ),
                                          np.asarray(S2._edges))

  if len(unconnected) > 0:
      print(f"There are {len(unconnected)} unconnected points: {unconnected}")
  else:
      print("All points are connected to at least one line.")

  params = {'weight_e': 0.01, 'match_ends_to_ends': False,  'use_labels' : True, 'label_penalty' : 1, 'debug': True}
  semantic_corres = skeleton_matching(S1, S2, params)
  S1_pcd = o3d.geometry.PointCloud()
  S1_pcd.points = o3d.utility.Vector3dVector(S1._XYZ)
  S1_pcd.paint_uniform_color([0,0,1])
  S2_pcd = o3d.geometry.PointCloud()
  S2_pcd.points = o3d.utility.Vector3dVector(S2._XYZ)
  S2_pcd.paint_uniform_color([1,0,0])

  corres_some = []
  relevant_ct_indices = np.argwhere(S1._labels==1)
  relevant_ivus_indices =  np.argwhere(S2._labels==1)

  filtered_semantic_corres=[]


  for corres_pair in semantic_corres:

      if(np.isin(corres_pair[0] , relevant_ct_indices)):

          if(np.isin(corres_pair[1] , relevant_ivus_indices)):
              filtered_semantic_corres.append(corres_pair)

  semantic_corres = filtered_semantic_corres



  for corres_pair in semantic_corres:

      diff = np.asarray(ct_skeleton_pc.points) - S1._XYZ[corres_pair[0],:]
      dist = np.linalg.norm(diff, axis=1)
      nearest_ct_idx = np.argmin(dist)

      diff = np.asarray(ivus_skeleton_pc.points) - S2._XYZ[corres_pair[1],:]
      dist = np.linalg.norm(diff, axis=1)
      nearest_ivus_idx = np.argmin(dist)

      corres_some.append([nearest_ct_idx, nearest_ivus_idx])

  semantic_corres_viterbi = np.asarray(corres_some)
  # semantic_corres_viterbi = np.asarray(
  #     corres_some,
  #     dtype=np.int64,
  # ).reshape(-1, 2)

  if semantic_corres_viterbi.shape[0] == 0:
      raise RuntimeError(
          "Viterbi produced no valid branch correspondences after "
          "label filtering."
      )

  return semantic_corres_viterbi

def get_unconnected_point_indices(points, lines):
  """Return point indices that do not occur in any line segment."""
  n_points = points.shape[0]

  # Flatten line indices and get unique point indices used in lines
  connected_indices = np.unique(lines.flatten())

  all_indices = np.arange(n_points)

  # Find which indices are not in connected_indices
  unconnected = np.setdiff1d(all_indices, connected_indices)

  return unconnected

class Skeleton():
  """Graph representation of a labeled 3D skeleton used by the matching HMM."""
  def __init__(self, XYZ = None, edges = None , labels = None):
    self._XYZ = XYZ if XYZ is not None else np.empty((0,3))
    self._edges = edges if edges is not None else []
    self._labels = labels if labels is not None else []
    self.__compute_adjacency_matrix__()

  def add_vertex(self, vertex):
    """ Adds a vertex/node to the skeleton graph
    """
    self._XYZ = np.vstack((self._XYZ, vertex))
    #update A matrix
    self.__compute_adjacency_matrix__()

  def add_edge(self, vertex1_id, vertex2_id):
    """ Adds an edge to the skeleton graph
    """
    edge = np.array([vertex1_id, vertex2_id], dtype=np.uint8)
    self._edges.append(edge)
    # update A matrix
    self.__compute_adjacency_matrix__()

  def add_label(self, label):
    """ Assigns a label to a vertex/node in the skeleton graph
    """
    self._labels.append(label)

  def __compute_adjacency_matrix__(self):
    """ Computes an adjecency matrix from the edge list.
    """
    num_vertices =self._XYZ.shape[0]
    self.A = np.zeros([num_vertices, num_vertices], dtype=np.uint8)
    if num_vertices > 0:
      for e in self._edges:
        if e[0] != e[1]:
          self.A[e[0], e[1]] = 1
          self.A[e[1], e[0]] = 1

  def get_sequence(self):
    """ Computes a sequence along the skeleton in a depth-first manner.
    """
    if self._XYZ.shape[0] > 0:
      root_idx = np.argmin(self.XYZ[:, 2])
      seq = self.__graph_depth_first_traversal__(root_idx)
    else:
      seq = None
    return seq

  def __graph_depth_first_traversal__(self, root_idx, seq=None, old_root_idx=-1):
    """ Recursive function to traverse the skeleton graph in depth first manner.
    """
    if seq == None:
      seq = []
    seq.append(root_idx)
    for i in range(self.A.shape[0]):
      if i != old_root_idx and self.A[root_idx, i] == 1:
        seq = self.__graph_depth_first_traversal__(i, seq, root_idx)
    return seq

  @property
  def XYZ(self):
    return self._XYZ

  @property
  def edges(self):
    return self._edges

  @property
  def labels(self):
    return self._labels


def skeleton_matching(S1, S2, params):
  """
  Computes the correspondences given for a skeleton pair using a HMM formulation.

  Parameters
  ----------
  S1, S2 : Skeleton Class
    Two skeletons for which we compute the correspondences
  params : Dictionary
    Can be an empty dict (All params have some default values)

    weight_e: Used in builing the emissions matrix E.It weighss the difference between the
              skeleton point vs. the difference in the degree of a graph node.
              Set to low number if you want more matches end-points to end-points
              default: 10.0
    match_ends_to_ends: if false, an endpoint to a middlepoint gets equal weights for E
                        as middlepoint to middlepoint.
                        default:false
    use_labels:         If true, then labels are used for the emmision cost.
                        Labels must be given in S.labels for every node.
                        default: false
    label_penalty:      penalty for emmision if labels are not the same.
                        default: 1 (1 means: same cost as node degree with a difference of 1)
    debug:              show plots and other info (true or false)
                        default: false
  Returns
  -------
  corres : numpy array [Mx2]
    column 0 has node ids from S1, and column 1 has corresponding node ids from S2.

  """
  print("Computing matches.")
  # set default params is not provided
  # in emissions E  : weights the difference between the
  # skeleton point vs. the difference in the degree of a graph node
  if 'weight_e' not in params:
    params['weight_e'] = 10.0

  # use semantic labels
  if 'use_labels' not in params:
    params['use_labels'] = False

  # apply label penalty or not
  if 'label_penalty' not in params:
    params['label_penalty'] = False

  # show debug msgs/vis
  if 'debug' not in params:
    params['debug'] = False

  # define HMM S1->S2
  print("HMM: Computing Transition and Emission probabilities")
  T1, E1, statenames1 = define_skeleton_matching_hmm(S1,S2, params);

  # Transform T and E to probability
  to_prob = lambda x : 1/x
  T1 = to_prob(T1)
  E1 = to_prob(E1)

  # compute correspondence pairs using viterbi
  print("Running Viterbi")
  V = np.array(S1.get_sequence())
  best_seq = viterbi(V, T1, E1, statenames1)
  corres = get_correspondences_from_seq(best_seq)

  # remove all matchings to virtual 'nothing' node
  ind_remove = np.where(corres[:,1]==-1)
  corres = np.delete(corres, ind_remove, 0)

  # post process
  corres = remove_double_matches_in_skeleton_pair(S1, S2, corres)
  # filtered_semantic_corres = np.asarray(
  #     corres,
  #     dtype=np.int64,
  # ).reshape(-1, 2)

  # if filtered_semantic_corres.shape[0] > 0:
  #     filtered_semantic_corres = remove_double_matches_in_skeleton_pair(
  #         S1,
  #         S2,
  #         filtered_semantic_corres,
  #     )

  # visualize matching results

  return corres


def define_skeleton_matching_hmm(S1, S2, params):
  """
  Define cost matrices for Hidden Markov Model for matching skeletons

  Parameters
  ----------
  S1, S2 :  Skeleton Class
            Two skeletons for which we compute the correspondences
  params :  Dictionary
            see skeleton_matching function for details

  Returns
  -------
  T : numpy array
      Defines the cost from one pair to another pair
  E : numpy array
      emmision cost matrixd efines the cost for observing one pair as a match
  statenames : list
               names for each state used in the HMM (all correspondence pairs)

  """
  # define statenames
  statenames = define_statenames(S1, S2)

  # Precompute geodesic distances
  GD1, NBR1  = compute_geodesic_distance_on_skeleton(S1)
  GD2, NBR2  = compute_geodesic_distance_on_skeleton(S2)

  # Precompute euclidean distances between all pairs
  ED = compute_euclidean_distance_between_skeletons(S1, S2)

  # compute transition matrix
  T = compute_transition_matrix(S1, S2, GD1, NBR1, GD2, NBR2, ED)

  # compute emission matrix
  E = compute_emission_matrix(S1, S2, ED, params)

  return T, E, statenames


def define_statenames(S1, S2):
  """Enumerate all matched-node and unmatched-node HMM states."""
  # number of nodes in each skeleton
  N = S1.XYZ.shape[0]
  M = S2.XYZ.shape[0]

  statenames = []
  # starting state
  statenames.append([-2, -2])
  for n1 in range(N):
    for m1 in range(M):
        statenames.append([n1, m1])
  for n1 in range(N):
    statenames.append([n1, -1])

  return statenames


def get_correspondences_from_seq(seq):
  """Convert a decoded state sequence into a correspondence array."""
  N = len(seq)
  corres = np.zeros((N,2), dtype=np.int64)
  for i in range(N):
    corres[i,0] = seq[i][0]
    corres[i,1] = seq[i][1]

  return corres


def compute_geodesic_distance_on_skeleton(S):
  """Compute pairwise geodesic distances and intervening branch counts."""
  G = Graph(undirected=True)
  edges = S.edges
  for e in edges:
    edge_length =  euclidean_distance(S.XYZ[e[0],:], S.XYZ[e[1],:])
    G.add_edge(e[0], e[1], edge_length)

  N = S.XYZ.shape[0]
  GD = np.zeros([N,N])
  NBR = np.zeros([N,N])
  for i in range(N):
    for j in range(i+1,N):
      gdist, nbr = geodesic_distance_no_branches(G, S.A, i, j)
      GD[i,j]=gdist
      GD[j,i]=gdist
      NBR[i,j]=nbr
      NBR[j,i]=nbr

  return GD, NBR


def geodesic_distance_no_branches(G, A, i, j):
  """Return shortest-path length and excess branch count between two nodes."""
  path = find_path(G, i, j)
  gdist = path.total_cost

  nbr = 0
  for i in range(1, len(path.nodes)):
    degree = sum(A[path.nodes[i],:])
    nbr = nbr + (degree-2) # A normal inner node has degree 2, thus additional branches are degree-2

  return gdist , nbr


def mean_distance_direct_neighbors(S, GD):
  """Compute the mean edge length of directly connected skeleton nodes."""
  N = S.XYZ.shape[0]
  sumd=0
  no=0
  for i in range(N):
    for j in range(N):
      if S.A[i,j]:
        sumd = sumd + GD[i,j]
        no=no+1

  mean_dist = sumd/no

  return mean_dist


def euclidean_distance(x1, x2):
  """Compute Euclidean distance between two 3D points."""
  dist = np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 + (x1[2]-x2[2])**2)
  return dist


def compute_euclidean_distance_between_skeletons(S1, S2):
  """Compute all pairwise Euclidean distances between two skeletons."""
  N = S1.XYZ.shape[0]
  M = S2.XYZ.shape[0]

  D = np.zeros([N,M])
  for n1 in range(N):
    for m1 in range(M):
      D[n1,m1] =  euclidean_distance(S1.XYZ[n1,:], S2.XYZ[m1,:])

  return D


def remove_double_matches_in_skeleton_pair(S1, S2, corres):
  """
  Remove double matches of two skeletons and keeps only the one with the smallest distance.

  Parameters
  ----------
  S1, S2 : Skeleton Class
           Two skeletons for which we compute the correspondences
  corres : numpy array (Mx2)
           correspondence between two skeleton nodes pontetially with one-to-many matches

  Returns
  -------
  corres: numpy array
          one-to-one correspondences between the skeleton nodes.

  """
  num_corres = corres.shape[0]
  distances = np.zeros((num_corres,1))
  for i in range(num_corres):
    distances[i] = euclidean_distance(S1.XYZ[corres[i,0],:], S2.XYZ[corres[i,1],:])

  # remove repeated corres 1 -> 2
  corres12, counts12 = np.unique(corres[:,0], return_counts = True)
  ind_remove12 = []
  for i in range(len(corres12)):
    if counts12[i] > 1:
      ind_repeat = np.argwhere(corres[:,0] == corres12[i])
      dist_repeat = distances[ind_repeat]
      ind_ = np.argsort(dist_repeat)[1:]
      ind_remove12.append(ind_repeat[ind_])
  corres = np.delete(corres, ind_remove12, axis=0)
  distances = np.delete(distances, ind_remove12, axis=0)

  # remove repeated corres 2 -> 1
  corres21, counts21 = np.unique(corres[:,1], return_counts = True)
  ind_remove21 = []
  for i in range(len(corres21)):
    if counts21[i] > 1:
      ind_repeat = np.argwhere(corres[:,1] == corres21[i]).flatten()
      dist_repeat = distances[ind_repeat].flatten()
      ind_ = np.argsort(dist_repeat)[1:]
      ind_remove21.append(ind_repeat[ind_])

  print("corres before final delete", corres)
  print("ind remove 21", ind_remove21)
  if not ind_remove21:
    print("list is empty")
  else:
    flat_indices = np.concatenate(ind_remove21)
    corres = np.delete(corres, flat_indices, axis=0)
    distances = np.delete(distances, flat_indices, axis=0)

  return corres


def compute_transition_matrix(S1, S2, GD1, NBR1, GD2, NBR2, ED):
  
  N = S1.XYZ.shape[0]
  M = S2.XYZ.shape[0]
  T= 1e6*np.ones([N*M +N, N*M +N], dtype=np.float64)

  mean_dist_S1 = mean_distance_direct_neighbors(S1, GD1)
  max_cost_normal_pairs=0

  # normal pair to normal pair:
  # (n1,m1) first pair, (n2,m2) second pair
  max_cost_normal_pairs = 0
  for n1 in range(N):
    for m1 in range(M):
      for n2 in range(N):
        for m2 in range(M):        
          # Avoid going in different directions on the skeleton. Then the
          # geodesic difference can be small, but actually one would assume a
          # large cost
          v_inS1 =  S1.XYZ[n2,:] - S1.XYZ[n1,:]
          v_inS2 = S2.XYZ[m2,:] - S2.XYZ[m1,:]
         
          # angle between vectors smaller 90 degrees
          if np.dot(v_inS1, v_inS2) >= 0:                    
            # geodesic distance and difference in number of branches along the way
            g1 = GD1[n1,n2]
            g2 = GD2[m1,m2]
            br1= NBR1[n1,n2]
            br2= NBR2[m1,m2]
            v = np.abs(br1-br2)*np.max(GD1) + np.abs(g1-g2) + mean_dist_S1           
            T[n1*M+m1, n2*M+m2] = v

            if v > max_cost_normal_pairs:
               max_cost_normal_pairs = v

  # Main diagonal should be large
  for i in range(N*M):
    T[i,i] = max_cost_normal_pairs

  # Normal pair -> not present
  for n1 in range(N):
    for m1 in range(M):
      for n2 in range(N):
        if n2 != n1:
          T[n1*M+m1 , N*M + n2] = np.max(GD1)/2 

  # Not present -> normal pair
  for n2 in range(N):
    for m1 in range(M):
      for n1 in range(N):
        T[N*M + n2, n1*M+m1] = max_cost_normal_pairs

  # Not present -> not present
  S1_seq = S1.get_sequence()
  for n1 in range(N):
    for n2 in range(N):
      pos_n1 = S1_seq.index(n1) if n1 in S1_seq else -1 
      pos_n2 = S1_seq.index(n2) if n2 in S1_seq else -1
      if pos_n2 > pos_n1:
        v = GD1[n1, n2]
        T[N*M + n1, N*M + n2] = v + mean_dist_S1 + np.min(ED[n1,:])
  
  # Add starting state (which will not be reported in hmmviterbi)           
  sizeT = N*M +N;
  T1 = np.hstack((1e6*np.ones((1,1)), np.ones((1,N*M)), 1e6*np.ones((1,N))))     
  T2 = np.hstack((1e6*np.ones((sizeT, 1)), T))
  T = np.vstack( (T1, T2))
  
  return T 

# def compute_transition_matrix(
#     S1,
#     S2,
#     GD1,
#     NBR1,
#     GD2,
#     NBR2,
#     ED,
#     angle_weight=None,
#     max_angle_deg=90.0,
# ):
#     """Build HMM transition costs from distance, branches, and direction."""

#     N = S1.XYZ.shape[0]
#     M = S2.XYZ.shape[0]

#     invalid_cost = 1e6
#     T = np.full(
#         (N * M + N, N * M + N),
#         invalid_cost,
#         dtype=np.float64,
#     )

#     mean_dist_S1 = mean_distance_direct_neighbors(S1, GD1)

#     geodesic_scale = max(np.max(GD1), np.max(GD2))

#     # Tune these after inspecting selected-path costs.
#     angle_weight = geodesic_scale
#     stationary_weight = geodesic_scale

#     eps = 1e-12
#     max_cost_normal_pairs = 0.0

#     for n1 in range(N):
#         for m1 in range(M):
#             source_index = n1 * M + m1

#             for n2 in range(N):
#                 for m2 in range(M):
#                     target_index = n2 * M + m2

#                     v_inS1 = S1.XYZ[n2] - S1.XYZ[n1]
#                     v_inS2 = S2.XYZ[m2] - S2.XYZ[m1]

#                     norm1 = np.linalg.norm(v_inS1)
#                     norm2 = np.linalg.norm(v_inS2)

#                     moving1 = norm1 > eps
#                     moving2 = norm2 > eps

#                     # Same correspondence state: handled later by the
#                     # large main-diagonal cost.
#                     if not moving1 and not moving2:
#                         continue

#                     if moving1 and moving2:
#                         cosine = np.dot(v_inS1, v_inS2) / (
#                             norm1 * norm2
#                         )
#                         cosine = np.clip(cosine, -1.0, 1.0)

#                         # Continuous angular penalty:
#                         #   0 degrees   -> 0
#                         #   90 degrees  -> angle_weight
#                         #   180 degrees -> 2 * angle_weight
#                         angle_penalty = angle_weight * (1.0 - cosine)

#                     else:
#                         # Allow different skeleton sampling densities, but
#                         # discourage repeatedly mapping moving S1 nodes to
#                         # one stationary S2 node.
#                         angle_penalty = stationary_weight

#                     g1 = GD1[n1, n2]
#                     g2 = GD2[m1, m2]

#                     br1 = NBR1[n1, n2]
#                     br2 = NBR2[m1, m2]

#                     branch_penalty = (
#                         abs(br1 - br2) * geodesic_scale
#                     )
#                     geodesic_penalty = abs(g1 - g2)

#                     cost = (
#                         branch_penalty
#                         + geodesic_penalty
#                         + angle_penalty
#                         + mean_dist_S1
#                     )

#                     T[source_index, target_index] = cost
#                     max_cost_normal_pairs = max(
#                         max_cost_normal_pairs,
#                         cost,
#                     )

#     # Main diagonal should be large.
#     for i in range(N * M):
#         T[i, i] = max_cost_normal_pairs

#     # Normal pair -> not present.
#     for n1 in range(N):
#         for m1 in range(M):
#             for n2 in range(N):
#                 if n2 != n1:
#                     T[n1 * M + m1, N * M + n2] = (
#                         geodesic_scale / 2
#                     )

#     # Not present -> normal pair.
#     for n2 in range(N):
#         for m1 in range(M):
#             for n1 in range(N):
#                 T[N * M + n2, n1 * M + m1] = (
#                     max_cost_normal_pairs
#                 )

#     # Not present -> not present.
#     S1_seq = S1.get_sequence()

#     for n1 in range(N):
#         for n2 in range(N):
#             pos_n1 = S1_seq.index(n1) if n1 in S1_seq else -1
#             pos_n2 = S1_seq.index(n2) if n2 in S1_seq else -1

#             if pos_n2 > pos_n1:
#                 cost = (
#                     GD1[n1, n2]
#                     + mean_dist_S1
#                     + np.min(ED[n1])
#                 )
#                 T[N * M + n1, N * M + n2] = cost

#     # Add artificial starting state.
#     sizeT = N * M + N

#     T1 = np.hstack((
#         invalid_cost * np.ones((1, 1)),
#         np.ones((1, N * M)),
#         invalid_cost * np.ones((1, N)),
#     ))

#     T2 = np.hstack((
#         invalid_cost * np.ones((sizeT, 1)),
#         T,
#     ))

#     return np.vstack((T1, T2))


def compute_emission_matrix(S1, S2, ED, params):
  """Build HMM emission costs from node geometry, degree, and labels."""
  # emmision matrix (state to sequence)
  # Here: degree difference + euclidean difference inside a pair
  N = S1.XYZ.shape[0]
  M = S2.XYZ.shape[0]
  E = 1e05*np.ones((N*M +N, N))

  for n1 in range(N):
    for m1 in range(M):

      degree1= float(np.sum(S1.A[n1,:]))
      degree2= float(np.sum(S2.A[m1,:]))

      # Do not penalize end node against middle node
      if not params['match_ends_to_ends'] and ((degree1==1 and degree2==2) or (degree1==2 and degree2==1)):
        E[n1*M+m1, n1] = params['weight_e'] * (ED[n1,m1] + 10e-10)
      else:
        E[n1*M+m1, n1]= np.abs(degree1-degree2) + params['weight_e'] * (ED[n1,m1] + 10e-10)

  # Add penalty if labels are not consistent
  if params['use_labels']:
    for n1 in range(N):
      for m1 in range(M):
        if S1.labels[n1] != S2.labels[m1]:
          E[n1*M+m1, n1] = E[n1*M+m1, n1] + params['label_penalty']

  # No match
  for n1 in range(N):
    # Take the  best
    I = np.argsort(E[0:N*M, n1])
    E[N*M +n1, n1] = E[I[0], n1]

  # Add starting state (which will not be reported in hmmviterbi)
  E = np.vstack((1e10*np.ones((1,N)),E))

  return E


def viterbi(V, T, E, StateNames):
  """
  This function computes a sequence given observations, transition and emission prob. using the Viterbi algorithm.

  Parameters
  ----------
  V : numpy array (Mx1)
      Observations
  T : numpy array (NxN)
      Transition probabilities
  E : numpy array (NxM)
      Emission probabilities
  StateNames : list
                ames for each state used in the HMM

  Returns
  -------
  S :  list
      Best sequence of hidden states

  """
  M = V.shape[0]
  N = T.shape[0]

  omega = np.zeros((M, N))
  omega[0, :] = np.log(E[:, V[0]])

  prev = np.zeros((M - 1, N))

  for t in range(1, M):
    for j in range(N):
      # Same as Forward Probability
      probability = omega[t - 1] + np.log(T[:, j]) + np.log(E[j, V[t]])

      # This is our most probable state given previous state at time t (1)
      prev[t-1, j] = np.argmax(probability)

      # This is the probability of the most probable state (2)
      omega[t, j] = np.max(probability)

    # Path Array
    S_ = np.zeros(M)

    # Find the most probable last hidden state
    last_state = np.argmax(omega[M - 1, :])

    S_[0] = last_state

  backtrack_index = 1
  for i in range(M-2,-1, -1):
    S_[backtrack_index] = prev[i, int(last_state)]
    last_state = prev[i, int(last_state)]
    backtrack_index += 1

  # Flip the path array since we were backtracking
  S_ = np.flip(S_, axis=0)

  S = []
  for s in S_:
    S.append(StateNames[int(s)])

  return S


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


def convertEdgesFormat(points, graph):
    """
    convert edges from xyz coordinates to indexes

    :param points: list of nodes in the skeleton
    :param graph: list of edges, as xyz coordinates
    :return: list of edges, as indexes
    """
    edges = []
    for g in graph:
        for point in g:
            edges.append(point)

    ids = []
    for e in edges:
        edge_ids = [np.inf, np.inf]
        for i, p in enumerate(points):
            if np.equal(e[0], p[:-1]).all():
                edge_ids[0] = i
            if np.equal(e[1], p[:-1]).all():
                edge_ids[1] = i
        ids.append(edge_ids)
    return ids


def convert_to_skeleton_class(cnodes, graph):
    """ converts som skeleton to general skeleton type """
    points = []
    for label, s in enumerate(cnodes):
      for p in s:
        points.append(np.append(p, label))
    edges = convertEdgesFormat(points, graph)

    # Put in the skeleton structure
    S = Skeleton()
    for i in range(len(points)):
      p = points[i]
      V = np.array(p[0:3], dtype=float)
      S.add_vertex(V)
      L = int(p[3])
      S.add_label(L)

    for e in edges:
      S.add_edge(e[0], e[1])

    return S


def get_branched_viterbi_skeleton(ct_skeleton_pc,closest_indices_ct,ct_centroids):
    """Construct a labeled skeleton graph containing centerline and branch nodes."""
    args_that_sort = np.argsort(closest_indices_ct)
    closest_indices_ct = np.sort(closest_indices_ct)

    ct_centroids = ct_centroids[args_that_sort,:]

    ct_skeleton_points = np.asarray(ct_skeleton_pc.points)

    print("closest_indices_ct before", closest_indices_ct)

    relevant_points = ct_skeleton_points[closest_indices_ct,:]
    i=0
    revised_points = np.empty((0,3))
    for point in ct_skeleton_points:
        if i%2==0 or i in closest_indices_ct:
            revised_points = np.vstack((revised_points, point))
        i=i+1

    i=0
    closest_indices_ct = []
    for point in revised_points:
        print("point", point)
        if np.any(np.all(relevant_points == point, axis=1)):
            closest_indices_ct.append(i)
        i=i+1

    closest_indices_ct = np.asarray(closest_indices_ct)
    ct_skeleton_points = revised_points

    beforehand = np.shape(ct_skeleton_points)[0]
    graph_new=generate_graph_from_centreline(ct_skeleton_points)

    for closest_index_ct, ct_centroid in zip(closest_indices_ct,ct_centroids):
        graph_edge = np.array([ct_skeleton_points[closest_index_ct, :], ct_centroid])
        graph_new = np.vstack((graph_new, graph_edge[np.newaxis, :, :]))

    ct_skeleton_points = np.vstack((ct_skeleton_points,ct_centroids))

    S1 = convert_to_skeleton_class([ct_skeleton_points], [graph_new])
    labels_1 = np.zeros_like(np.asarray(ct_skeleton_points))[:,0]
    mask = np.zeros(labels_1.shape, dtype=bool)
    mask[closest_indices_ct] = True
    labels_1[mask] = 3
    labels_1[~mask] = 0
    labels_1[beforehand:] = 1

    S1._labels = labels_1

    return S1
