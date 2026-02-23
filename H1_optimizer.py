from collections import defaultdict

from ripser import ripser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import warnings
from scipy.sparse.linalg import lsqr
import math
from dreimac import CircularCoords
from dreimac.utils import CohomologyUtils 
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.csgraph import connected_components


def H0_basis(cobmat):
    '''
    Computes a basis for the 0th cohomology given the 1-coboundary matrix by finding connected components
    '''
    incidence = cobmat.T
    adj_matrix = np.zeros((incidence.shape[0], incidence.shape[0]))
    u = cobmat.nonzero()[1][::2]
    v = cobmat.nonzero()[1][1::2]
    adj_matrix[u, v] = 1
    n_components, labels = connected_components(adj_matrix, directed=False, return_labels=True)
    binary_vectors = np.array([
        np.array(labels == i, dtype=int) for i in range(n_components)
    ])
    return binary_vectors.T
    
def fix_integer_lift(cocycle, p, delta1):
    '''
    Fixes an integer lift of a cocycle by solving a linear program.
    '''
    y = (delta1 @ cocycle) // p
    bounds = Bounds(lb=-np.inf, ub=np.inf)
    coboundary_constraints = LinearConstraint(delta1, y, y)
    integrality = np.ones(delta1.shape[1])
    optimizer_solution = milp(np.zeros(delta1.shape[1]), integrality=integrality, constraints=coboundary_constraints, bounds=bounds)
    if optimizer_solution.success:
        solution = optimizer_solution["x"]
        eta = cocycle - p * np.array(np.rint(solution), dtype=int)
    else:
        print('Failed to fix!')
        solution = None
        eta = cocycle  
    return eta

def decode_combinatorial_number(N, k):
    '''
    Decode a combinatorial number N into a k-combination (c_1, c_2, ..., c_k)  
    '''
    # This will store the decoded digits in order: c_1, c_2, ..., c_k.
    combination = [0] * k

    # Process from the highest index k down to 1.
    for i in range(k, 0, -1):
        # Find the maximum c_i (starting from i) such that math.comb(c_i, i) <= N.
        c = i  # The minimum possible value for c_i is i.
        while math.comb(c, i) <= N:
            c += 1
        # When the loop exits, math.comb(c, i) > N, so the correct value is c - 1.
        c -= 1
        combination[i - 1] = c
        # Subtract the contribution of the current digit.
        N -= math.comb(c, i)

    return tuple(combination)

def set_csr_row_to_zero(csr, row):
    '''
    Helper function to set a row of a csr matrix to zero in place
    '''
    csr.data = np.delete(csr.data, range(csr.indptr[row], csr.indptr[row + 1]))  # drop nnz values
    csr.indices = np.delete(csr.indices, range(csr.indptr[row], csr.indptr[row + 1]))  # drop nnz column indices
    csr.indptr[(row + 1):] = csr.indptr[(row + 1):] - (csr.indptr[row + 1] - csr.indptr[row])

def set_csr_rows_to_zero(csr, row_list):
    '''
    Helper function to set rows of a csr matrix to zero in place
    '''
    to_delete = np.concatenate([range(csr.indptr[row], csr.indptr[row + 1]) for row in row_list])
    csr.data[np.array(to_delete, dtype=int)] = 0
    csr.eliminate_zeros()

def find_birth_death_cochains(X_old, show_solutions=False, save_plots=False, epsilon=0.1, p=41, step=0, lazy_data=None, relative_epsilon=True, q=2, M=None):
    '''
    Finds birth and death cochains for the most prominent 1-dimensional cohomology class in the Rips complex of X.

    Parameters:
        X_old: the euclidean dataset
        show_solutions: whether to plot the solutions
        save_plots: whether to save plots to make a figure for the paper, hard-coded style parameters
        epsilon: how close to each end of the bar to look for relative cochains
        p: the prime to use for the integer lift
        step: the current step for plotting
        lazy_data: a dictionary to store precomputed data
        relative_epsilon: whether epsilon is relative to the persistence or absolute
        q: the norm to use for distances, default 2
        M: precomputed distance matrix (optional)

        
    Returns:
        birth_cochain_support: the edges on which the birth cochain has non-zero values
        birth_cochain_values: the value of the birth cochain on those edges
        death_cochain_support: the shortest edges of each triangle on which the death cochain has non-zero values
        death_cochain_values: the value of the death cochain on those shortest edges
        death_cochain_triangles: the triangles on which the death cochain has non-zero values
        death_cochain_triangle_values: the value of the death cochain on those triangles
        original_birth_time: the original birth time (mostly just for convenience)
        original_death_time: the original death time (mostly just for convenience)
        lazy_data: a set of precomputed boundary matrices which speed up large scale experiments, ignore for now

    '''

    # Step 1: Compute the persistence diagrams and coboundary matrices

    # Step 1a: Compute the persistence diagram
    X = X_old.copy()
    if M is None:
        if q == 2:
            M = euclidean_distances(X, X)
        else:
            M = np.linalg.norm(X[:,None,:] - X[None,:,:], ord=q, axis=2)
        M = 0.5*(M + M.T)
    cc = CircularCoords(M, n_landmarks=len(M), prime=p, distance_matrix=True)
    dgm = cc.dgms_[1]
    if dgm.shape[0] == 0:
        return None, None, None, None, None, None, None, None, lazy_data
    b = dgm[0][0]
    d = dgm[0][1]
    if relative_epsilon:
        dahead = d+(d-b)*epsilon
        dbehind = d-(d-b)*epsilon
        bbehind = b-(d-b)*epsilon
        bahead = b+(d-b)*epsilon
    else:
        dahead = d+epsilon
        dbehind = d-epsilon
        bbehind = b-epsilon
        bahead = b+epsilon


    # Step 1b: Check whether our space has changed enough for us to recompute the coboundary matrices
    cc._dist_land_land = M
    cc._n_landmarks = len(M)
    if lazy_data is None:
        lazy_data = {}
        lazy_data['entries'] = None
        lazy_data['cobmat1'] = CohomologyUtils.make_delta0(M, M.max()*1.1, cc._cns_lookup_table)
        lazy_data['cobmat2'] = CohomologyUtils.make_delta1(M, M.max()*1.1, cc._cns_lookup_table)

    # Step 1c: Find a representative real cocycle alpha for the most persistent H1 point

    def get_triangles(threshold):
        # This helper function gets the indices of all triangles which are in the complex at the given threshold
        M_edge_ordered = np.array([M[decode_combinatorial_number(i,2)] for i in range(lazy_data['cobmat2'].shape[1])])
        triangle_distances = np.abs(lazy_data['cobmat2']) * M_edge_ordered
        max_triangle_distances = triangle_distances.max(axis=1).todense()
        all_three_edges = ((triangle_distances > 0).sum(axis=1) == 3)
        triangles_pre_death = np.where((max_triangle_distances < threshold)*all_three_edges)[0]
        return triangles_pre_death
    
    alpha = cc.get_representative_cocycle(0,1)[2]
    alpha = CohomologyUtils.lift_to_integer_cocycle(alpha, prime=p)
    alpha = CohomologyUtils.sparse_cocycle_to_vector(alpha, cc._cns_lookup_table, cc._n_landmarks, int)
    cobmat2 = lazy_data['cobmat2']
    triangles_at_dbehind = get_triangles(dbehind)
    cobmat2_at_dbehind = cobmat2[triangles_at_dbehind,:]
    if np.any(np.abs(cobmat2_at_dbehind @ alpha) > 0.5):  # this is an integer so easy to check if non-zero
        error = np.max(np.abs(cobmat2_at_dbehind @ alpha))
        print('Need to fix integer lift... ', error, " != 0")
        alpha = fix_integer_lift(alpha, p, cobmat2_at_dbehind)
        print('Fixed lift to now have coboundary at most', np.max(np.abs(cobmat2_at_dbehind @ alpha)))
    alpha = np.array(alpha, dtype=float)

    # Step 2: Compute the birth cochain (the optimal 1-cochain in X_u which is zero in X_t)

    # Step 2a: If alpha is not already zero in X_t, adjust it by subtracting a coboundary which restricts to alpha on X_t
    t = bbehind
    u = bahead
    edges_not_at_u = [i for i in range(len(alpha)) if M[decode_combinatorial_number(i,2)] > u]
    edges_not_at_t = [i for i in range(len(alpha)) if M[decode_combinatorial_number(i,2)] > t]
    cobmat1 = lazy_data['cobmat1'].copy()
    cobmat1_at_u = lazy_data['cobmat1'].copy()

    set_csr_rows_to_zero(cobmat1_at_u, edges_not_at_u)
    cobmat1_at_t = cobmat1_at_u.copy()
    set_csr_rows_to_zero(cobmat1_at_t, edges_not_at_t)
    alpha_at_t = alpha.copy()
    alpha_at_t[edges_not_at_t] = 0
    if np.any(alpha_at_t != 0):
        #print('Adjusting alpha to be zero before birth time')
        alpha -= cobmat1 @ lsqr(cobmat1_at_t, alpha_at_t)[0]
        alpha_at_t = alpha.copy()
        alpha_at_t[edges_not_at_t] = 0
        #if np.any(alpha_at_t != 0):
            #print('Alpha adjusted to be at most ', np.max(np.abs(alpha_at_t)), 'before birth time.')

    # Step 2b: Find the birth cochain
    # We have a long exact sequence ... -> H^0(K) -> H^1(L,K) -> H^1(L) -> ... where L = X_b and K = X_t
    # We want the optimal representative in H^1(L,K) which maps to [alpha] in H^1(L)
    # Since C^0(L,K) is trivial, we need only adjust alpha by elements in the image of the map H^0(K) -> H^1(L,K)
    basis_for_H0_at_t = H0_basis(cobmat1_at_t)
    alpha_at_u = alpha.copy()
    alpha_at_u[edges_not_at_u] = 0
    birth_cochain = cobmat1_at_u @ basis_for_H0_at_t @ lsqr(cobmat1_at_u @ basis_for_H0_at_t, -alpha_at_u)[0] + alpha_at_u
    birth_cochain = np.array(birth_cochain, dtype=float)
    birth_cochain = birth_cochain / np.linalg.norm(birth_cochain, ord=1)
    birth_cochain_support = [
        decode_combinatorial_number(i,2) for i in np.where(np.abs(birth_cochain) > 0)[0]
    ]
    birth_cochain_values = birth_cochain[np.where(np.abs(birth_cochain) > 0)[0]]
    birth_cochain_values = np.array(birth_cochain_values, dtype=float)
    birth_cochain_values = birth_cochain_values / np.linalg.norm(birth_cochain_values, ord=1)

    # Step 3: Compute the death cochain (the optimal 2-coboundary in X_t of a cochain cohomologous to alpha at X_d-)

    # Step 3a: Find the death cochain
    # We know that the 2-coboundary of alpha is in H^2(X_t, X_d-), so we want to find the optimal representative in this class
    # We do this by adding relative 2-coboundaries, i.e. coboundaries of 1-cochains which are zero in X_d-
    s = dbehind
    t = dahead
    triangles_at_t = get_triangles(t)
    edges_at_t = [i for i in range(len(alpha)) if M[decode_combinatorial_number(i,2)] < t]
    edges_not_at_s = [i for i in range(len(alpha)) if M[decode_combinatorial_number(i,2)] > s]
    edges_at_t_not_at_s = [i for i in range(len(alpha)) if s < M[decode_combinatorial_number(i,2)] < t]
    alpha_at_s = alpha.copy()
    alpha_at_s[edges_not_at_s] = 0
    cobmat2_at_t = lazy_data['cobmat2'].copy()[triangles_at_t,:]
    cobmat2_at_t_on_edges_not_at_s = cobmat2_at_t[:,edges_at_t_not_at_s]
    solution = lsqr(cobmat2_at_t_on_edges_not_at_s, -(cobmat2_at_t @ alpha_at_s))[0]
    death_cochain = cobmat2_at_t_on_edges_not_at_s @ solution + (cobmat2_at_t @ alpha_at_s)
    death_cochain = np.array(death_cochain, dtype=float)
    death_cochain = death_cochain / np.linalg.norm(death_cochain, ord=1)

    # Step 3b: Return the death cochain in a usable format for adjusting filtration values
    # The death cochain is a 2-cochain, but we want the list of edges involved in all the triangles and their values

    tri_to_edge_propagation = np.abs(cobmat2_at_t_on_edges_not_at_s).transpose()
    tri_to_edge_propagation = tri_to_edge_propagation / np.max([tri_to_edge_propagation.sum(axis=0), 1e-3*np.ones(tri_to_edge_propagation.shape[1])], axis=0)
    edge_propagation = tri_to_edge_propagation @ np.abs(death_cochain)
    if not isinstance(edge_propagation, np.ndarray):
        edge_propagation = np.array([edge_propagation])
    non_zero_death_cochain_edge_values = np.where(np.abs(edge_propagation) > 1e-12)[0]
    death_cochain_edge_values = edge_propagation[non_zero_death_cochain_edge_values]
    death_cochain_edge_support = [decode_combinatorial_number(edges_at_t_not_at_s[i],2) for i in non_zero_death_cochain_edge_values]

    non_zero_death_cochain_values = np.where(np.abs(death_cochain) > 1e-12)[0]
    death_cochain_support = [decode_combinatorial_number(triangles_at_t[i],3) for i in non_zero_death_cochain_values]
    death_cochain_values = death_cochain[non_zero_death_cochain_values]

    # Step 4 (optional): Make plots for visualization
    if save_plots and step % 50 == 0:
        # plot for paper
        _, ax = plt.subplots(1,1,figsize=(6,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.scatter(X[:,0], X[:,1], c='black', s=100)
        for i,(x,y) in enumerate(birth_cochain_support):
            ax.plot(
                X[[x,y],0], X[[x,y],1], c='green',
                linewidth=np.abs(birth_cochain_values[i]*20), zorder=2
            )
        for i in range(len(death_cochain)):
            (x,y,z) = decode_combinatorial_number(triangles_at_t[i],3)
            vertices = X[[x,y,z],:]
            ax.fill(vertices[:, 0], vertices[:, 1], 'tab:purple', alpha=0.2*np.abs(death_cochain[i])/np.max(np.abs(death_cochain)), edgecolor='tab:purple')
        ax.set_aspect('equal', adjustable='box')
        plt.savefig('figs/cochains_step'+str(step)+'.png', bbox_inches='tight', dpi=150)
        plt.close()
    if show_solutions:
        # plot for detailed analysis
        fig, ax = plt.subplots(1,2,figsize=(6,6))
        extension = alpha.copy()
        extension[edges_at_t_not_at_s] = solution
        ax[0].scatter(X[:,0], X[:,1], c='black')
        ax[1].scatter(X[:,0], X[:,1], c='black')
        for i,(x,y) in enumerate(birth_cochain_support):
            ax[0].plot(
                X[[x,y],0], X[[x,y],1], c='green',
                linewidth=np.abs(birth_cochain_values[i]*10), zorder=2
            )
        for i,(x,y) in enumerate(death_cochain_edge_support):
            ax[0].plot(
                X[[x,y],0], X[[x,y],1], c='purple',
                linewidth=np.abs(death_cochain_edge_values[i]*10), zorder=2
            )
        for i, e in enumerate(edges_at_t):
            x,y = decode_combinatorial_number(e,2)
            ax[1].plot(
                X[[x,y],0], X[[x,y],1], c='red',
                linewidth=np.abs(extension[i]*2), zorder=2
            )
        for i in range(len(death_cochain)):
            (x,y,z) = decode_combinatorial_number(triangles_at_t[i],3)
            vertices = X[[x,y,z],:]
            ax[1].fill(vertices[:, 0], vertices[:, 1], 'tab:purple', alpha=0.4*np.abs(death_cochain[i])/np.max(np.abs(death_cochain)), edgecolor='tab:purple')
        ax[0].set_aspect('equal', adjustable='box')
        ax[1].set_aspect('equal', adjustable='box')
        plt.suptitle('Step {} Birth cochain {:.3f}+-{:.3f}, Death cochain {:.3f}+-{:.3f}'.format(step, b, bahead-b, d, d-dbehind))
        plt.show()
        plt.close() 


    return birth_cochain_support, birth_cochain_values, death_cochain_edge_support, death_cochain_edge_values, death_cochain_support, death_cochain_values, b, d, lazy_data

def get_birth_death_edges(X, q=2):
    '''
    Finds a birth and death edge for the most persistent H1 point.
    Right now, we just look it up by birth and death.
    '''
    if q == 2:
        M = euclidean_distances(X, X)
    else:
        M = np.linalg.norm(X[:,None,:] - X[None,:,:], ord=q, axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ph = ripser(M, distance_matrix=True)
    if ph['dgms'][1].shape[0] == 0:
        return None, None, None, None
    b, d = sorted(ph['dgms'][1], key=lambda x: -x[1]+x[0])[0]
    b_index = np.unravel_index(
        np.argmin(np.abs(M-b)), M.shape
    )
    d_index = np.unravel_index(
        np.argmin(np.abs(M-d)), M.shape
    )
    return b_index, d_index, b, d
  

def update_weights(X, T_old, A=None, method='cochains', gamma=0.01, epsilon=0.1, relative_epsilon=True,
                   lazy_data=None, M=None, regularization=None, q=1):
    '''
    Update feature weights to promote an H1 point

    Parameters:
        X: a n x k array of vectors
        T_old: current feature weights
        A: an optional matrix so that we optimize TAX instead of TX
        method: either single_edges (birth and death edges) or cochains (smoothed cocycle stuff)
        gamma: learning rate
        epsilon: how close to each end of the bar to look for relative cochains
        relative_epsilon: whether epsilon is relative to persistence or absolute
        lazy_data: dictionary to store precomputed data
        M: precomputed distance matrix (optional)
        regularization: optional regularization parameter
        q: the norm to use for distances

    Returns:
        X: the updated dataset
        b_index, d_index: the old birth and death edges (if single_edges method)
        b, d: old birth and death
    '''
    T = T_old.copy()
    if not isinstance(epsilon, list):
        epsilons = [epsilon]
    else:
        epsilons = epsilon
    if A is None:
        A = np.eye(T.shape[0])
        
    
    if method == 'single_edges':
        b_index, d_index, b, d = get_birth_death_edges((T@A)*X, q=1)
        v = X[b_index[0],:] - X[b_index[1],:]
        w = X[d_index[0],:] - X[d_index[1],:]
        grad = np.zeros(T.shape)

        grad -= A @ np.abs(v)
        grad += A @ np.abs(w)

    elif method == 'cochains':
        grad = np.zeros(T.shape)
        TAX = (T @ A)*X
        for epsilon in epsilons:
            b_edges, b_coeffs, d_edges, d_coeffs, _, _, b, d, lazy_data = find_birth_death_cochains(
                TAX, epsilon=epsilon, relative_epsilon=relative_epsilon, lazy_data=lazy_data, q=1, M=M
            )       

            b_coeffs = np.abs(b_coeffs)
            b_edges = np.array(b_edges)
            d_coeffs = np.abs(d_coeffs)
            d_edges = np.array(d_edges)

            for i in range(len(b_edges)):
                grad -= b_coeffs[i] * A @ np.abs(X[b_edges[i, 0], :] - X[b_edges[i, 1], :])
            for i in range(len(d_edges)):
                grad += d_coeffs[i] * A @ np.abs(X[d_edges[i, 0], :] - X[d_edges[i, 1], :])
        
        b_index = None
        d_index = None
            
    else:
        print('invalid method')

    if method == 'cochains':
        grad = grad / len(epsilons)
    if regularization is not None:
        grad -= regularization * T_old
    grad_proj = grad - np.mean(grad)*np.ones(grad.shape)
    
    T = T_old + gamma*grad_proj
    if np.any(T < 0):
        T = project_to_simplex(T)
    return X, T, b_index, d_index, b, d, lazy_data   

def project_to_simplex(weights):
    '''
    Projects a vector to the probability simplex
    '''
    sorted_weights = np.sort(weights)[::-1]
    cumulative_sum = np.cumsum(sorted_weights)
    rho = np.where(sorted_weights + (1 - cumulative_sum) / (np.arange(1, len(weights) + 1)) > 0)[0][-1]
    theta = (cumulative_sum[rho] - 1) / (rho + 1)
    projected_weights = np.maximum(weights - theta, 0)
    return projected_weights

def update_points(
        X_old, method='cochains', penalty=False, normalize=False, gamma=0.01, epsilon=0.1, relative_epsilon=True,
        save_plots=False, show_solutions=False, step=0, lazy_data=None, return_losses=False
    ):
    '''
    Update the points to promote an H1 point

    Parameters:
        X_old: a n x k array of vectors
        method: either single_edges (birth and death edges) or cochains (smoothed cocycle stuff)
        penalty: whether to apply penalty for points leaving unit ball
        normalize: whether to normalize gradient updates
        gamma: learning rate
        epsilon: how close to each end of the bar to look for relative cochains
        relative_epsilon: whether epsilon is relative to persistence or absolute
        save_plots: whether to save plots to disk
        show_solutions: whether to display solution plots
        step: current step number for plotting
        lazy_data: dictionary to store precomputed data

    Returns:
        X: the updated dataset
        b_index, d_index: the old birth and death edges (if single_edges method)
        b, d: old birth and death
    '''
    X = X_old.copy()
    losses = {}
    if not isinstance(epsilon, list):
        epsilons = [epsilon]
    else:
        epsilons = epsilon
        
    if method == 'single_edges':
        b_index, d_index, b, d = get_birth_death_edges(X)
        grad = np.zeros(X.shape)
        if b_index is not None and d_index is not None:
            v = X[b_index[0],:] - X[b_index[1],:]
            v = v/np.linalg.norm(v)
            w = X[d_index[0],:] - X[d_index[1],:]
            w = w/np.linalg.norm(w)
            grad[b_index[0],:] -= v
            grad[b_index[1],:] -= -v
            grad[d_index[0],:] += w
            grad[d_index[1],:] += -w

    elif method == 'cochains':
        grad = np.zeros(X.shape)
        losses['persistence_content'] = 0
        for epsilon in epsilons:
            b_edges, b_coeffs, d_edges, d_coeffs, _, _, b, d, lazy_data = find_birth_death_cochains(
                X, epsilon=epsilon, save_plots=save_plots, show_solutions=show_solutions, step=step, lazy_data=lazy_data, relative_epsilon=relative_epsilon
            )
            if b_edges is not None and d_edges is not None:
                b_coeffs = np.abs(b_coeffs)
                b_edges = np.array(b_edges)
                d_coeffs = np.abs(d_coeffs)
                d_edges = np.array(d_edges)

                vraw = X[b_edges[:, 0], :] - X[b_edges[:, 1], :]
                v = vraw / np.linalg.norm(vraw, axis=1)[:, None]
                wraw = X[d_edges[:, 0], :] - X[d_edges[:, 1], :]
                w = wraw / np.linalg.norm(wraw, axis=1)[:, None]

                for i in range(len(b_edges)):
                    grad[b_edges[i, 0], :] -= b_coeffs[i] * v[i]
                    grad[b_edges[i, 1], :] -= -b_coeffs[i] * v[i]
                for i in range(len(d_edges)):
                    grad[d_edges[i, 0], :] += d_coeffs[i] * w[i]
                    grad[d_edges[i, 1], :] += -d_coeffs[i] * w[i]
                
                if return_losses:
                    losses['persistence_content'] += 1/len(epsilons)*(
                       -np.sum(b_coeffs*np.linalg.norm(vraw, axis=1)) + np.sum(d_coeffs*np.linalg.norm(wraw, axis=1))
                    )
            else:
                print('No H1 class found')

        b_index = None
        d_index = None
            
    else:
        print('invalid method')

    X_old = X.copy()
    grad = grad / len(epsilons)
    if penalty:
        X = X_old + gamma*grad
        X = X - gamma*(X_old - X_old/np.linalg.norm(X_old, axis=1)[:,None])*np.array(np.linalg.norm(X_old, axis=1) > 1)[:,None]
        losses['penalty'] = np.sum(np.maximum(0, (np.linalg.norm(X_old, axis=1) - 1)**2))
    elif normalize:
        grad_proj = grad - np.vdot(grad, X_old)*X_old/np.linalg.norm(X_old)**2
        if show_solutions:
            plt.scatter(X_old[:,0], X_old[:,1], c='black')
            print('Norm of projected grad:', np.linalg.norm(grad_proj))
            for i in range(len(grad)):
                plt.plot([X_old[i,0], X_old[i,0] + grad[i,0]], [X_old[i,1], X_old[i,1] + grad[i,1]], c='red')
            plt.title('Gradient vectors')
            plt.gca().set_aspect(1)
            plt.show()
        X = np.cos(gamma*np.linalg.norm(grad_proj))*X + np.sin(gamma*np.linalg.norm(grad_proj))*grad_proj/np.linalg.norm(grad_proj)
    
    if return_losses:
        return X, b_index, d_index, b, d, lazy_data, losses
    else:
        return X, b_index, d_index, b, d, lazy_data  

def one_step_reweight_enhance_H1(
    X, method='cochains', epsilon=0.1, relative_epsilon=True, A=None):
    '''
    Optimize feature weights using simple line search on a single gradient step
    '''
    T, log = reweight_features_enhance_H1(X, log=True, gamma=1e-6, method=method, max_iter=1, epsilon=epsilon, relative_epsilon=relative_epsilon, A=A)
    grad = log['T'][1] - log['T'][0]
    fastest = np.max(np.abs(grad[np.where(grad < 0)]))
    steplength = 1/(T.shape[0])/fastest
    Tguess = log['T'][0] + grad*steplength
    return Tguess


def move_points_enhance_H1(
    X, gamma=0.1, method='cochains', epsilon=0.1, penalty=False, normalize=False, relative_epsilon=True,
    max_iter=1000, tol=1e-4, verbose=False, log=False, save_plots=False, show_solutions=False, step=0, return_losses=False):
    '''
    Optimize points to promote a H1 feature

    Parameters:
        X: the Euclidean dataset
        gamma: learning rate
        method: single_edges or cochains
        epsilon: how close to each end of the bar to look for relative cochains
        penalty: whether to apply penalty for points leaving unit ball
        normalize: whether to normalize gradient updates
        relative_epsilon: whether epsilon is relative to persistence or absolute
        max_iter: max iterations of gradient descent
        tol: relative convergence tolerance
        verbose: verbose flag
        log: whether to store steps
        save_plots: whether to save plots to disk
        show_solutions: whether to display solution plots
        step: starting step number

    Returns:
        X: the new point cloud
        full_log: log of X values, only if log=True
    '''    
    full_log = defaultdict(list)

    X_old = X.copy()
    lazy_data = None

    for step in tqdm(range(max_iter)):
        thisepsilon = epsilon
        if return_losses:
            X_new, b_index, d_index, b, d, lazy_data, losses = update_points(
                X_old, gamma=gamma, method=method, epsilon=thisepsilon, save_plots=save_plots, relative_epsilon=relative_epsilon,
                show_solutions=show_solutions, step=step, lazy_data=lazy_data, normalize=normalize, penalty=penalty, return_losses=True
            )        
        else:  
            X_new, b_index, d_index, b, d, lazy_data = update_points(
                X_old, gamma=gamma, method=method, epsilon=thisepsilon, save_plots=save_plots, relative_epsilon=relative_epsilon,
                show_solutions=show_solutions, step=step, lazy_data=lazy_data, normalize=normalize, penalty=penalty
            )
        if verbose:
            print(b_index, d_index, b, d)
            plt.scatter(X_old[:,0], X_old[:,1])
            plt.plot(
                [X_old[:,0], X_old[:,0] + 10*(X_new[:,0] - X_old[:,0])],
                [X_old[:,1], X_old[:,1] + 10*(X_new[:,1] - X_old[:,1])],
                c='red'
            )
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show()
        if log:
            if return_losses:
                for loss_type, loss_value in losses.items():
                    full_log[loss_type].append(loss_value)
            full_log['bd'].append([b,d])
            full_log['X'].append(X_old)
        if np.max(np.abs(X_new - X_old)) <= tol:
            break
        else:
            X_old = X_new.copy()
        
    if step >= max_iter-1:
        print('Warning: did not converge')
    else:
        print('Converged in {} steps'.format(step))
    if not log:
        return X_new
    else:
        if return_losses:
            for loss_type, loss_value in losses.items():
                full_log[loss_type].append(loss_value)
        full_log['bd'].append([b,d])
        full_log['X'].append(X_new)
        return X_new, full_log
    
def reweight_features_enhance_H1(
    X, gamma=0.1, method='cochains', epsilon=0.1, relative_epsilon=True, A=None,
    max_iter=1000, tol=1e-6, verbose=False, log=False, step=0, regularization=None):
    '''
    Optimize feature weights to promote a H1 feature

    Parameters:
        X: the Euclidean dataset
        gamma: learning rate
        method: single_edges or cochains
        epsilon: how close to each end of the bar to look for relative cochains
        relative_epsilon: whether epsilon is relative to persistence or absolute
        A: optional transformation matrix to optimize TAX instead of TX
        max_iter: max iterations of gradient descent
        tol: relative convergence tolerance
        verbose: verbose flag
        log: whether to store steps
        step: current step number
        regularization: optional regularization parameter

    Returns:
        T: the feature_weights
        full_log: log of X values, only if log=True
    '''    
    bd_log = []
    T_log = []

    if A is None:
        A = np.eye(X.shape[1])
    T_old = np.ones(A.shape[0])
    T_old = T_old/np.linalg.norm(T_old, ord=1)
    lazy_data = None

    for step in tqdm(range(max_iter)):
        X, T_new, _, _, b, d, lazy_data = update_weights(
            X, T_old, A=A, gamma=gamma, method=method, epsilon=epsilon,
            lazy_data=lazy_data, regularization=regularization, relative_epsilon=relative_epsilon
        )
        if log:
            bd_log.append([b,d])
            T_log.append(T_old)
        if np.all(np.abs(T_new - T_old).sum() <= tol):
            break
        else:
            T_old = T_new.copy()
        
    if step >= max_iter-1:
        print('Warning: did not converge')
    else:
        print('Converged in {} steps'.format(step))
    if not log:
        return T_new
    else:
        bd_log.append([b,d])
        T_log.append(T_new)
        full_log = {'bd': np.array(bd_log), 'T': np.array(T_log)}
        return T_new, full_log
    

def persistence_content(X, epsilon):
    return death_content(X, epsilon) - birth_content(X, epsilon)

def birth_content(X, epsilon):
    birth_cochain_support, birth_cochain_values, _, _, _, _, _, _, _ = find_birth_death_cochains(
        X, epsilon=epsilon
    )
    birth_cochain = np.abs(birth_cochain_values/np.linalg.norm(birth_cochain_values, ord=1))
    birth_content = np.sum([np.linalg.norm(X[i]-X[j])*birth_cochain[k] for k,(i,j) in enumerate(birth_cochain_support)])
    return birth_content

def death_content(X, epsilon):
    _, _, death_cochain_edge_support, death_cochain_edge_values, _, _, _, _, _ = find_birth_death_cochains(
        X, epsilon=epsilon
    )
    death_cochain = np.abs(death_cochain_edge_values)/np.linalg.norm(death_cochain_edge_values, ord=1)
    death_content = np.sum([np.linalg.norm(X[i]-X[j])*death_cochain[k] for k,(i,j) in enumerate(death_cochain_edge_support)])
    return death_content