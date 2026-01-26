import numpy as np
from dreimac.utils import CohomologyUtils
from dreimac.combinatorial import combinatorial_number_system_d1_forward as cns_d1_fwd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse.linalg import lsqr
import math

def vec_to_sparse(cc, eta_vec, thresh=None):
    """Convert vector to sparse representation based on threshold."""
    thresh = thresh or cc._rips_threshold
    n_landmarks = cc._n_landmarks
    eta_sparse = []
    for i in range(n_landmarks):
        for j in range(i + 1, n_landmarks):
            if cc._dist_land_land[i, j] < thresh:
                index = cns_d1_fwd(i, j, cc._cns_lookup_table)
                if eta_vec[index] != 0:
                    eta_sparse.append([j, i, eta_vec[index]])
    return eta_sparse

# The following function is made by Katya. Ling will ask for permission to use it.
def decode_combinatorial_number(N, k):
    """
    Decode the number N from the combinatorial number system representation
    back to the k-combination (c_1, c_2, ..., c_k) such that:

      N = (c_k choose k) + (c_{k-1} choose k-1) + ... + (c_1 choose 1)

    The greedy algorithm finds the unique sequence corresponding to N.

    Parameters:
      N (int): The nonnegative integer in the combinatorial number system.
      k (int): The number of elements in the combination.

    Returns:
      tuple: A tuple (c_1, c_2, ..., c_k) representing the combination.
    """
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

def get_lifted_cocycle(tc, thresh=None, cocycle_idx=0):
    """Compute lifted cocycle using MILP optimization."""
    thresh = thresh or tc._rips_threshold
    p = tc._prime
    eta_prime = tc.get_representative_cocycle(cocycle_idx, 1)[2]
    eta_ini = CohomologyUtils.lift_to_integer_cocycle(eta_prime, prime=p)
    eta_ini = CohomologyUtils.sparse_cocycle_to_vector(eta_ini, tc._cns_lookup_table, tc._n_landmarks, int)

    delta1 = CohomologyUtils.make_delta1_compact(tc._dist_land_land, thresh, tc._cns_lookup_table)
    d1cocycle = delta1 @ eta_ini.T
    
    if not np.all(d1cocycle == 0):
        y = d1cocycle // p
        bounds = Bounds(lb=-np.inf, ub=np.inf)
        coboundary_constraints = LinearConstraint(delta1, y, y)
        integrality = np.ones(delta1.shape[1])
        optimizer_solution = milp(np.zeros(delta1.shape[1]), integrality=integrality, constraints=coboundary_constraints, bounds=bounds)
        if optimizer_solution.success:
            solution = optimizer_solution["x"]
            eta = eta_ini - p * np.array(np.rint(solution), dtype=int)
        else:
            solution = None
            eta = eta_ini  
    else:
        solution = np.zeros(len(d1cocycle))  
        eta = eta_ini

    return eta
    
# def compute_cocycle_matrix(tc, cocycle, thresh=None):
#     """Compute the cocycle matrix truncated by the threshold."""
#     thresh = thresh or tc._rips_threshold
#     n_landmarks = tc._n_landmarks
#     dist_land = tc._dist_land_land
#     table = tc._cns_lookup_table
#     cocycle_matrix = np.zeros((n_landmarks, n_landmarks))

#     for i in range(n_landmarks):
#         for j in range(i + 1, n_landmarks):
#             if dist_land[i, j] < thresh:
#                 index = cns_d1_fwd(i, j, table)
#                 cocycle_matrix[i, j] = cocycle[index]
#                 cocycle_matrix[j, i] = -cocycle[index]
#     return cocycle_matrix

def get_harmonic_representative(cc, eta, thresh=None):
    """Get the harmonic representative of a cocycle.
    Args:
        cc: CircularCoords object
        eta: integer 1-cocycle in vector form
        thresh: threshold for building coboundary matrix delta0
    Returns:
        theta: harmonic representative of the 1-cocycle
        integrand: 0-cocycle that gives the harmonic representative
    """
    
    thresh = thresh or cc._rips_threshold
    delta0 = CohomologyUtils.make_delta0(cc._dist_land_land, thresh, cc._cns_lookup_table)
    integrand = lsqr(delta0, eta)[0]
    theta = eta - delta0.dot(integrand)
    return theta, integrand