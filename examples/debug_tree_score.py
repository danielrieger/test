
import numpy as np
from sklearn.neighbors import KDTree

# Mock IMP class
class MockXYZ:
    def __init__(self, coords):
        self.coords = coords
    def get_coordinates(self):
        return self.coords

# Mock AV class
class MockAV:
    def __init__(self, coords):
        self.coords = coords
    def get_particle(self):
        return MockXYZ(self.coords)
    # Mocking what IMP.core.XYZ(av) expects - usually it expects a particle,
    # but in the code it does IMP.core.XYZ(av) directly if av is a particle decorator,
    # or IMP.core.XYZ(av.get_particle()) if av is an AV object.
    # Let's check tree_score.py line 37: IMP.core.XYZ(av)
    # If 'av' is an IMP.bff.AV object, does it inherit from Particle?
    # Usually AVs decorate a particle.
    # Let's assume for this mock that we can pass something that behaves like what IMP.core.XYZ expects.

def computescoretree(tree, modelavs, dataxyz, var, scaling=10.0, searchradius=10.0, offsetxyz=None):
    print("tree_score.py - computescoretree (MOCK)")
    modelxyzs = []
    # In the original code:
    # for av in modelavs:
    #     modelpxyz = IMP.core.XYZ(av)
    #     modelxyz = np.array(modelpxyz.get_coordinates(),dtype=np.float64)
    #     modelxyzs.append(modelxyz)

    # Simplified for mock:
    for av in modelavs:
        modelxyzs.append(av)

    modelxyzs = np.array(modelxyzs, dtype=np.float64)
    
    if offsetxyz is not None:
        modelxyzs = modelxyzs + offsetxyz

    print(f"Model coordinates (with offset): {modelxyzs}")
    print(f"Search radius: {searchradius}")

    ind = tree.query_radius(modelxyzs, searchradius)
    print(f"Indices found: {ind}")
    
    scoretotal = 0
    # BUG SUSPICION: The loop logic here
    # Original:
    # for avidx, dataidx in enumerate(ind):
    #     xM = modelxyzs[avidx]
    #     xD = dataxyz[dataidx]  <-- dataidx is an ARRAY of indices if multiple points found
    #     varD = var[dataidx]
    #     for xDi, varDi in zip(xD, varD):
    #         di = -0.5 * ((xM - xDi) * (xM - xDi)) / varDi
    #         scoretotal += di

    for avidx, data_indices in enumerate(ind):
        xM = modelxyzs[avidx]
        
        if len(data_indices) == 0:
            continue
            
        xD = dataxyz[data_indices]
        varD = var[data_indices]
        
        print(f"AV {avidx} matches {len(data_indices)} points.")
        # print(f"xM: {xM}")
        # print(f"First match xD: {xD[0]}")
        
        # Calculate score manually to see
        for i in range(len(data_indices)):
             xDi = xD[i]
             varDi = varD[i]
             # (xM - xDi) is a vector distance?
             # The original code: di = -0.5 * ((xM - xDi) * (xM - xDi)) / varDi
             # If xM and xDi are 3D points (arrays), then (xM - xDi) * (xM - xDi) is element-wise multiplication
             # So this results in a 3D array [dx^2, dy^2, dz^2]
             # varDi should also be 3D? Or scalar?
             # In NPC_example_BD.py, var is 'smlm_variances'.
             # In flexible_filter_smlm_data, var is returned.
             
             delta = xM - xDi
             dist_sq = delta * delta
             term = -0.5 * dist_sq / varDi
             # If term is 3D array, does scoretotal += di sum all elements?
             # Python += on numpy array might accumulate into a generic variable if it started as 0 (int) or 0.0 (float)?
             # No, if scoretotal=0, scoretotal += array makes scoretotal an array.
             
             scoretotal += term
             
    return scoretotal

# Test Data
data_points = np.array([[0,0,0], [10,10,10], [20,20,20]], dtype=float)
variances = np.array([1.0, 1.0, 1.0]) # Scalar variances? Or 3D? 
# NPC_example_BD.py: smlm_variances is 1D array of scalars usually from localization precision.
# If var is 1D array of size N_points, then var[dataidx] returns array of scalars.
# Then loop: for xDi, varDi in zip(xD, varD):
# xDi is (3,), varDi is scalar.
# (xM - xDi) is (3,). (xM-xDi)*(xM-xDi) is (3,).
# / varDi is (3,).
# scoretotal += (3,) array.

# If the function returns an array [s_x, s_y, s_z] instead of a single scalar score, 
# IMP might treat it as 0 if it expects a float? 
# Or maybe scoretotal is initialized as 0 (scalar) and becomes array.

tree = KDTree(data_points)
model_points = [np.array([1,1,1])] # Near [0,0,0]

score = computescoretree(tree, model_points, data_points, variances, searchradius=5.0)
print(f"Calculated Score: {score}")
