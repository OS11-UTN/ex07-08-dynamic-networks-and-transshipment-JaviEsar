import numpy as np
from scipy.optimize import linprog

def expandInTime(cnn, ubnn, lbnn, startNode, endNode, tnn, steps):
    # Verify input
    if (cnn.shape[0] != ubnn.shape[0]) | (cnn.shape[0] != lbnn.shape[0]) | (cnn.shape[0] != tnn.shape[0]):
        print("ERROR: Matrix must have the same number of rows")
        return [], [], (), []
    if (cnn.shape[1] != ubnn.shape[1]) | (cnn.shape[1] != lbnn.shape[1]) | (cnn.shape[1] != tnn.shape[1]):
        print("ERROR: Matrix must have the same number of columns")
        return [], [], (), []

    # Find the existing arcs
    arcs = np.argwhere((cnn != 0) | (ubnn != 0) | (lbnn != 0))

    # Create the node-arc matrix and bound tuple
    stnc = ubnn.shape[0]  # Same Time Node Count (Block size of a single time node collection)
    stac = arcs.shape[0]  # Same Time Arc Count (Block size of a single time arc collection)
    na = np.zeros([(stnc*steps)+2, ((stac+2)*steps)+1]).astype(int)
    c = np.zeros([na.shape[1]])
    ub = []
    lb = []
    expArcs = []

    # Connection between collector nodes
    na[0, 0] = -1
    na[1, 0] = 1
    c[0] = -1
    lb.append(None)
    ub.append(None)
    expArcs.append(["max flow loop"])

    # Arcs from collector source node
    for j in range(steps):
        na[0, 1+j] = 1
        na[startNode+2+j*stnc, 1+j] = -1
        c[1+j] = 0
        lb.append(None)
        ub.append(None)
        expArcs.append(["source", j])

    # Arcs to collector sink node
    for j in range(steps):
        na[1, 1+steps+j] = -1
        na[endNode+2+j*stnc, 1+steps+j] = 1
        c[1+steps+j] = 0
        lb.append(None)
        ub.append(None)
        expArcs.append(["sink", j])

    # For each arc, update the two corresponding entries in the node-arc matrix
    arcOff = 2*steps+1  # Offset to first non collector arc
    nodeOff = 2         # Offset to first non collector node
    for j in range(steps):
        for i in range(stac):
            # If the destination node is outside of the graph, ignore it
            if j+tnn[arcs[i, 0], arcs[i, 1]] < steps:
                # Fill the Node-Arc matrix
                na[nodeOff + arcs[i, 0] + j*stnc,                               arcOff + i + j*stac] = 1
                na[nodeOff + arcs[i, 1] + (j+tnn[arcs[i, 0], arcs[i, 1]])*stnc, arcOff + i + j*stac] = -1
                c[arcOff + i + j*stnc] = cnn[arcs[i, 0], arcs[i, 1]]

                # Complete the bound lists
                bound = lbnn[arcs[i, 0], arcs[i, 1]]
                if bound == np.inf:
                    bound = None
                lb.append(bound)
                bound = ubnn[arcs[i, 0], arcs[i, 1]]
                if bound == np.inf:
                    bound = None
                ub.append(bound)

                # Expanded arcs
                expArcs.append([arcs[i, 0], arcs[i, 1], j])



    # Delete empty rows and columns, create a bound tuple using the lower and upper bound lists and return
    colDeleteMask = np.where(~na.any(axis=0))[0]
    rowDeleteMask = np.where(~na.any(axis=1))[0]
    na = np.delete(na, colDeleteMask, axis=1)
    na = np.delete(na, rowDeleteMask, axis=0)
    c = np.delete(c, colDeleteMask, axis=0)
    bounds = tuple(zip(lb, ub))
    return na, c, bounds, expArcs


# Time required to go from node 'row' to node 'column'
tnn = np.array([
        [0, 1, 3, 0],
        [0, 0, 2, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ])

# Capacity upper bound to go from node 'row' to node 'column'
ubnn = np.array([
        [0, 5, 10, 0],
        [0, 0, 6, 3],
        [0, 0, 0, 3],
        [0, 0, 0, 0]
    ])

# Cost to go from node 'row' to node 'column'
cnn = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

# Capacity lower bound to go from node 'row' to node 'column'
lbnn = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])

# Node-Arc Matrix and Cost Matrix computation
A, C, bounds, arcs = expandInTime(cnn, ubnn, lbnn, 0, 3, tnn, 6)

# Net flow for each node (>0 is a source / <0 is a sink)
B = np.zeros([A.shape[0]])



# Solve the linear program using interior-point
res = linprog(C, A_eq=A, b_eq=B, bounds=bounds, method='interior-point')
print(bounds)

# Print Result
print("Solver: Interior-Point")
print("Raw Solution: ", res.x)
print("Transported Units:")
for i in range(res.x.shape[0]):
    print(arcs[i], " -> ", res.x[i])
print("Objective Function Value: ", res.fun)




# Solve the linear program using simplex
res = linprog(C, A_eq=A, b_eq=B, bounds=bounds, method='simplex')

# Print Result
print("\n\n\nSolver: Simplex")
print("Raw Solution: ", res.x)
print("Transported Units:")
for i in range(res.x.shape[0]):
    print(arcs[i], " -> ", res.x[i])
print("Objective Function Value: ", res.fun)