import numpy as np
from scipy.optimize import linprog

# Converts a two Cost Node-Node matrix to the corresponding Node-Arc and Cost matrices
def cnn2nac (cnn, ubNodes):
    # Find the existing arcs
    arcs = np.argwhere(cnn)

    # Create the node-arc matrix and cost matrix
    na_eq = np.zeros([cnn.shape[0], arcs.shape[0]]).astype(int)
    na_ub = np.zeros([cnn.shape[0], arcs.shape[0]]).astype(int)
    c = np.zeros([arcs.shape[0]])

    # For each arc, update the two corresponding entries in the node-arc matrix
    for i in range(arcs.shape[0]):
        if np.any(ubNodes == arcs[i, 0]):
            na_ub[arcs[i, 0], i] = 1
        else:
            na_eq[arcs[i, 0], i] = 1

        if np.any(ubNodes == arcs[i, 1]):
            na_ub[arcs[i, 1], i] = -1
        else:
            na_eq[arcs[i, 1], i] = -1

        c[i] = cnn[arcs[i, 0], arcs[i, 1]]

    # Return
    return na_eq, na_ub, c, arcs



# Cost to go from node 'row' to node 'column' for product A
# Nodes: P1 P2 P3 St1 St2 S1 S2 S3
cnna = np.array([
        [0, 0, 0, 100, 100, 0, 0, 0],
        [0, 0, 0, 150, 150, 0, 0, 0],
        [0, 0, 0, 200, 200, 0, 0, 0],
        [0, 0, 0, 0, 0, 100, 150, 200],
        [0, 0, 0, 0, 0, 100, 150, 200],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

# Cost to go from node 'row' to node 'column' for product B
# Nodes: P1 P2 P3 St1 St2 S1 S2 S3
cnnb = np.array([
        [0, 0, 0, 200, 200, 0, 0, 0],
        [0, 0, 0, 150, 150, 0, 0, 0],
        [0, 0, 0, 100, 100, 0, 0, 0],
        [0, 0, 0, 0, 0, 200, 150, 100],
        [0, 0, 0, 0, 0, 200, 150, 100],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])

# Cost to go from node 'row' to node 'column' for products A & B
# Nodes: P1a P2a P3a St1a St2a S1a S2a S3a P1b P2b P3b St1b St2b S1b S2b S3b
cnnab = np.concatenate((np.concatenate((cnna, np.zeros(cnna.shape)), axis=1), np.concatenate((np.zeros(cnna.shape), cnnb), axis=1)), axis=0)

# Node-Arc Matrix and Cost Matrix computation
Aeqa, Auba, Ca, arcsa = cnn2nac(cnna, [0, 1, 2])
Aeqb, Aubb, Cb, arcsb = cnn2nac(cnnb, [0, 1, 2])
Aeqab, Aubab, Cab, arcsab = cnn2nac(cnnab, [0, 1, 2, 8, 9, 10])

# Net flow for each node (>0 is a source / <0 is a sink)
Beqa = np.array([0, 0, 0, 0, 0, -30, -10, -20])
Beqb = np.array([0, 0, 0, 0, 0, -40, -20, -20])
Buba = np.array([20, 10, 30, 0, 0, 0, 0, 0])
Bubb = np.array([30, 40, 10, 0, 0, 0, 0, 0])
Beqab = np.concatenate((Beqa, Beqb), axis=0)
Bubab = np.concatenate((Buba, Bubb), axis=0)

# Decision variables bounds
bounds = tuple([0, None] for arcs in range(Ca.shape[0]))
boundsab = tuple([0, None] for arcs in range(2*Ca.shape[0]))


# Solve the linear program using simplex
res = linprog(Ca, A_eq=Aeqa, b_eq=Beqa, A_ub=Auba, b_ub=Buba, bounds=bounds, method='simplex')
#res = linprog(Ca, A_eq=Aeqa+Auba, b_eq=Beqa+Buba, bounds=bounds, method='simplex')

# Print Result
print("Products: A")
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Transported Units:")
for i in range(res.x.shape[0]):
    print(arcsa[i]+1, " -> ", res.x[i])
print("Objective Function Value: ", res.fun)

# Solve the linear program using simplex
res = linprog(Cb, A_eq=Aeqb, b_eq=Beqb, A_ub=Aubb, b_ub=Bubb, bounds=bounds, method='simplex')

# Print Result
print("\n\n\nProducts: B")
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Transported Units:")
for i in range(res.x.shape[0]):
    print(arcsb[i]+1, " -> ", res.x[i])
print("Objective Function Value: ", res.fun)

# Solve the linear program using simplex
res = linprog(Cab, A_eq=Aeqab, b_eq=Beqab, A_ub=Aubab, b_ub=Bubab, bounds=boundsab, method='simplex')

# Print Result
print("\n\n\nProducts: A&B")
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Transported Units:")
for i in range(res.x.shape[0]):
    print(arcsab[i]+1, " -> ", res.x[i])
print("Objective Function Value: ", res.fun)

print("\n\n\nThe obtained result is the same because there is no relation between nodes belonging to the first product "
      "and nodes belonging to the second one, this shields two separate graphs, which can be studied independently")
