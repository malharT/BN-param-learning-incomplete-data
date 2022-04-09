# describes a cluster graph
# could be a tree/forest, but it might not be
class clustergraph:
    def __init__(self):
        self._clusters = []  # list
        self._adj = []  # list of set of indexes into _clusters

    # vars is a set of variables
    # returns an identifier for this cluster (an int)
    def addcluster(self,vars):
        i = len(self._clusters)
        self._clusters.append(vars)
        self._adj.append(set())
        return i

    # adds an edge between cluster1 and cluster2
    # (cluster1 and cluster2 are identifiers, as returned by addcluster)
    def addedge(self,cluster1,cluster2):
        self._adj[cluster1].add(cluster2)
        self._adj[cluster2].add(cluster1)

    # list of the clusters (each element in the list is a set)
    @property
    def clusters(self):
        return self._clusters

    # returns a set of the adjacent cluster identifiers
    def adj(self,i):
        return self._adj[i]


