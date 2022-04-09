from factor import *
import copy

# describes a distribution over a cluster graph
# as parameterized by betas (factors over the clusters)
# and mus (factors over the sep-sets)
class clusterdist:
    # U is the clustergraph on which this distribution is based
    # alpha is a mapping from factors to cluster indexes in U
    def __init__(self,U,alpha):
        self._U = copy.copy(U) # so that the original can change
        self._beta = [] # list of local factors, _beta[i] matches cluster i in U
        self._mu = {} # dictionary mapping (i,j) to factor for edge i-j
                # note that (3,5) and (5,3) are the same, so we only
                # keep (3,5) -- because 3<5
        self._initializegraph(alpha)

    # assumes that sep-set is intersection of scopes of either side
    # (this is true for clique trees, but not necessarily true for other cluser graphs)
    def _initializegraph(self,alpha):
        for i in range(len(self._U.clusters)):
            self._beta.append(discretefactor(self._U.clusters[i],1.0))
        for i in range(len(self._U.clusters)):
            for j in self._U.adj(i):
                if i<j:
                    self._mu[(i,j)] = discretefactor(self._U.clusters[i] & self._U.clusters[j],1.0)
        for f,i in alpha.items():
            self._beta[i] = self._beta[i] * f

    @property
    def graph(self):
        return self._U

    # after calibration, should be the marginal over the cluster i
    def getbeta(self,i):
        return self._beta[i]

    # after calibration, should be the marginal over the sep-set btwn i&j
    def getmu(self,i,j):
        return self._mu[(i,j)] if i<j else self._mu[(j,i)]

    def get_neighbors(self, i):
        return self.graph.adj(i)

    # calibrates, assuming that the clustergraph is a *forest*
    # (not necessarily a tree, despite the name -- it might be
    #  multiple disconnected trees -- this happens often once
    #  evidence is introduced)
    def bu_message(self, i, j):
        if i > j:
            mu = self.getmu(j, i)
            edge = (j, i)
        else:
            mu = self.getmu(i, j)
            edge = (i, j)
        beta_i = self.getbeta(i)
        sigma_i_to_j = beta_i.marginalize(beta_i.scope - mu.scope)
        beta_j = self.getbeta(j)
        beta_j = beta_j *(sigma_i_to_j/mu)
        mu = sigma_i_to_j

        self._beta[j] =  beta_j
        self._mu[edge] = mu

    def treecalibrate(self):
        non_caliberated_clusters = set(range(len(self.graph.clusters)))
        while non_caliberated_clusters:
            # Getting a random cluster and in a way its tree to start with
            random_cluster = non_caliberated_clusters.pop()
            non_caliberated_clusters.add(random_cluster)

            # Considering the selected random cluster as root and
            # caliberating for the messages flowing upstream using BFS
            stack = [(random_cluster, None)]
            while stack:
                # Presence of string 'p' on top of stack indicates
                # that all the children of node just after 'p' are 
                # processed for upward message.
                if stack[-1] == 'p':
                    # Removing the indicator 'p'
                    stack.pop()
                    # Since all incoming messages from children are received
                    # adjusting the beta and mu according to the received
                    # messages
                    current_node, parent = stack.pop()
                    for child in self.get_neighbors(current_node) - {parent}:
                        self.bu_message(child, current_node)
                else:
                    # If the node has children adding all children the to the
                    # stack to process them first over the indicator 'p'.
                    current_node, parent = stack[-1]
                    children = self.get_neighbors(current_node) - {parent}
                    if children:
                        stack.append("p")
                        for child in children:
                            stack.append((child, current_node))
                    # If there are no children the node is ready to send
                    # message upwards no need to process it.
                    else:
                        stack.pop()
            # Considering the selected random cluster as root and
            # caliberating for the messages flowing downstream using BFS
            stack = [(random_cluster, None)]
            while stack:
                # For each node all the edges connected to children are 
                current_node, parent = stack.pop()
                non_caliberated_clusters.remove(current_node)
                children = self.get_neighbors(current_node) - {parent}
                for child in children:
                    self.bu_message(current_node, child)
                    stack.append((child, current_node))
