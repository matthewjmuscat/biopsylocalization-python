import numpy as np
from sklearn.neighbors import KDTree
import scipy

class Node(object):
    """
    This class defines the Node objects for constructing kd trees.
    """
    def __init__(self):
        self.left = None
        self.right = None
        self.dim = None
        self.val = None


def build_kdtree(vectors, node, dimension, max_dimension):
    print('dim = ',dimension)
    vals = sorted(vectors, key=lambda v: v[dimension])
    print('vals = ',vals)
    if len(vals) == 1:
        node.val = vals[0]
        return node
    if len(vals) == 2:
        node.val = vals[1]
        node.left = Node()
        node.left.val = vals[0]
        return node
    mid = int((len(vals) - 1) / 2)
    print('mid = ',mid)
    print('midval = ', vals[mid])
    node.val = vals[mid]
    node.left = build_kdtree(vals[:mid], Node(), (dimension+1) % max_dimension, max_dimension)
    node.right = build_kdtree(vals[mid+1:], Node(), (dimension+1) % max_dimension, max_dimension)
    return node


def check_leaf(node):
    if node.left == None and node.right == None:
        leaf_status = True
    else:
        leaf_status = False
    return leaf_status

def main():
    #print('This is the kd data tree programme, it is not meant to be run directly.')

    vecs = [[5,4],[2,3],[8,1],[9,6],[7,2],[4,7]]
    sortedvecs = sorted(vecs, key=lambda v: v[0])
    print(sortedvecs)
    root = Node()
    root1 = build_kdtree(vecs, root, 0, 2)
    print(root1)
    print('hello')
    query_point = [1,2]

    active_node = root1
    leaf = check_leaf(active_node)
    active_dimension = 0
    max_dimension = 2
    while leaf == False:
        if query_point[active_dimension] <= active_node.val[active_dimension]:
            active_node = active_node.left
        elif query_point[active_dimension] > active_node.val[active_dimension]:
            active_node = active_node.right
        leaf = check_leaf(active_node)
        active_dimension = (active_dimension+1) % max_dimension
    
    best_dist_estimate = np.linalg.norm(np.array(active_node.val) - np.array(query_point))

    print('hello')


    
    
    tree = KDTree(vecs, leaf_size=2)
    tree_data, index, tree_nodes, node_bounds = tree.get_arrays()
    tree_nodes
    treescipy = scipy.spatial.KDTree(vecs)
    nn = treescipy.query(query_point)
    nearest_neighbour = treescipy.data[nn[1]]
    print('hello')
if __name__ == '__main__':    
    main()