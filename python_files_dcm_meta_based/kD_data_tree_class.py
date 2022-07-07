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
    vals = sorted(vectors, key=lambda v: v[dimension])
    if len(vals) == 1:
        node.val = vals[0]
        return node
    if len(vals) == 2:
        node.val = vals[1]
        node.left = Node()
        node.left.val = vals[0]
        return node
    mid = int((len(vals) - 1) / 2)
    node.val = vals[mid]
    node.left = build_kdtree(vals[:mid], Node(), (dimension+1) % max_dimension, max_dimension)
    node.right = build_kdtree(vals[mid+1:], Node(), (dimension+1) % max_dimension, max_dimension)
    return node

def main():
    #print('This is the kd data tree programme, it is not meant to be run directly.')

    vecs = [[5,4],[2,3],[8,1],[9,6],[7,2],[4,7]]
    sortedvecs = sorted(vecs, key=lambda v: v[0])
    print(sortedvecs)
    root = Node()
    root1 = build_kdtree(vecs, root, 0, 1)
    print(root1)
    print('hello')


if __name__ == '__main__':    
    main()