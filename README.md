MinimumGraphPath: a very simple Python implementation of the Dijkstra and A* algorithms.

Nodes are represented by coordinates in the 2D plane, and distance is calculated based on the Euclidean distance.

If networkx is not in your installed packages, run `pip install networkx`.

---------------------
HOW TO DEFINE A GRAPH
---------------------

First, initialise a class `Graph()`. The functions `addNode` and `addEdge` are inherited from the `nx.Graph()` class. `search_algorithm` is the algorithm you want to use for finding the minimum-distance path from a "start" to an "end" node, and must be either Dijkstra (`"dijkstra"`) or A-star (`"a*"`). `search_metric` is the distance metric, which is automatically set to `"euclidean"` if unspecified. There is also support for the L1-norm (`"l1"`) and custom user-specified metrics, set by weights specified in the `addNode` function (default = 1).

A simple example of code is given under `if __name__ == "__main__"` for illustrative purposes.
