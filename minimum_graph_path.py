import networkx as nx
import numpy as np
import heapq

# set of functions for finding the minimum path
# in a two-dimensional user-defined networkx graph
def distance(graph: nx.Graph(), coord1: tuple, coord2: tuple, metric: str):
    '''Calculates the distance between two coordinates'''
    if metric == "euclidean":
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        return np.sqrt(dx ** 2 + dy ** 2)
    elif metric == "l1":
        dx = coord1[0] - coord2[0]
        dy = coord1[1] - coord2[1]
        return np.abs(dx) + np.abs(dy)
    elif metric == "custom":
        assigned_weight = graph.get_edge_data(coord1, coord2)
        if assigned_weight is None:
            return 0
        else:
            return assigned_weight['weight']
    else:
        raise ValueError("Metric must be either Euclidean, L1 or a custom user-specified distance metric!")

def findMinPath(graph: nx.Graph(), namedDict: dict, start: str, end: str, algorithm: str, metric: str):
    '''Calculates the minimum-distance path through a graph'''
    if algorithm == "dijkstra":
        heuristic = 0
    elif algorithm == "a*":
        heuristic = 1
    else:
        raise ValueError("Algorithm must be either Dijkstra or A*!")

    if metric not in ["euclidean", "l1", "custom"]:
        raise ValueError("Metric must be either Euclidean, L1 or a custom user-specified distance metric!")
    
    if start == None or end == None:
        raise ValueError("Start and end must be specified!")

    start_coord = namedDict[start]
    end_coord   = namedDict[end]

    # `dist` is a dictionary that traces all distance histories
    # starts at infinity for all nodes
    dist = {node: float("inf") for node in graph.nodes}
    prev = {node: None for node in graph.nodes}
    dist[start_coord] = 0

    pq = [(0, start_coord)]  # (distance, node)
    visited = set()

    while pq:
        current_dist, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        if node == end_coord:
            break

        for neighbor in graph.neighbors(node):
            newdist = current_dist + distance(graph, node, neighbor, metric) + heuristic * distance(graph, neighbor, end_coord, metric)
            if newdist < dist[neighbor]:
                dist[neighbor] = newdist
                prev[neighbor] = node
                heapq.heappush(pq, (newdist, neighbor))

    # reconstruct path
    path = []
    node = end_coord
    while node is not None:
        path.append(node)
        node = prev[node]

    path.reverse()
    return dist[end_coord], path

# graph class for more user-friendly graph definition
class Graph:
    def __init__(self):
        self.graph = nx.Graph()
        self.dict = dict()

    def addNode(self, coords: tuple, name = None):
        self.graph.add_node(coords)
        if name != None:
            self.dict[name] = coords

    def addEdge(self, coord1: tuple, coord2: tuple, weight = None):
        if coord1 not in self.graph.nodes or coord2 not in self.graph.nodes:
            raise ValueError("Either coordinates not in Graph!")
        if weight is None:
            w = distance(self.graph, coord1, coord2, metric = "euclidean")
        else:
            w = weight
        self.graph.add_edge(coord1, coord2, weight = w)

    def findMinPath(self, algorithm: str, metric = None):
        if metric is None:
            m = "euclidean"
        else:
            m = metric
        minDist, minPath = findMinPath(self.graph, self.dict, "start", "end", algorithm, metric = m)
        print("The minimum distance found is " + f"{minDist:.2f}")
        print("And the minimum path is {}".format(minPath))

    def drawGraph(self):
        nx.draw(self.graph)

if __name__ == "__main__":
    search_algorithm = "dijkstra"
    search_metric = "euclidean"

    Graph = Graph()
    Graph.addNode((0, 0), "start")
    Graph.addNode((0, 1))
    Graph.addNode((1, 1), "end")
    Graph.addEdge((0, 0), (0, 1), 1)
    Graph.addEdge((0, 1), (1, 1), 1)
    Graph.addEdge((0, 0), (1, 1), 3)

    Graph.findMinPath(algorithm = search_algorithm, metric = search_metric)
