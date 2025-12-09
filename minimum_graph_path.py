import networkx as nx
import numpy as np
import heapq

# set of functions for finding the minimum path
# in a two-dimensional user-defined networkx graph
def distance(coord1: tuple, coord2: tuple):
    '''Calculates the Euclidean distance between two coordinates'''
    dx = coord1[0] - coord2[0]
    dy = coord1[1] - coord2[1]
    return np.sqrt(dx ** 2 + dy ** 2)

def findMinPath(graph: nx.Graph(), namedDict: dict, start: str, end: str, algorithm: str):
    '''Calculates the minimum-distance path through a graph'''
    if algorithm == "dijkstra":
        heuristic = 0
    elif algorithm == "a*":
        heuristic = 1
    else:
        raise ValueError("Algorithm must be either Dijkstra or A*!")
    
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
            newdist = current_dist + distance(node, neighbor) + heuristic * distance(neighbor, end_coord)
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

    def addEdge(self, coord1: tuple, coord2: tuple):
        if coord1 not in self.graph.nodes or coord2 not in self.graph.nodes:
            raise ValueError("Either coordinates not in Graph!")
        self.graph.add_edge(coord1, coord2, weight = distance(coord1, coord2))

    def findMinPath(self):
        minDist, minPath = findMinPath(self.graph, self.dict, "start", "end", "dijkstra")
        print("The minimum distance found is " + f"{minDist:.2f}")
        print("And the minimum path is {}".format(minPath))

    def drawGraph(self):
        nx.draw(self.graph)

if __name__ == "__main__":
    algorithm = "dijkstra"

    Graph = Graph()
    Graph.addNode((0, 0), "start")
    Graph.addNode((0, 1))
    Graph.addNode((1, 1), "end")
    Graph.addEdge((0, 0), (0, 1))
    Graph.addEdge((0, 1), (1, 1))
    Graph.addEdge((0, 0), (1, 1))
    Graph.findMinPath()
