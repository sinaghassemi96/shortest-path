# a class to input the graph and find the shortest path via Dijkstra and Incremental Dijkstra
import matplotlib
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def impact_function(initial_travel_time, ratio_volume_capacity):
    # Example impact function: Multiply initial travel time by a function of the ratio of volume over capacity
    # return initial_travel_time * (1 + ratio_volume_capacity ** 2)
    return initial_travel_time * (1 + 2 * ratio_volume_capacity ** 2)


def min_distance(dist, spt_set):
    min_val = 1e7
    min_index = -1

    for v in range(len(dist)):
        if dist[v] < min_val and not spt_set[v]:
            min_val = dist[v]
            min_index = v

    return min_index


def construct_path(path, src, dest):
    result = []
    current = dest
    while current != -1:
        result.insert(0, current)
        current = path[current]
    return " -> ".join(map(str, result))


def construct_path_incremental_assignment(path, src, dest):
    result = []
    current = dest
    while current != -1:
        result.insert(0, current)
        current = path[current]
    return result


def print_solution(dist, path):
    print("Vertex \t Cost \t Path")
    for node in range(len(dist)):
        print(f"\t{node}\t\t{dist[node]}\t\t{construct_path(path, 0, node)}")


class Graph:

    def __init__(self, pos):
        self.graph = nx.Graph()
        self.pos = pos
        self.sink_node = None
        self.link_capacity = 40

    # Function to add an edge to the graph
    def add_edge(self, u, v, weight, capacity, free_flow_time):
        self.graph.add_edge(u, v, weight=weight, capacity=capacity, free_flow_time=free_flow_time)

    # Function to read graph properties from a file
    def read_graph_from_file(self, filename):
        try:
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                self.add_edge(row['FromNodeId'], row['ToNodeId'], row['TravelTime'], self.link_capacity, row['TravelTime'])
        except Exception as e:
            print(f"Error reading graph from file: {e}")

    # Function that implements Dijkstra's algorithm
    def dijkstra(self, src):
        V = len(self.graph.nodes)
        dist = [1e7] * V
        dist[src] = 0
        spt_set = [False] * V
        path = [-1] * V

        for _ in range(V):
            u = min_distance(dist, spt_set)
            spt_set[u] = True

            for v in range(V):
                if (
                    self.graph.has_edge(u, v)
                    and not spt_set[v]
                    and dist[v] > dist[u] + self.graph[u][v]['weight']
                ):
                    dist[v] = dist[u] + self.graph[u][v]['weight']
                    path[v] = u

        return dist, path

    # Function to get the set of vertices from the graph
    def get_vertices(self):
        return set(self.graph.nodes)

    # Function to draw the graph
    def draw_graph(self, filename=None, pos=None):
        if self.sink_node is not None:
            pos = pos if pos is not None else nx.shell_layout(self.graph)
            labels = nx.get_edge_attributes(self.graph, 'weight')
            nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8)
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, label_pos=0.4)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.sink_node], node_color='red', node_size=700)

            # Calculate the shortest path
            if self.sink_node is not None:
                src_node = 0
                dist, path = self.dijkstra(src_node)
                min_index = dist.index(min(dist))
                shortest_path = construct_path(path, src_node, min_index)
                shortest_path_edges = list(zip(shortest_path, shortest_path[1:]))
                nx.draw_networkx_edges(self.graph, pos, edgelist=shortest_path_edges, edge_color='r', width=2)
                print_solution(dist, path)

            plt.title("Graph with Travel Times")

            if filename is not None:
                plt.savefig(filename)
                print(f"Plot saved as {filename}")
            else:
                plt.show()
        else:
            print("Sink node not set. Use set_sink_node() to set the sink node.")

    def set_sink_node(self, sink_node):
        self.sink_node = sink_node

    def update_edge_weight(self, u, v, new_weight):
        if self.graph.has_edge(u, v):
            self.graph[u][v]['weight'] = new_weight
        else:
            print(f"Edge ({u}, {v}) does not exist in the graph.")

    def incremental_assignment(self, demand, increment):
        src_node = 0  # Source node
        dist, path = self.dijkstra(src_node)

        if self.sink_node is None:
            print("Sink node not set. Use set_sink_node() to set the sink node.")
            return

        dest_node = self.sink_node

        current_vehicles = 0
        while current_vehicles < demand and path[dest_node] != -1:
            # Update the volume of each link on the shortest path
            shortest_path = construct_path_incremental_assignment(path, src_node, dest_node)
            for u, v in zip(shortest_path, shortest_path[1:]):
                current_volume = self.graph[u][v].get('volume', 0)
                new_volume = min(current_volume + increment, self.graph[u][v]['capacity'])
                self.graph[u][v]['volume'] = new_volume

            # Update the weights based on the new volumes
            for u, v in self.graph.edges():
                capacity = self.graph[u][v]['capacity']
                volume = self.graph[u][v].get('volume', 0)
                new_weight = self.graph[u][v]['free_flow_time'] * (1 + 2 * (volume / capacity) ** 2)
                self.update_edge_weight(u, v, new_weight)

            # Draw the graph with the updated travel times
            self.draw_graph(f"graph_plot_step_{current_vehicles + increment}.png", pos=self.pos)

            # Increment the number of vehicles
            current_vehicles += increment

        print(f"Incremental assignment completed. Total vehicles: {current_vehicles}")

