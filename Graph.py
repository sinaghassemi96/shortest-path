# a class to input the graph and find the shortest path via Dijkstra and Incremental Dijkstra
import heapq

import matplotlib
import networkx as nx
import dask.dataframe as dd
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def impact_function(initial_travel_time, ratio_volume_capacity):
    # Example impact function: Multiply initial travel time by a function of the ratio of volume over capacity
    # return initial_travel_time * (1 + ratio_volume_capacity ** 2)
    return initial_travel_time * (1 + 0.3 * ratio_volume_capacity ** 2)


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

    def __init__(self, pos=None):
        self.graph = nx.Graph()
        self.pos = pos
        self.sink_node = None
        self.link_capacity = 40
        self.src = None

    # Function to add an edge to the graph
    def add_edge(self, u, v, weight, capacity, free_flow_time):
        self.graph.add_edge(u, v, weight=weight, capacity=capacity, free_flow_time=free_flow_time, active=True)

    # Function to read graph properties from a file
    def read_graph_from_file(self, filename):
        try:
            df = dd.read_csv(filename)
            for _, row in df.iterrows():
                self.add_edge(row['FromNodeId'], row['ToNodeId'], row['TravelTime'], self.link_capacity,
                              row['TravelTime'])
            print("network created successfully.")
        except Exception as e:
            print(f"Error reading graph from file: {e}")

    # Function that implements Dijkstra's algorithm
    # def dijkstra(self, src):
    #     V = len(self.graph.nodes)
    #     dist = [1e7] * V
    #     dist[src] = 0
    #     spt_set = [False] * V
    #     path = [-1] * V
    #
    #     for _ in range(V):
    #         u = min_distance(dist, spt_set)
    #         spt_set[u] = True
    #
    #         for v in range(V):
    #             if (
    #                     self.graph.has_edge(u, v)
    #                     and self.graph[u][v]['active']
    #                     and not spt_set[v]
    #                     and dist[v] > dist[u] + self.graph[u][v]['weight']
    #             ):
    #                 dist[v] = dist[u] + self.graph[u][v]['weight']
    #                 path[v] = u
    #
    #     return dist, path

    def dijkstra(self, src):
        V = len(self.graph.nodes)
        dist = [float('inf')] * V
        dist[src] = 0
        path = [-1] * V
        priority_queue = [(0, src)]

        while priority_queue:
            current_dist, u = heapq.heappop(priority_queue)

            if current_dist > dist[u]:
                continue  # Skip outdated entry

            for v in self.graph.neighbors(u):
                if (
                        self.graph.has_edge(u, v)
                        and self.graph[u][v]['active']
                        and dist[v] > dist[u] + self.graph[u][v]['weight']
                ):
                    dist[v] = dist[u] + self.graph[u][v]['weight']
                    path[v] = u
                    heapq.heappush(priority_queue, (int(dist[v]), v))

        print("Length of dist:", len(dist))  # Debugging line
        print("Length of path:", len(path))  # Debugging line

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
                # src_node = 0
                dist, path = self.dijkstra(self.src)
                min_index = dist.index(min(dist))
                shortest_path = construct_path(path, self.src, min_index)
                shortest_path_edges = list(zip(shortest_path, shortest_path[1:]))
                nx.draw_networkx_edges(self.graph, pos, edgelist=shortest_path_edges, edge_color='r', width=2)
                # print_solution(dist, path)
                return shortest_path, dist[min_index]

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

    def update_edge_activity(self, u, v, active):
        if self.graph.has_edge(u, v):
            self.graph[u][v]['active'] = active
        else:
            print(f"Edge ({u}, {v}) does not exist in the graph.")

    def get_solution_to_destination(self, destination):
        if self.sink_node is None:
            print("Sink node not set. Use set_sink_node() to set the sink node.")
            return None, None

        dist, path = self.dijkstra(self.src)

        if path[destination] == -1:
            print(f"No path found to destination {destination}.")
            return None, None

        shortest_path = construct_path(path, self.src, destination)
        cost_to_destination = dist[destination]

        return shortest_path, cost_to_destination

    def incremental_assignment(self, demand, increment, use_congested_links=False):
        src_node = 0  # Source node
        dest_node = self.sink_node
        dist, path = self.dijkstra(src_node)
        cost_to_destination = dist[dest_node]
        if self.sink_node is None:
            print("Sink node not set. Use set_sink_node() to set the sink node.")
            return

        num_steps = demand // increment
        current_vehicles = 0

        for step in range(1, num_steps + 1):
            if current_vehicles < demand and path[dest_node] != -1:
                # Update the volume of each link on the shortest path
                shortest_path = construct_path_incremental_assignment(path, src_node, dest_node)
                for u, v in zip(shortest_path, shortest_path[1:]):

                    # Activate the link for the current step
                    # self.update_edge_activity(u, v, True)
                    current_volume = self.graph[u][v].get('volume', 0)
                    # new_volume = min(current_volume + increment, self.graph[u][v]['capacity'])
                    new_volume = current_volume + increment
                    self.graph[u][v]['volume'] = new_volume

                    if not use_congested_links and self.graph[u][v].get('volume', 0) >= self.graph[u][v]['capacity']:
                        # Deactivate congested link
                        self.update_edge_activity(u, v, False)
                        continue  # Skip congested link

                # Update the weights based on the new volumes
                for u, v in self.graph.edges():
                    if self.graph[u][v]['active']:
                        capacity = self.graph[u][v]['capacity']
                        volume = self.graph[u][v].get('volume', 0)
                        new_weight = impact_function(self.graph[u][v]['free_flow_time'], volume / capacity)
                        self.update_edge_weight(u, v, new_weight)

                # Calculate the new shortest path and its cost
                dist, path = self.dijkstra(src_node)
                min_index = dist.index(min(dist))
                shortest_path = construct_path(path, src_node, dest_node)

                # Print the outcome path and cost
                print(f"Step {step}: Shortest path at this step: {shortest_path}")
                print(f"Cost to destination at this step: {cost_to_destination}")

                # Draw the graph with the updated travel times
                self.draw_graph(pos=self.pos)

                # Increment the number of vehicles
                cost_to_destination = dist[dest_node]
                current_vehicles += increment

                print(f"Step {step}: Incremental assignment completed. Total vehicles: {current_vehicles}")

        print("Incremental assignment completed for all steps.")

    def set_src_node(self, src_node):
        self.src = src_node


