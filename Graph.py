# a class to input the graph and find the shortest path via Dijkstra and Incremental Dijkstra
# implemented by Sina Ghassemi
# import necessary tools or libraries
import heapq
import matplotlib
import networkx as nx
import dask.dataframe as dd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# Impact function is used to update the weight (cost) of the links during the inc. assignment.
# Example impact function: Multiply initial travel time by a function of the ratio of volume over capacity
def impact_function(initial_travel_time, ratio_volume_capacity):
    return initial_travel_time * (1 + 2 * ratio_volume_capacity ** 2)


# This function builds the shortest path to destination and returns the sequence as an array
def construct_path(path, src, dest):
    result = []
    current = dest
    while current != -1:
        result.insert(src, current)
        current = path[current]
    return result


# This function is used once the incremental assignment is completed to show the total network's travel time
def calculate_volume_times_travel_time(graph):
    result = 0.0

    for u, v in graph.edges():
        volume = graph[u][v].get('volume', 0)
        travel_time = graph[u][v]['weight']

        # Multiply volume by travel time and update the result
        result += volume * travel_time

    return result


# Class contains necessary implementations to establish a shortest path search.
class Graph:

    # Initializes the class by building a networkx graph object
    # Defines the source node, the sink node, and link's capacity
    def __init__(self, pos=None):
        self.graph = nx.Graph()
        self.pos = pos
        self.sink_node = None
        self.link_capacity = None
        self.src = None

    # Function to add an edge to an nx graph
    def add_edge(self, u, v, weight, capacity, free_flow_time):
        self.graph.add_edge(u, v, weight=weight, capacity=capacity, free_flow_time=free_flow_time, active=True)

    # Function to read graph properties from a file
    # The input directory must be provided
    def read_graph_from_file(self, filename):
        try:
            df = dd.read_csv(filename)
            for _, row in df.iterrows():
                # assumes link's capacities are constant
                self.add_edge(row['FromNodeId'], row['ToNodeId'], row['TravelTime'], self.link_capacity,
                              row['TravelTime'])
            print("network created successfully.")
            print("-" * 100)
        except Exception as e:
            print(f"Error reading graph from file: {e}")

    # Function to calculate the shortest path using Dijkstra and minimum heap data structure
    def dijkstra(self, src):
        # Vector to hold the node's label. It is defined as big as the biggest label plus one to skip
        # the difference between list indices and normal numeration
        vector = int(max(self.graph.nodes)) + 1

        # Array to hold the distance from the src node
        dist = [float('inf')] * vector

        # Set the distance from the src equal to zero
        dist[src] = 0

        # construct a path equal to the size of the nodes
        path = [-1] * vector

        # insert the first element of the priority queue
        priority_queue = [(0, src)]

        # main loop to find and update the shortest path
        while priority_queue:
            # getting the current distance to u
            current_dist, u = heapq.heappop(priority_queue)

            # comparing the distance updated in the loop with the distance defined before (inf)
            if current_dist > dist[u]:
                continue  # Skip outdated entry

            # update the current dist and check its existence, availability, and positivity of the cost
            # push new value into the heap map and search until it fails finding smaller distance
            for v in self.graph.neighbors(u):
                if (
                        self.graph.has_edge(u, v)
                        and self.graph[u][v]['active']
                        and dist[v] > dist[u] + self.graph[u][v]['weight']
                ):
                    dist[v] = dist[u] + self.graph[u][v]['weight']
                    path[v] = u
                    heapq.heappush(priority_queue, (int(dist[v]), v))

        return dist, path

    # Function to get the set of vertices from the graph
    def get_vertices(self):
        return set(self.graph.nodes)

    # Function to draw the graph
    def draw_graph(self, path, filename=None, pos=None):
        plt.clf()
        if self.sink_node is not None:
            pos = pos if pos is not None else nx.shell_layout(self.graph)
            labels = nx.get_edge_attributes(self.graph, 'weight')
            nx.draw(self.graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8)
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels, label_pos=0.4)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[self.sink_node], node_color='red', node_size=700)

            # Calculate the shortest path
            if self.sink_node is not None:
                shortest_path = construct_path(path, self.src, self.sink_node)
                shortest_path_edges = list(zip(shortest_path, shortest_path[1:]))
                nx.draw_networkx_edges(self.graph, pos, edgelist=shortest_path_edges, edge_color='r', width=2)
                # return shortest_path, dist[self.sink_node]

            plt.title("Graph with Travel Times")

            if filename is not None:
                plt.savefig(filename)
                print(f"Plot saved as {filename}")
            else:
                plt.show()
        else:
            print("Sink node not set. Use set_sink_node() to set the sink node.")

    # Function to set the sink node
    def set_sink_node(self, sink_node):
        self.sink_node = sink_node

    # Function to set the capacity of the links
    def set_link_capacity(self, link_capacity):
        self.link_capacity = link_capacity

    # Function to update the link's weight (travel time) after each assignment step
    def update_edge_weight(self, u, v, new_weight):
        if self.graph.has_edge(u, v):
            self.graph[u][v]['weight'] = new_weight
        else:
            print(f"Edge ({u}, {v}) does not exist in the graph.")

    # Function to update the edge's availability due to limited capacity
    def update_edge_activity(self, u, v, active):
        if self.graph.has_edge(u, v):
            self.graph[u][v]['active'] = active
        else:
            print(f"Edge ({u}, {v}) does not exist in the graph.")

    # Function to incrementally update the network's travel time on links
    def incremental_assignment(self, demand, increment, use_congested_links=False, draw=False):
        src_node = self.src  # Source node
        dest_node = self.sink_node  # Sink node
        dist, path = self.dijkstra(src_node)  # Shortest path at zero stage
        cost_to_destination = dist[dest_node]  # Cost to the sink node
        if self.sink_node is None:
            print("Sink node not set. Use set_sink_node() to set the sink node.")
            return

        # number of steps. Works for integer quotients only
        num_steps = demand // increment
        current_vehicles = 0
        if draw:
            self.draw_graph(filename=f"network_{0}.png", pos=self.pos, path=path)

        # main loop
        for step in range(1, num_steps + 1):
            if current_vehicles < demand and path[dest_node] != -1:
                # Update the volume of each link on the shortest path
                shortest_path = construct_path(path, src_node, dest_node)
                for u, v in zip(shortest_path, shortest_path[1:]):

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
                    capacity = self.graph[u][v]['capacity']
                    volume = self.graph[u][v].get('volume', 0)
                    new_weight = impact_function(self.graph[u][v]['free_flow_time'], volume / capacity)
                    self.update_edge_weight(u, v, new_weight)

                # Calculate the new shortest path and its cost
                dist, path = self.dijkstra(src_node)

                print(f"Step {step}: Shortest path at this step: {' -> '.join(map(str, shortest_path))}")
                print(f"Number of nodes met in the shortest path: {len(shortest_path)}")
                print(f"Cost to destination at this step: {cost_to_destination}")

                # Draw the graph with the updated travel times
                if step < num_steps and draw:
                    self.draw_graph(filename=f"network_{step}.png", pos=self.pos, path=path)

                # Increment the number of vehicles
                cost_to_destination = dist[dest_node]
                current_vehicles += increment

                print(f"Step {step}: Incremental assignment completed. Total vehicles: {current_vehicles}")
                print("." * 60)

        print("Incremental assignment completed for all steps.")
        print("-" * 60)
        print(f'Total travel time is: {calculate_volume_times_travel_time(self.graph)} veh*hr')

    # Function to set the source node
    def set_src_node(self, src_node):
        self.src = src_node


