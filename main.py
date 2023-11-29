import Graph
import os

filename = os.getcwd() + '\\network.csv'
src_node = 68
sink_node = 6575
num_vehicles = 8000
capacity_increment = 1000
link_capacity = 4000
pos = {
    0: (0, 2),
    1: (1, 3),
    2: (1, 2),
    3: (1, 1),
    4: (2, 3),
    5: (2, 2),
    6: (2, 1),
    7: (2, 0),
    8: (3, 2),
    9: (3, 0)
}


def main():
    g = Graph.Graph()
    g.read_graph_from_file(filename=filename)
    g.set_sink_node(sink_node)
    g.set_src_node(src_node)
    # g.dijkstra(src=src_node)
    # g.incremental_assignment(demand=num_vehicles, increment=capacity_increment)
    # g.draw_graph(filename='network', pos=pos)
    # shortest_path, cost_to_destination = g.get_solution_to_destination(sink_node)
    # if shortest_path is not None:
    #     print(f"Shortest path to destination {sink_node}: {shortest_path}")
    #     print(f"Cost to destination: {cost_to_destination}")
    g.incremental_assignment(num_vehicles, capacity_increment)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
