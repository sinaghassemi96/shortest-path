import Graph
import os

filename = os.getcwd() + '\\simple_net.csv'
src_node = 0
sink_node = 8
num_vehicles = 100
capacity_increment = 20
link_capacity = 40
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
    g = Graph.Graph(pos=pos)
    g.read_graph_from_file(filename=filename)
    g.set_sink_node(sink_node)
    g.dijkstra(src=src_node)
    g.incremental_assignment(demand=num_vehicles, increment=capacity_increment)
    g.draw_graph(filename='network', pos=pos)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
