import time

import Graph
import os

filename = os.getcwd() + '\\network.csv'
src_node = 723
sink_node = 6856
num_vehicles = 8000
capacity_increment = 1000
link_capacity = 4000
draw = False
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
    t0 = time.time()
    g = Graph.Graph(pos=pos)
    g.set_link_capacity(link_capacity)
    g.read_graph_from_file(filename=filename)
    g.set_sink_node(sink_node)
    g.set_src_node(src_node)
    g.incremental_assignment(num_vehicles, capacity_increment, draw=draw)
    t1 = time.time()

    print(f'Time required to apply incremental shortest path: {t1 - t0:.2f} seconds')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
