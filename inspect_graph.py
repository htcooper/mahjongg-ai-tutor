import tensorflow as tf

# Load frozen inference graph
def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

    with graph.as_default():
        tf.compat.v1.import_graph_def(graph_def, name='')

    return graph

# Path to model file
model_file = 'model.pb'

# Load graph
graph = load_graph(model_file)

# Print out all operation names
with graph.as_default():
    for op in graph.get_operations():
        print(op.name)
