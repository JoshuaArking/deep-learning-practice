import numpy as np

# this example follows https://www.datacamp.com/community/tutorials/introduction-deep-learning

# this is the input data with 2 points. the 40 is the person's age, and 0 is their retirement status.
input_data = np.array([40,0])

# nodes 0 and 1 are the hidden layer, and output is the output layer
weights = {'node0':([1,1]),
           'node1':([1,-1]),
           'output':([1,-1])}

# calculate the value of node 0 and 1in the hidden layer
node0_val = (input_data * weights['node0']).sum()
node1_val = (input_data * weights['node1']).sum()

# combine the two hidden layer values
hidden_val = np.array([node0_val,node1_val])

# multiply hidden value layers with output layer weight and use sum to make a prediction (0 is yes, 1 is no)
out_val = (hidden_val * weights['output']).sum()

# this demonstrates a linear activation function
print("linear activation function output is " + str(out_val))

# apply an s-shaped activation function
node0_act = np.tanh(node0_val)
node1_act = np.tanh(node1_val)

# combine new hidden layer values
hidden_act = np.array([node0_act,node1_act])

# calculate new output
out_act = (hidden_act * weights['output']).sum()

# output of a s-shaped activation function
print("s-shaped activation function output is " + str(out_act))

# define a Rectifier Linear Unit (ReLU) activation function
def relu(input):

    output = max(0, input)

    return output

# calculate new node values with relu activation function
node0_relu = relu(node0_val)
node1_relu = relu(node1_val)

# combine new hidden layer
hidden_relu = np.array([node0_relu,node1_relu])

# calculate new output
out_relu = (hidden_relu * weights['output']).sum()

# output of a relu activation function
print("ReLU activation function output is " + str(out_relu))

# example with more layers in the neural network
input_data_m = np.array([3,5])

# first number in key name is first layer, second number is second layer
# node 0_0 and 0_1 are top and bottom of first layer, 1_0 and 1_1 are top and bottom of second layer, viewing left to right
weights_m = {'node0_0':([2,4]),
           'node0_1':([4,-5]),
           'node1_0':([-1,2]),
           'node1_1':([1,2]),
           'output':([2,7])}

def predict_with_network(input_data_m):
    node_0_0_val = (input_data_m * weights_m['node0_0']).sum()
    node_0_1_val = (input_data_m * weights_m['node0_1']).sum()

    node_0_0_relu = relu(node_0_0_val)
    node_0_1_relu = relu(node_0_1_val)

    hidden_0 = np.array([node_0_0_relu,node_0_1_relu])

    node_1_0_val = (hidden_0 * weights_m['node1_0']).sum()
    node_1_1_val = (hidden_0 * weights_m['node1_1']).sum()

    node_1_0_relu = relu(node_1_0_val)
    node_1_1_relu = relu(node_1_1_val)

    hidden_1 = np.array([node_1_0_relu,node_1_1_relu])

    out_m_r = (hidden_1 * weights_m['output']).sum()

    return out_m_r

out_m = predict_with_network(input_data_m)
print("Multi-layer neural network with ReLU activation function output is " + str(out_m))