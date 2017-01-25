# Import the tensorflow library, and reference it as `tf`
library(tensorflow)

# Build our graph nodes, starting from the inputs
a <- tf$constant(5L, name="input_a")
b <- tf$constant(3L, name="input_b")
c <- tf$mul(a,b, name="mul_c")
d <- tf$add(a,b, name="add_d")
e <- tf$add(c,d, name="add_e")

# Open up a TensorFlow Session
sess = tf$Session()

# Execute our output node, using our Session
sess.run(e)

# Open a TensorFlow SummaryWriter to write our graph to disk
writer = tf.train.SummaryWriter('./my_graph', sess.graph)

# Close our SummaryWriter and Session objects
writer.close()
sess.close()

# To start TensorBoard after running this file, execute the following command:
# $ tensorboard --logdir='./my_graph'