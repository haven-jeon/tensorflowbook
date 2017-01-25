library(tensorflow)

sess <- tf$InteractiveSession()

input_batch <- tf$constant(array(c(0.0, 2.0, 1.0, 3.0, 2.0,6.0,4.0,8.0), dim = c(2,2,2,1)),dtype = tf$float32)

sess$run(input_batch)


kernel <- tf$constant(array(c(1.0,2.0),c(1,1,1,2)),dtype = tf$float32)


sess$run(kernel)


conv2 <- tf$nn$conv2d(input_batch, kernel, strides=c(1.0, 1.0, 1.0, 1.0), padding='SAME')

sess$run(conv2)




