library(tensorflow)



# this time weights form a matrix, not a column vector, one "weight vector" per class.
W <- tf$Variable(tf.zeros(c(4L, 3L)), name="weights")
# so do the biases, one per class.
b <- tf$Variable(tf$zeros(3L, name="bias"))



combine_inputs <- function(X){
  return(tf$matmul(X, W) + b)  
}

inference <- function(X){
  return(tf$nn$softmax(combine_inputs(X)))
}

loss <- function(X, Y){
  return(tf$reduce_mean(tf$nn$sparse_softmax_cross_entropy_with_logits(combine_inputs(X), Y)))
}

read_csv <- function(batch_size, file_name, record_defaults){
  filename_queue <- tf$train$string_input_producer(list(file.path(file_name)))

  reader <- tf$TextLineReader(skip_header_line=1L)
  
  key_value = reader$read(filename_queue)

  # decode_csv will convert a Tensor from type string (the text line) in
  # a tuple of tensor columns with the specified defaults, which also
  # sets the data type for each column
  decoded <- tf$decode_csv(value, record_defaults=record_defaults)
  return(tf$train$shuffle_batch(decoded,
                                batch_size=batch_size,
                                capacity=batch_size * 50,
                                min_after_dequeue=batch_size))

}


inputs <- function(){
  records <- read_csv(100L, "iris.data", list(tf$constant(list(0.0)),tf$constant(list(0.0)),
                                              tf$constant(list(0)),tf$constant(list(0)),
                                              tf$constant(list(""))))
  label_number <- tf$to_int32(tf$argmax(tf$to_int32(tf$pack(list(
        tf$equal(label, "Iris-setosa"),
        tf$equal(label, "Iris-versicolor"),
        tf$equal(label, "Iris-virginica")
    ))), 0L))
  features <- tf$transpose(tf$pack(list(sepal_length, sepal_width, petal_length, petal_width)))
  return(list(X=features, Y=label_number))
}


train <- function(total_loss){
  learning_rate <- 0.01
  return(tf$train$GradientDescentOptimizer(learning_rate)$minimize(total_loss))
}

evaluate <- function(sess, X, Y){
  predicted < tf$cast(tf$arg_max(inference(X), 1), tf$int32)

  print(sess$run(tf$reduce_mean(tf$cast(tf$equal(predicted, Y), tf$float32))))

}


with(tf$Session %as% sess, {
  tf$global_variables_initializer()$run()

  X_Y <- inputs()

  total_loss <- loss(X_Y$X, X_Y$Y)
  train_op <- train(total_loss)

  coord <- tf$train$Coordinator()
  threads <- tf$train$start_queue_runners(sess=sess, coord=coord)

  # actual training loop
  training_steps = 1000
  for(step in 1:training_steps){
      sess$run(train_op)
      # for debugging and learning purposes, see how the loss gets decremented thru training steps
      if(step %% 10 == 0){
          print(sprintf("loss: %f", sess$run(total_loss)))
      }
  }
  evaluate(sess, X_Y$X, X_Y$Y)

  coord$request_stop()
  coord$join(threads)
  #sess$close()
})

