library(tensorflow)

# Logistic regression example in TF using Kaggle's Titanic Dataset.
# library(titanic)


W <- tf$Variable(tf$zeros(c(4,1)), name='weights', dtype = tf$float32)
b <- tf$Variable(0.0, name='bias',  dtype = tf$float32)

combine_inputs <- function(X){
  return(tf$matmul(tf$cast(X, tf$float32), W) + b)
}


inference <- function(X){
  return(tf$sigmoid(combine_inputs(X)))
}

loss <- function(X,Y){
  return(tf$reduce_mean(tf$nn$sigmoid_cross_entropy_with_logits(combine_inputs(X), tf$cast(Y, tf$float32))))
}


read_csv <- function(batch_size, file_name, record_defaults){
    print(list(file.path(getwd(), file_name)))
    filename_queue <- tf$train$string_input_producer(list(file.path(getwd(), file_name)))

    reader <- tf$TextLineReader(skip_header_lines=1)
    key_value <- reader$read(filename_queue)

    # decode_csv will convert a Tensor from type string (the text line) in
    # a tuple of tensor columns with the specified defaults, which also
    # sets the data type for each column
    decoded <- tf$decode_csv(key_value$value, record_defaults=record_defaults)
    
    return(tf$train$shuffle_batch(decoded, 
                                  batch_size=batch_size, 
                                  capacity=batch_size * 50, allow_smaller_final_batch=TRUE,
                                  min_after_dequeue=batch_size))
}


inputs <- function(){
  #install.packages('titanic')
  titanic <- na.omit(titanic::titanic_train)
  x_mtrx <- model.matrix(~ factor(Pclass) + factor(Sex) + Age , data=titanic)[,-1]
  y_mtrx <- as.matrix(titanic[,"Survived"])
  # 
  # batch_records <- read_csv(100L, "titanic.csv",list(tf$constant(list(0)), tf$constant(list(0)), tf$constant(list(0L)), 
  #      tf$constant(list("")),  tf$constant(list("")), tf$constant(list(0.0)),tf$constant(list(0L)),tf$constant(list(0)), 
  #      tf$constant(list("")), tf$constant(list(0.0)), tf$constant(list("")), tf$constant(list(""))))
  # # convert categorical data
  # is_first_class <- tf$to_float(tf$equal(batch_records[[3]], c(1L)))
  # is_second_class<- tf$to_float(tf$equal(batch_records[[3]], c(2L)))
  # is_third_class <- tf$to_float(tf$equal(batch_records[[3]], c(3L)))
  # 
  # gender <- tf$to_float(tf$equal(batch_records[[5]], c("female")))
  # 
  # age <- batch_records[[6]] 
  # survived <- batch_records[[2]]
  # # Finally we pack all the features in a single matrix;
  # # We then transpose to have a matrix with one example per row and one feature per column.
  # features <- tf$transpose(tf$pack(list(is_first_class, is_second_class, is_third_class, gender, age)))
  # 
  # survived <- tf$reshape(survived, shape = c(100L, 1L))
  # #tf$Print(survived)
  return(list(features=x_mtrx, survived=y_mtrx))
}



train <- function(total_loss){
  learning_rate <- 0.001
  return(tf$train$GradientDescentOptimizer(learning_rate)$minimize(total_loss))
}


evaluate <- function(sess, X, Y){
  predicted <- tf$cast(inference(X) > 0.5, tf$float32)
  print(sess$run(tf$reduce_mean(tf$cast(tf$equal(predicted, Y), tf$float32))))
}

with(tf$Session() %as% sess, {

  tf$global_variables_initializer()$run(session=sess)
  
  X_Y <- inputs()
  total_loss <- loss(X_Y$features, X_Y$survived) 
  train_op <- train(total_loss)
  
  
  coord <- tf$train$Coordinator()
  
  threads <- tf$train$start_queue_runners(sess=sess, coord=coord)

  # actual training loop
  training_steps <- 1000
  for(step in 1:training_steps){
    sess$run(train_op)
    # for debugging and learning purposes, see how the loss gets decremented thru training steps
    if(step %% 10 == 0){
      print(sprintf("loss : %f", sess$run(total_loss)))
    }
    evaluate(sess, X_Y$features, X_Y$survived)
    Sys.sleep(0.5)
    coord$request_stop()
    coord$join(threads)
    #sess.close()
  }
})
