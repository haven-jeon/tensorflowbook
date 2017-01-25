library(tensorflow)

W <- tf$Variable(tf$zeros(c(2,1)), name='weights', dtype=tf$float32)
b <- tf$Variable(0.0, name="bias",  dtype=tf$float32)

inference <- function(X){
  return(tf$matmul(tf$cast(X, tf$float32), W) + b)
}


loss <- function(X, Y){
  Y_predicted <- inference(X)
  return(tf$reduce_sum(tf$squared_difference(Y, Y_predicted)))
}


#http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt
data <- 
"
 1  1  84  46  354
 2  1  73  20  190
 3  1  65  52  405
 4  1  70  30  263
 5  1  76  57  451
 6  1  69  25  302
 7  1  63  28  288
 8  1  72  36  385
 9  1  79  57  402
10  1  75  44  365
11  1  27  24  209
12  1  89  31  290
13  1  65  52  346
14  1  57  23  254
15  1  59  60  395
16  1  69  48  434
17  1  60  34  220
18  1  79  51  374
19  1  75  50  308
20  1  82  34  220
21  1  59  46  311
22  1  67  23  181
23  1  85  37  274
24  1  55  40  303
25  1  63  30  244
"


inputs <- function(){
  tr_data <- read.table(text=data)
  weight_age <- as.matrix(tr_data[,3:4])
  blood_fat_content <- as.matrix(tr_data[,5])
  return(list(tf$to_float(weight_age), tf$to_float(blood_fat_content))) 
}


train <- function(total_loss){
  learning_rate <- 0.0000001
  return(tf$train$GradientDescentOptimizer(learning_rate)$minimize(total_loss))
}

evaluate <- function(sess, X, Y){
  print(sess$run(inference(matrix(c(80.0, 25.0),ncol=2))))
  print(sess$run(inference(matrix(c(65.0, 25.0),ncol=2))))
}


with(tf$Session() %as% sess, {
  tf$global_variables_initializer()$run()
  
  X_Y <- inputs()
  
  total_loss <- loss(X_Y[[1]], X_Y[[2]])
  train_op <- train(total_loss)
  
  coord <- tf$train$Coordinator()
  threads <- tf$train$start_queue_runners(sess=sess, coord=coord)
  
  # actual training loop
  training_steps <- 1000
  for(step in 1:training_steps){
    sess$run(train_op)
    if(step %% 10 == 0){
      print(sprintf("loss : %f", sess$run(total_loss)))
    }
  }
  evaluate(sess, X, Y)
  coord$request_stop()
  coord$join(threads)
  sess$close()
})






