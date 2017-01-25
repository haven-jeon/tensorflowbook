library(tensorflow)

a <- tf$random_normal(c(2L,20L))
sess <- tf$Session()
out <- sess$run(a)
x_y <- t(out)

plot(x_y[,1], x_y[,2])
