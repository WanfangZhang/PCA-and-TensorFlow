#install.packages("tensorflow") 
library(tensorflow) 
#install_tensorflow()
#install.packages("keras") 
library(keras)
library(nnet)

mnist=dataset_mnist()
X_train=mnist$train$x
y_train=mnist$train$y
# visualize the digits
jpeg(file="C:/Users/wanfang/Downloads/BIOS825/Project/mn4.jpg")
par(mfcol=c(4,4))
par(mar=c(0, 0, 0, 0), xaxs='i', yaxs='i')
for (idx in 1:16) { 
  im=X_train[idx,,] 
  im=t(apply(im, 2, rev)) 
  image(1:28, 1:28, im, col=gray((0:255)/255), 
        xaxt='n')
}
dev.off()


#install_keras()
# Load libraries
library(tidyverse)
library(jpeg)
library(factoextra)
library(knitr)
library(xtable)

# Data points
x=c(2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1)
y=c(2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9)

# Data frame
data=data.frame(x, y)
xtable(data)
# Scatter plot
pdf("C:\\Users\\wanfang\\Downloads\\BIOS825\\Project\\F1.pdf",width=5,height=4)
data %>%
  ggplot(aes(x,y)) +
  geom_point(size=2, shape=3, color="blue") +
  theme_bw() +
  labs(title="Original data points")
dev.off()

# Structure
str(data)
# Eigenvectors and eigenvalues calculation
eigen=eigen(data.frame(c(0.616555556, 0.615444444), 
                          c(0.615444444, 0.716555556)))

# Eigenvectors
eigen$vectors
# Eigenvalues
eigen$values
# Points with the mean substracted 
xMeanSubstracted=x - mean(x)
yMeanSubstracted=y - mean(y)
data2=data.frame(xMeanSubstracted, yMeanSubstracted)
xtable(cbind(data,data2))
data3=data2^2
xtable(cbind(data,data2,data2[,1]*data2[,2],data3))

# Eigenvectors functions  
fun.1=function(x) (0.7351787/0.6778734)*x
fun.2=function(x) (0.6778734/-0.7351787)*x

# Scatter plot with the eigenvectors overlayed
pdf("C:\\Users\\wanfang\\Downloads\\BIOS825\\Project\\F2.pdf",width=5,height=4)
data2 %>%
  ggplot(aes(xMeanSubstracted, yMeanSubstracted)) +
  geom_point(size=2, shape=3, color="blue") +
  stat_function(fun=fun.1, linetype="dashed") +
  stat_function(fun=fun.2, linetype="dashed") +
  theme_bw() +
  xlim(-1.5, 1.5) +
  labs(title="Mean adjusted data with eigenvectors overlayed",
       x="x", y="y") +
  annotate("text", x=c(-1.1, 0.9), y=c(1.5, 1.5), 
           label=c("Second Component", "First Component"))
dev.off()

# Principal Component Analysis
pca=prcomp(data, center=TRUE)

# We can visualize the eigenvectors with the function fviz_eig() 
# Documentation: https://www.rdocumentation.org/packages/factoextra/versions/1.0.5/topics/eigenvalue
# Unfortunately, I get an error when I execute this function on Kaggle
fviz_eig(pca)

# Cumulative proportion
summary(pca)
# Data expressed in terms of our 2 eigenvectors
dataNewAxes=as.data.frame(t(t(eigen$vectors) %*% rbind(x - mean(x), y - mean(y))))
names(dataNewAxes)=c("x", "y")

# New data set 
xtable(dataNewAxes)
# Visualization
pdf("C:\\Users\\wanfang\\Downloads\\BIOS825\\Project\\F3.pdf",width=5,height=4)
dataNewAxes %>%
  ggplot(aes(x, y)) +
  geom_point(size=2, shape=3, color="blue") +
  theme_bw() +
  labs(title="Data expressed in terms of our 2 eigenvectors",
       x="First Component", y="Second Component") 
dev.off()

# Reconstructed data using only the first principal component 
pdf("C:\\Users\\wanfang\\Downloads\\BIOS825\\Project\\F4.pdf",width=5,height=4)
as.data.frame(t(t(pca$x[, 1] %*% t(pca$rotation[, 1])) + pca$center)) %>%
  ggplot(aes(x, y)) +
  geom_point(size=2, shape=3, color="blue") +
  theme_bw() +
  labs(title="Original data restored using only a single eigenvector")
dev.off()

image=readJPEG("C:/Users/wanfang/Downloads/BIOS825/Project/mn4.jpg")
image=readJPEG("C:/Users/wanfang/Downloads/BIOS825/Project/meme.jpg")
img_gray=apply(image, c(1,2), function(x) weighted.mean(x, c(0.299, 0.587, 0.114)))
jpeg(file="C:/Users/wanfang/Downloads/BIOS825/Project/gray.jpg")
image(img_gray, col=gray(0:255/255))
dev.off()

str(image)
str(im)
# RGB color matrices
rimage=image[,,1]
gimage=image[,,2]
bimage=image[,,3]

# PCA for each color scheme
pcar=prcomp(rimage, center=FALSE)
pcag=prcomp(gimage, center=FALSE)
pcab=prcomp(bimage, center=FALSE)

# PCA objects into a list
pcaimage=list(pcar, pcag, pcab)

# PCs values
pcnum=c(2, 30, 200, 600)

# Reconstruct the image four times
for(i in pcnum){
  pca.img=sapply(pcaimage, function(j){
    compressed.img=j$x[, 1:i] %*% t(j$rotation[, 1:i])
  }, simplify='array') 
  writeJPEG(pca.img, 
  paste("C:/Users/wanfang/Downloads/BIOS825/Project/meme reconstruction with", 
                           round(i, 0), "principal components.jpg"))
}



########################## TensorFlow #########################################


#mnist=dataset_mnist()
X_train=mnist$train$x
X_test=mnist$test$x
y_train=mnist$train$y
y_test=mnist$test$y
X_train=array_reshape(X_train, c(nrow(X_train), 784))
X_train=X_train / 255

X_test=array_reshape(X_test, c(nrow(X_test), 784))
X_test=X_test / 255

model_lr=multinom(as.factor(y_train) ~ ., data=as.data.frame(X_train), 
                  MaxNWts=10000,decay=5e-3, maxit=100)
prediction_lr=predict(model_lr, X_test, type = "class")
accuracy=mean(prediction_lr == mnist$test$y)
cat('Test accuracy:', accuracy, '\n')
xtable(table(prediction_lr,mnist$test$y))


y_train=to_categorical(y_train, num_classes = 10)
y_test=to_categorical(y_test, num_classes = 10)
model=keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu", input_shape = c(784)) %>%
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>% 
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.25) %>%
  layer_dense(units = 10, activation = "softmax")
summary(model)

model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_adam(),
  metrics = c("accuracy")
)

history=model %>% 
  fit(X_train, y_train, epochs = 50, batch_size = 128, validation_split = 0.15)

model %>%
  evaluate(X_test, y_test)

y_pred_probs=model %>% predict(X_test)
apply(y_pred_probs, 1, which.max) - 1  # Get the class label with highest probability
predicted_classes=apply(y_pred_probs, 1, which.max) - 1  

accuracy=mean(predicted_classes == mnist$test$y)
cat('Test accuracy:', accuracy, '\n')
xtable(table(mnist$test$y, predicted_classes))