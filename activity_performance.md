---
title: "Sport Wearable Devices and Activity Performance"
author: "Riccardo Finotello"
date: "24 June 2020"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---



## Summary

In this analysis we will consider data of sport wearable devices and we will
tackle the task of predicting the performance of dumbell exercises as perceived
from the device. In other words we will try to predict the assigned class of the
exercise from motion sensor data.

## Cleaning the Data

We will first access the training data:


```r
data <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                 na.strings = c("#DIV/0!", "", "NA"),
                 stringsAsFactors = FALSE
                )
```

We then take a look at the number of entries and features in the dataset:


```r
print(paste("No. of entries:", nrow(data)))
```

```
## [1] "No. of entries: 19622"
```

```r
print(paste("No. of columns<:", ncol(data)))
```

```
## [1] "No. of columns<: 160"
```

From the dataset we remove information related to the specific user and keep
only numerical features and non empty columns:


```r
data <- data[,-(1:7)]
data$classe <- as.factor(data$classe) # convert classe to factor
```

We then compute the fraction of **NA** data in each column and remove those
variables with more than 50% of missing data. This shows that we will keep only
a fraction of the original variables:


```r
# find fraction of NA values
data.na.fraction <- sapply(data, function(x) {mean(is.na(x))})
data.cols <- (data.na.fraction <= 0.5)
print(paste("Fraction of retained variables:", round(mean(data.cols),2)))
```

```
## [1] "Fraction of retained variables: 0.35"
```

```r
# select the relevant columns
train.data <- data[, data.cols]
```
We therefore have a tidy dataset containing
100% of complete cases and new dimensions:


```r
print(paste("No. of entries:", nrow(train.data)))
```

```
## [1] "No. of entries: 19622"
```

```r
print(paste("No. of columns:", ncol(train.data)))
```

```
## [1] "No. of columns: 53"
```

We finally divide the features we use for training from the labels we try to
predict:


```r
# shuffle the dataset
train.data <- train.data[sample(nrow(train.data)),]

# store the id of the classes (last column)
labels <- c(ncol(train.data))
```

As an additional step before the exploratory data analysis, we divide the
training set into a further partition for testing:


```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.6.3
```

```
## Warning: package 'lattice' was built under R version 3.6.3
```

```
## Warning: package 'ggplot2' was built under R version 3.6.3
```

```r
train.part <- createDataPartition(train.data[,ncol(train.data)], p=0.8, list=FALSE)
train <- train.data[train.part,]
test <- train.data[-train.part,]
```

## Exploratory Data Analysis

To better understand the distribution of the variables, we study their
correlations properties (**we consider only the training partition and we do not
look at the labels to avoid biasing the strategy**):


```r
library(reshape2)
```

```
## Warning: package 'reshape2' was built under R version 3.6.3
```

```r
corr.mat <- cor(train[,-labels])
corr.mat.melt <- melt(corr.mat)

# plot the correlation matrix
library(ggplot2)
g <- ggplot(data = corr.mat.melt, aes(Var1, Var2, fill = value)) +
     geom_tile() +
     xlab("") +
     ylab("") +
     scale_fill_gradient2(low = "red", mid= "white", high = "blue",
                          midpoint = 0, limit = c(-1,1)
                         ) +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1))
print(g)
```

![](activity_performance_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

As we can see the variables are in general not correlated, apart from (mainly)
the accelerometers and gyroscope (partly as expected). We may expect them to
play a greater part in the prediction or at least we will see that they will be
more interacting.

In fact we can dig a bit more on the subject and visualise the distribution of
these sensors with respect to the `classe` label:


```r
library(gridExtra)
```

```
## Warning: package 'gridExtra' was built under R version 3.6.3
```

```r
g1 <- ggplot(data = train,
             aes(x = accel_belt_x, y = accel_belt_y, col = accel_belt_z)) +
      geom_point() +
      scale_color_gradient2(low = "red", mid = "white", high = "blue",
                            midpoint = 0)
g2 <- ggplot(data = train,
             aes(x = gyros_belt_x, y = gyros_belt_y, col = gyros_belt_z)) +
      geom_point() +
      scale_color_gradient2(low = "red", mid = "white", high = "blue",
                            midpoint = 0)

grid.arrange(g1, g2, nrow = 1)
```

![](activity_performance_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

We then divide the training variables from the labels:


```r
x.train <- train[,-labels]
y.train <- train[,labels]
x.test  <- test[,-labels]
y.test  <- test[,labels]
```

The summary of the training variables can then be useful to understand the next
step in the analysis:


```r
library(skimr)
```

```
## Warning: package 'skimr' was built under R version 3.6.3
```

```r
skim(x.train)
```


Table: Data summary

                                   
-------------------------  --------
Name                       x.train 
Number of rows             15699   
Number of columns          52      
_______________________            
Column type frequency:             
numeric                    52      
________________________           
Group variables            None    
-------------------------  --------


**Variable type: numeric**

skim_variable           n_missing   complete_rate      mean       sd         p0       p25       p50       p75      p100  hist  
---------------------  ----------  --------------  --------  -------  ---------  --------  --------  --------  --------  ------
roll_belt                       0               1     64.37    62.78     -28.90      1.10    113.00    123.00    162.00  ▇▁▁▅▅ 
pitch_belt                      0               1      0.33    22.41     -55.80      1.72      5.28     15.00     60.30  ▃▁▇▅▁ 
yaw_belt                        0               1    -11.20    95.19    -180.00    -88.30    -13.10     12.80    179.00  ▁▇▅▁▃ 
total_accel_belt                0               1     11.32     7.74       0.00      3.00     17.00     18.00     29.00  ▇▁▂▆▁ 
gyros_belt_x                    0               1     -0.01     0.21      -1.04     -0.05      0.03      0.11      2.22  ▁▇▁▁▁ 
gyros_belt_y                    0               1      0.04     0.08      -0.64      0.00      0.02      0.11      0.64  ▁▁▇▁▁ 
gyros_belt_z                    0               1     -0.13     0.24      -1.46     -0.20     -0.10     -0.02      1.62  ▁▂▇▁▁ 
accel_belt_x                    0               1     -5.64    29.71    -120.00    -21.00    -15.00     -5.00     85.00  ▁▂▇▁▂ 
accel_belt_y                    0               1     30.14    28.66     -69.00      3.00     34.00     61.00    164.00  ▁▇▇▁▁ 
accel_belt_z                    0               1    -72.56   100.46    -275.00   -162.00   -152.00     27.00    105.00  ▁▇▁▅▃ 
magnet_belt_x                   0               1     55.55    64.17     -52.00      9.00     35.00     59.00    485.00  ▇▁▂▁▁ 
magnet_belt_y                   0               1    593.69    35.68     354.00    581.00    601.00    610.00    673.00  ▁▁▁▇▃ 
magnet_belt_z                   0               1   -345.36    65.87    -623.00   -375.00   -320.00   -306.00    293.00  ▁▇▁▁▁ 
roll_arm                        0               1     18.38    72.80    -180.00    -30.75      0.00     77.60    180.00  ▁▃▇▆▂ 
pitch_arm                       0               1     -4.61    30.80     -88.80    -25.90      0.00     11.30     88.50  ▁▅▇▂▁ 
yaw_arm                         0               1     -0.35    71.30    -180.00    -42.75      0.00     46.15    180.00  ▁▃▇▃▂ 
total_accel_arm                 0               1     25.50    10.57       1.00     17.00     27.00     33.00     66.00  ▃▆▇▁▁ 
gyros_arm_x                     0               1      0.04     1.99      -6.37     -1.33      0.08      1.56      4.87  ▁▃▇▆▂ 
gyros_arm_y                     0               1     -0.26     0.85      -3.44     -0.80     -0.24      0.16      2.84  ▁▂▇▂▁ 
gyros_arm_z                     0               1      0.27     0.55      -2.33     -0.07      0.25      0.72      3.02  ▁▂▇▂▁ 
accel_arm_x                     0               1    -59.08   182.21    -383.00   -241.00    -42.00     84.00    437.00  ▇▅▇▃▁ 
accel_arm_y                     0               1     32.58   109.54    -318.00    -54.00     14.00    138.00    308.00  ▁▃▇▆▂ 
accel_arm_z                     0               1    -71.47   135.24    -636.00   -144.00    -47.00     23.00    292.00  ▁▁▅▇▁ 
magnet_arm_x                    0               1    192.39   443.71    -584.00   -299.00    292.00    638.00    782.00  ▆▃▂▃▇ 
magnet_arm_y                    0               1    156.20   202.30    -386.00    -11.00    202.00    323.00    582.00  ▁▅▅▇▂ 
magnet_arm_z                    0               1    305.00   327.43    -597.00    129.00    443.00    544.00    694.00  ▁▂▂▃▇ 
roll_dumbbell                   0               1     23.62    69.92    -153.71    -19.63     48.01     67.59    153.55  ▂▂▃▇▂ 
pitch_dumbbell                  0               1    -10.65    36.93    -137.34    -40.42    -20.97     17.57    149.40  ▁▇▇▂▁ 
yaw_dumbbell                    0               1      2.12    82.60    -150.87    -77.56     -2.20     80.52    154.75  ▃▇▅▅▆ 
total_accel_dumbbell            0               1     13.68    10.19       0.00      4.00     10.00     19.00     42.00  ▇▅▃▃▁ 
gyros_dumbbell_x                0               1      0.17     0.39      -1.99     -0.03      0.13      0.35      2.22  ▁▁▇▁▁ 
gyros_dumbbell_y                0               1      0.04     0.49      -2.10     -0.14      0.03      0.21      4.37  ▁▇▁▁▁ 
gyros_dumbbell_z                0               1     -0.14     0.32      -2.38     -0.31     -0.13      0.03      1.72  ▁▁▇▂▁ 
accel_dumbbell_x                0               1    -28.15    66.99    -237.00    -50.00     -8.00     11.00    235.00  ▁▂▇▁▁ 
accel_dumbbell_y                0               1     52.42    80.94    -189.00     -9.00     41.00    111.00    315.00  ▁▇▇▅▁ 
accel_dumbbell_z                0               1    -37.82   109.13    -334.00   -141.00     -1.00     39.00    318.00  ▁▆▇▃▁ 
magnet_dumbbell_x               0               1   -325.78   341.59    -643.00   -535.00   -478.00   -296.00    584.00  ▇▂▁▁▂ 
magnet_dumbbell_y               0               1    218.66   329.68   -3600.00    231.00    310.00    390.00    633.00  ▁▁▁▁▇ 
magnet_dumbbell_z               0               1     46.48   140.30    -250.00    -45.00     14.00     96.00    451.00  ▂▇▅▂▂ 
roll_forearm                    0               1     33.79   108.39    -180.00     -0.70     21.90    140.00    180.00  ▃▂▇▂▇ 
pitch_forearm                   0               1     10.75    28.06     -72.50      0.00      9.07     28.50     89.80  ▁▁▇▃▁ 
yaw_forearm                     0               1     19.74   103.08    -180.00    -67.75      0.00    110.00    180.00  ▅▅▇▆▇ 
total_accel_forearm             0               1     34.72    10.04       0.00     29.00     36.00     41.00     79.00  ▁▃▇▁▁ 
gyros_forearm_x                 0               1      0.16     0.63      -4.95     -0.22      0.05      0.56      3.97  ▁▁▇▃▁ 
gyros_forearm_y                 0               1      0.05     2.17      -6.62     -1.49      0.03      1.62      6.13  ▁▅▇▅▁ 
gyros_forearm_z                 0               1      0.14     0.60      -8.09     -0.18      0.08      0.49      4.31  ▁▁▁▇▁ 
accel_forearm_x                 0               1    -61.29   180.18    -498.00   -178.00    -56.00     77.00    370.00  ▂▅▇▆▂ 
accel_forearm_y                 0               1    164.09   200.08    -632.00     57.00    201.00    313.00    591.00  ▁▂▃▇▃ 
accel_forearm_z                 0               1    -55.34   138.47    -446.00   -182.00    -40.00     26.00    291.00  ▁▇▅▅▃ 
magnet_forearm_x                0               1   -312.37   346.79   -1280.00   -616.00   -377.00    -78.00    672.00  ▁▇▇▅▂ 
magnet_forearm_y                0               1    379.25   509.29    -896.00     -0.50    592.00    737.00   1480.00  ▂▂▂▇▁ 
magnet_forearm_z                0               1    393.27   369.45    -973.00    186.50    512.00    654.00   1090.00  ▁▁▃▇▃ 

## Machine Learning Predictions

we first preprocess the data and standardise the training features:


```r
preprocess <- preProcess(x.train, method = c("center", "scale"))
x.new.train <- predict(preprocess, newdata = x.train)
x.new.test  <- predict(preprocess, newdata = x.test)
```

We finally apply a **random forest** algorithm to the training data. We use a
10-fold cross-validation asstrategy (partly to prevent overfit and improve
predictions):


```r
# set multithreading
library(doParallel)
```

```
## Warning: package 'doParallel' was built under R version 3.6.3
```

```
## Warning: package 'foreach' was built under R version 3.6.3
```

```
## Warning: package 'iterators' was built under R version 3.6.3
```

```r
cl <- makeCluster(detectCores())
registerDoParallel(cl)

# begin training
set.seed(42)
train.ctrl <- trainControl(method = "cv", number = 10, search = "grid")
train.rf <- train(x.new.train, y.train, trControl = train.ctrl, method = "rf")

# stop multithreading
stopCluster(cl)
```

We the show the summary of the training procedure:

```r
print(train.rf)
```

```
## Random Forest 
## 
## 15699 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 14130, 14128, 14129, 14130, 14128, 14130, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9930572  0.9912172
##   27    0.9938215  0.9921848
##   52    0.9858586  0.9821103
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 27.
```

In training we performed a hyperparameter grid search in order to improve the
accuracy of the prediction. In fact we show the plot of such search:


```r
plot(train.rf, main="Accuracy over the Hyperparameter Search")
```

![](activity_performance_files/figure-html/unnamed-chunk-15-1.png)<!-- -->

We finally plot the feature importance assigned by the algorithm to check that
our analysis (including the exploratory analysis and dataset cleaning) was in
fact meaningful:


```r
var.imp <- varImp(train.rf)
plot(var.imp, main = "Variable Ranking (random forest)")
```

![](activity_performance_files/figure-html/unnamed-chunk-16-1.png)<!-- -->

Finally we show the training performance:


```r
train.pred <- predict(train.rf, newdata = x.new.train)
confusionMatrix(data = train.pred, reference = y.train, mode = "everything")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4461    6    0    3    0
##          B    3 3024   20    4    0
##          C    0    0 2696   18    1
##          D    0    8   22 2548    1
##          E    0    0    0    0 2884
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9945          
##                  95% CI : (0.9932, 0.9956)
##     No Information Rate : 0.2843          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9931          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9954   0.9847   0.9903   0.9993
## Specificity            0.9992   0.9979   0.9985   0.9976   1.0000
## Pos Pred Value         0.9980   0.9912   0.9930   0.9880   1.0000
## Neg Pred Value         0.9997   0.9989   0.9968   0.9981   0.9998
## Precision              0.9980   0.9912   0.9930   0.9880   1.0000
## Recall                 0.9993   0.9954   0.9847   0.9903   0.9993
## F1                     0.9987   0.9933   0.9888   0.9891   0.9997
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1926   0.1717   0.1623   0.1837
## Detection Prevalence   0.2847   0.1943   0.1729   0.1643   0.1837
## Balanced Accuracy      0.9993   0.9966   0.9916   0.9940   0.9997
```

and for the test data:


```r
test.pred <- predict(train.rf, newdata = x.new.test)
confusionMatrix(data = test.pred, reference = y.test, mode = "everything")
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1114    3    0    0    0
##          B    1  755    7    0    0
##          C    0    0  674    7    1
##          D    0    1    3  636    1
##          E    1    0    0    0  719
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9936          
##                  95% CI : (0.9906, 0.9959)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9919          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9947   0.9854   0.9891   0.9972
## Specificity            0.9989   0.9975   0.9975   0.9985   0.9997
## Pos Pred Value         0.9973   0.9895   0.9883   0.9922   0.9986
## Neg Pred Value         0.9993   0.9987   0.9969   0.9979   0.9994
## Precision              0.9973   0.9895   0.9883   0.9922   0.9986
## Recall                 0.9982   0.9947   0.9854   0.9891   0.9972
## F1                     0.9978   0.9921   0.9868   0.9907   0.9979
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2840   0.1925   0.1718   0.1621   0.1833
## Detection Prevalence   0.2847   0.1945   0.1738   0.1634   0.1835
## Balanced Accuracy      0.9986   0.9961   0.9915   0.9938   0.9985
```

We finally plot the distribution of the **validation set** and compare the
predictions:


```r
library(dplyr)
library(data.table)
test.pred <- data.table(test.pred)
test.pred.n <- test.pred %>% count(test.pred)
y.test <- data.table(y.test)
y.test.n <- y.test %>% count(y.test)

# plot the counts
g <- ggplot() +
     geom_bar(data = test.pred , aes(test.pred, fill = test.pred), width = 0.75) +
     geom_bar(data = y.test, aes(y.test), width = 0.3, fill = "white") +
     xlab("classe") +
     ylab("count") +
     ggtitle("Validation set predictions and true values")
print(g)
```

![\label{fig:pred}Predictions on the validation set: coloured bars represent true values and white superimposed values represent predictions.](activity_performance_files/figure-html/unnamed-chunk-19-1.png)

## Test Set Predictions

We finally consider the final test set:


```r
final.test <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                 na.strings = c("#DIV/0!", "", "NA"),
                 stringsAsFactors = FALSE)

# perform the same transformations
final.test <- final.test[,-(1:7)]
final.test$problem_id <- as.factor(final.test$problem_id)
test.data.na.fraction <- sapply(final.test, function(x) {mean(is.na(x))})
test.data.cols <- (test.data.na.fraction <= 0.5)

# select the relevant columns
test.data <- final.test[, test.data.cols]
test.data <- test.data[,-ncol(test.data)]

# apply pre-process transformation
test.data <- predict(preprocess, newdata = test.data)
```

We can now make the test set predictions:


```r
final.predictions <- predict(train.rf, newdata = test.data)
print(final.predictions)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

## Conclusions

We showed that we were able to make meaningful predictions on a cleaned version
of the wearable database, achieving a very high level of accuracy (more than 99%
of accuracy has been reached) using a random forest approach.
