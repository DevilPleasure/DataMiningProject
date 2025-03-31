library(ggcorrplot)
library(ggplot2)
library(devtools)
library(FactoMineR)
library(factoextra)
library(caret)  
library(pROC)
library(randomForest)
library(e1071)
library(dplyr)


if (!dir.exists("Graph")) dir.create("Graph")

save_plot <- function(filename, plot_expr) {
  png(file.path("Graph", filename), width = 800, height = 600)
  plot_expr()
  dev.off()
}

#Load the dataset and erase columns that won't be useful.
datas = read.csv2("./high_diamond_ranked_10min.csv", sep = ',', dec = '.')
datas$gameId = NULL
datas$redGoldDiff = NULL
datas$redExperienceDiff = NULL
datas$redCSPerMin = NULL
datas$blueCSPerMin = NULL
datas$blueGoldPerMin = NULL
datas$redGoldPerMin = NULL
datas$redFirstBlood = NULL

print(summary(datas))

#Impact of the vision
save_plot("Blue_ward_Win.png", function() {
  plot(datas$blueWins, datas$blueWardsPlaced,
       main = "Relation entre blueWins et blueWardsPlaced",
       xlab = "blueWins", ylab = "blueWardsPlaced", col = "blue")
})

save_plot("Boxplot_wards.png", function() {
  boxplot(datas$blueWardsPlaced, datas$redWardsPlaced,
          names = c('Blue wards', 'Red Wards'),
          col = c("blue", "red"),
          main = "Comparaison des wards placés")
})

# Corelation matrix
matrix_cor = cor(datas)
corr_plot <- ggcorrplot(matrix_cor, type = "upper", show.diag = TRUE, lab = TRUE, 
                        hc.order = TRUE, lab_size = 2, 
                        ggtheme = ggplot2::theme_minimal()) +
  theme(axis.text.x = element_text(size = 8, angle = 90, vjust = 1, hjust = 1),
        axis.text.y = element_text(size = 8)) + coord_fixed()

ggsave("Graph/Corr_matrix.png", plot = corr_plot, width = 10, height = 8, dpi = 300, bg = "white")

#PCA
features <- datas[, !(names(datas) %in% c("blueWins"))]
features_scaled <- scale(features)
target <- as.factor(datas$blueWins)

pca = PCA(features_scaled, graph = TRUE)

pca_plot <- fviz_pca_var(pca, col.var = "contrib", 
             gradient.cols = c('grey',"cyan",'green', "yellow", "orange", 'red'),
             repel = TRUE, max.overlaps = 20) +
  ggtitle("Importance of each variable to the PCA")
ggsave("Graph/PCA_variable_importance.png", plot = pca_plot, width = 10, height = 8, dpi = 300, bg="white")


ind_plot <- fviz_pca_ind(pca,
                         geom.ind = "point",
                         col.ind = target,
                         palette = c("red", "blue"),
                         addEllipses = TRUE,
                         legend.title = "blueWins") +
  ggtitle("PCA of the data")

ggsave("Graph/PCA_projection_individus.png", plot = ind_plot, width = 10, height = 8, dpi = 300, bg="white")


print(pca$var$contrib)

important_vars <- as.data.frame(pca$var$contrib)
important_vars$Variable <- rownames(important_vars)

important_vars <- important_vars[order(-important_vars$Dim.1), ]
print(important_vars)  


#Logistic regression with pca

eig_values <- pca$eig  
cumsum_variance <- cumsum(eig_values[,2])  
num_total_pc <- ncol(pca$ind$coord)
num_pc <- min(which(cumsum_variance >= 80), num_total_pc)  #min value to have cumsum >= 80%


pca_data <- as.data.frame(pca$ind$coord[, 1:num_pc])  
pca_data$blueWins <- target  

# Split 80/20
set.seed(123)
trainIndex <- createDataPartition(pca_data$blueWins, p = 0.8, list = FALSE)
train_data <- pca_data[trainIndex, ]
test_data  <- pca_data[-trainIndex, ]

#Prediction
model <- glm(blueWins ~ ., data = train_data, family = binomial)

predictions <- predict(model, newdata = test_data, type = "response")

predicted_classes <- ifelse(predictions > 0.5, 1, 0)


conf_matrix <- confusionMatrix(as.factor(predicted_classes), test_data$blueWins, positive = "1")
print(conf_matrix)

#ROC Curve
roc_curve <- roc(as.numeric(as.character(test_data$blueWins)), predictions)


save_plot("roc_logistic.png", function() {
  plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve on logistic regression")
  legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 3)), col = "blue", lwd = 2)
})


#Random Forest Model
set.seed(123)
trainIndex <- createDataPartition(target, p = 0.8, list = FALSE)
train_data <- datas[trainIndex, ]
test_data  <- datas[-trainIndex, ]
rf_model <- randomForest(as.factor(blueWins) ~ ., data = train_data, ntree = 500, mtry = sqrt(ncol(features)))

predictions <- predict(rf_model, newdata = test_data, type = "response")

conf_matrix <- confusionMatrix(predictions, as.factor(test_data$blueWins), positive = "1")
print(conf_matrix)

prob_predictions <- predict(rf_model, newdata = test_data, type = "prob")[,2]
roc_curve <- roc(as.numeric(as.character(test_data$blueWins)), prob_predictions)
auc_rf = auc(roc_curve)
save_plot("roc_random_forest.png", function() {
  plot(roc_curve, col = "lightgreen", lwd = 2, main = "Courbe ROC - Random Forest")
  legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 3)), col = "lightgreen", lwd = 2)
})



importance(rf_model)
varImpPlot(rf_model)




#Tuning of hyperparameters

set.seed(123)
trainIndex <- createDataPartition(target, p = 0.8, list = FALSE)
train_data <- datas[trainIndex, ]
test_data  <- datas[-trainIndex, ]

#Labels for caret (1 for win, 0 for lose)
train_data$blueWins <- factor(ifelse(train_data$blueWins == 1, "Win", "Lose"))
test_data$blueWins <- factor(ifelse(test_data$blueWins == 1, "Win", "Lose"))

# Training only on mtry 
mtry_vals <- floor(sqrt(ncol(train_data) - 1))  
tune_grid <- expand.grid(mtry = c(mtry_vals - 2, mtry_vals, mtry_vals + 2, mtry_vals + 4))

# Parameters for the tuning
control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)


set.seed(123)
rf_tuned <- train(
  blueWins ~ .,
  data = train_data,
  method = "rf",
  metric = "ROC",
  trControl = control,
  tuneGrid = tune_grid,
  ntree = 500
)


print(rf_tuned)
plot(rf_tuned)

# predictions on test
predictions <- predict(rf_tuned, newdata = test_data)
conf_matrix <- confusionMatrix(predictions, test_data$blueWins, positive = "Win")
print(conf_matrix)

prob_predictions <- predict(rf_tuned, newdata = test_data, type = "prob")[, "Win"]
roc_curve <- roc(test_data$blueWins, prob_predictions)
auc_rf = auc(roc_curve)
save_plot("roc_rf_tuned.png", function() {
  plot(roc_curve, col = "orange", lwd = 2, main = "ROC Curve - Tuned Random Forest")
  legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 3)), col = "orange", lwd = 2)
})


#SVM
set.seed(123)
trainIndex <- createDataPartition(target, p = 0.8, list = FALSE)
train_features <- features[trainIndex, ]
test_features  <- features[-trainIndex, ]
train_target   <- factor(ifelse(datas$blueWins[trainIndex] == 1, "Win", "Lose"))
test_target    <- factor(ifelse(datas$blueWins[-trainIndex] == 1, "Win", "Lose"))

#Pre-processing
preProc <- preProcess(train_features, method = c("center", "scale"))
train_features_scaled <- predict(preProc, train_features)
test_features_scaled  <- predict(preProc, test_features)

#Dataset final
train_data <- cbind(train_features_scaled, blueWins = train_target)
test_data  <- cbind(test_features_scaled, blueWins = test_target)


control <- trainControl(method = "cv", number = 5,
                        classProbs = TRUE, summaryFunction = twoClassSummary, verboseIter = TRUE) #Show the progression 

### ---------- 1️⃣ SVM LINEAR ----------
set.seed(123)
svm_linear <- train(blueWins ~ ., data = train_data,
                    method = "svmLinear",
                    trControl = control,
                    preProcess = NULL,
                    tuneLength = 5,
                    metric = "ROC")

print("\n===== SVM Linear =====\n")
print(svm_linear)

pred_linear <- predict(svm_linear, newdata = test_data)
conf_matrix_linear <- confusionMatrix(pred_linear, test_data$blueWins, positive = "Win")
print(conf_matrix_linear)
prob_linear <- predict(svm_linear, newdata = test_data, type = "prob")[, "Win"]
roc_linear <- roc(test_data$blueWins, prob_linear)
cat("AUC SVM Linear :", auc(roc_linear), "\n")


### ---------- 2️⃣ SVM RBF ----------
set.seed(123)
svm_rbf <- train(blueWins ~ ., data = train_data,
                 method = "svmRadial",
                 trControl = control,
                 preProcess = NULL,
                 tuneLength = 5,
                 metric = "ROC")

print("\n===== SVM RBF =====\n")
print(svm_rbf)

pred_rbf <- predict(svm_rbf, newdata = test_data)
conf_matrix_rbf <- confusionMatrix(pred_rbf, test_data$blueWins, positive = "Win")
print(conf_matrix_rbf)
prob_rbf <- predict(svm_rbf, newdata = test_data, type = "prob")[, "Win"]
roc_rbf <- roc(test_data$blueWins, prob_rbf)
cat("AUC SVM RBF :", auc(roc_rbf), "\n")


### ---------- 3️⃣ SVM POLYNOMIAL ----------
set.seed(123)
grid_poly <- expand.grid(
  degree = c(2, 3),
  scale = 1,
  C = c(0.25, 0.5, 1)
)

control <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                        summaryFunction = twoClassSummary,
                        verboseIter = TRUE)  

svm_poly <- train(blueWins ~ ., data = train_data,
                  method = "svmPoly",
                  trControl = control,
                  preProcess = NULL,
                  tuneGrid = grid_poly,
                  metric = "ROC")

print("\n===== SVM Polynomial =====\n")
print(svm_poly)

pred_poly <- predict(svm_poly, newdata = test_data)
conf_matrix_poly <- confusionMatrix(pred_poly, test_data$blueWins, positive = "Win")
print(conf_matrix_poly)
prob_poly <- predict(svm_poly, newdata = test_data, type = "prob")[, "Win"]
roc_poly <- roc(test_data$blueWins, prob_poly)
cat("AUC SVM Poly :", auc(roc_poly), "\n")

### ---------- 4️⃣ SVM SIGMOID (e1071) ----------


set.seed(123)
svm_sigmoid <- svm(as.factor(blueWins) ~ ., data = train_data,
                   kernel = "sigmoid", probability = TRUE)

pred_sigmoid <- predict(svm_sigmoid, newdata = test_data, probability = TRUE)
conf_matrix_sigmoid <- confusionMatrix(pred_sigmoid, test_data$blueWins, positive = "Win")
print("\n===== SVM Sigmoid =====\n")
print(conf_matrix_sigmoid)

prob_sigmoid <- attr(predict(svm_sigmoid, newdata = test_data, probability = TRUE), "probabilities")[, "Win"]
roc_sigmoid <- roc(test_data$blueWins, prob_sigmoid)
cat("AUC SVM Sigmoid :", auc(roc_sigmoid), "\n")


param_grid <- expand.grid(
  C = c(0.1, 1, 10),
  gamma = c(0.01, 0.1, 1),
  coef0 = c(0, 1)
)

evaluate_svm <- function(params) {
  C_val <- as.numeric(params$C)
  gamma_val <- as.numeric(params$gamma)
  coef0_val <- as.numeric(params$coef0)
  
  cat("Testing: C =", C_val, "gamma =", gamma_val, "coef0 =", coef0_val, "\n")
  
  svm_model <- svm(as.factor(blueWins) ~ ., data = train_data,
                   kernel = "sigmoid",
                   cost = C_val,
                   gamma = gamma_val,
                   coef0 = coef0_val,
                   probability = TRUE)
  
  pred <- predict(svm_model, newdata = test_data, probability = TRUE)
  conf_matrix <- confusionMatrix(pred, test_data$blueWins, positive = "Win")
  
  prob_pred <- attr(predict(svm_model, newdata = test_data, probability = TRUE), "probabilities")[, "Win"]
  roc_curve <- roc(test_data$blueWins, prob_pred, quiet = TRUE)
  auc_val <- as.numeric(auc(roc_curve))
  
  return(data.frame(
    C = C_val,
    gamma = gamma_val,
    coef0 = coef0_val,
    Balanced_Accuracy = conf_matrix$byClass["Balanced Accuracy"],
    AUC = auc_val
  ))
}
grid_results <- param_grid %>%
  split(1:nrow(param_grid)) %>%
  lapply(evaluate_svm) %>%
  bind_rows()

grid_results <- grid_results %>% arrange(desc(AUC))
print(grid_results)





best_sigmoid <- grid_results %>% slice_max(AUC, n = 1)

svm_sigmoid_best <- svm(as.factor(blueWins) ~ ., data = train_data,
                        kernel = "sigmoid",
                        cost = best_sigmoid$C,
                        gamma = best_sigmoid$gamma,
                        coef0 = best_sigmoid$coef0,
                        probability = TRUE)

prob_sigmoid_best <- attr(predict(svm_sigmoid_best, newdata = test_data, probability = TRUE), "probabilities")[, "Win"]
roc_sigmoid_best <- roc(test_data$blueWins, prob_sigmoid_best, quiet = TRUE)


save_plot("roc_svm_comparatif.png", function() {
  plot(roc_linear, col = "orange", lwd = 2, main = "ROC Curves- Best SVM ")
  plot(roc_rbf, col = "blue", lwd = 2, add = TRUE)
  plot(roc_poly, col = "green", lwd = 2, add = TRUE)
  plot(roc_sigmoid_best, col = "purple", lwd = 2, add = TRUE)
  abline(a = 0, b = 1, col = "gray", lty = 2)
  legend("bottomright",
         legend = c(
           paste("Linear (AUC:", round(auc(roc_linear), 3), ")"),
           paste("RBF (AUC:", round(auc(roc_rbf), 3), ")"),
           paste("Polynomial (AUC:", round(auc(roc_poly), 3), ")"),
           paste("Sigmoid (AUC:", round(auc(roc_sigmoid_best), 3), ")")
         ),
         col = c("orange", "blue", "green", "purple"),
         lwd = 2)
})




# kNN
set.seed(123)
control <- trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary)

tune_grid_knn <- expand.grid(k = seq(3, 21, 2))

knn_model <- train(
  x = features_scaled,
  y = factor(ifelse(datas$blueWins == 1, "Win", "Lose")),
  method = "knn",
  trControl = control,
  tuneGrid = tune_grid_knn,
  metric = "ROC"
)

print(knn_model)


set.seed(123)
trainIndex <- createDataPartition(datas$blueWins, p = 0.8, list = FALSE)
x_train <- features_scaled[trainIndex, ]
y_train <- factor(ifelse(datas$blueWins[trainIndex] == 1, "Win", "Lose"))
x_test <- features_scaled[-trainIndex, ]
y_test <- factor(ifelse(datas$blueWins[-trainIndex] == 1, "Win", "Lose"))

knn_final <- train(x_train, y_train,
                   method = "knn",
                   trControl = control,
                   tuneGrid = data.frame(k = knn_model$bestTune$k),
                   metric = "ROC")

pred_knn <- predict(knn_final, x_test)
prob_knn <- predict(knn_final, x_test, type = "prob")[, "Win"]
conf_matrix_knn <- confusionMatrix(pred_knn, y_test, positive = "Win")
print(conf_matrix_knn)

roc_knn <- roc(y_test, prob_knn)
save_plot("roc_knn.png", function() {
  plot(roc_knn, col = "lightblue", lwd = 2, main = "ROC Curve - kNN")
  legend("bottomright", legend = paste("AUC =", round(auc(roc_knn), 3)), col = "lightblue", lwd = 2)
})



best_params <- data.frame(
  Model = c("SVM Linear", "SVM RBF", "SVM Polynomial", "SVM Sigmoid", "Random Forest Tuned", "kNN"),
  Parameters = c(
    paste("C =", svm_linear$bestTune$C),
    paste("sigma =", round(svm_rbf$bestTune$sigma, 5), ", C =", svm_rbf$bestTune$C),
    paste("degree =", svm_poly$bestTune$degree, ", scale =", svm_poly$bestTune$scale, ", C =", svm_poly$bestTune$C),
    paste("C =", best_sigmoid$C, ", gamma =", best_sigmoid$gamma, ", coef0 =", best_sigmoid$coef0),
    paste("mtry =", rf_tuned$bestTune$mtry),
    paste("k =", knn_model$bestTune$k)
  ),
  AUC = c(
    round(auc(roc_linear), 4),
    round(auc(roc_rbf), 4),
    round(auc(roc_poly), 4),
    round(auc(roc_sigmoid_best), 4),
    round(auc_rf, 4),
    round(auc(roc_knn), 4)
  ),
  Balanced_Accuracy = c(
    conf_matrix_linear$byClass["Balanced Accuracy"],
    conf_matrix_rbf$byClass["Balanced Accuracy"],
    conf_matrix_poly$byClass["Balanced Accuracy"],
    conf_matrix_sigmoid$byClass["Balanced Accuracy"],
    conf_matrix$byClass["Balanced Accuracy"],
    conf_matrix_knn$byClass["Balanced Accuracy"]
  )
)

# Order by best AUC
best_params <- best_params[order(-best_params$AUC), ]

print(best_params)
write.csv2(best_params, "Graph/Best_Model_Parameters.csv", row.names = FALSE)
