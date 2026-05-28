

setwd('~/Downloads/ML_AI')

library(psych)
library(ggplot2)
library(dplyr)
library(Amelia)
library(corrplot)
library(scales)

# ============================================
# Load Data

Financial_Transactions <- read.csv('Financial_Transactions.csv')


# EDA - Data Structure and Summary

str(Financial_Transactions)
describe(Financial_Transactions)


# Create Working Copy

FT <- Financial_Transactions

# Convert empty strings to NA
FT$Amount.Category[FT$Amount.Category == ""] <- NA
FT$type[FT$type == ""] <- NA

# Convert Fraud to factor with labels
FT$isFraud <- factor(FT$isFraud,
                     levels = c(0, 1),
                     labels = c("NonFraud", "Fraud"))


# Missing Values Analysis


# Count missing values
missing_counts <- colSums(is.na(FT))
missing_percent <- colMeans(is.na(FT)) * 100

missing_counts
missing_percent

# Missingness plot

missmap(FT)


# Remove missing values
FT <- na.omit(FT)


# Bivariate Analysis

# Transaction type vs isFraud
p_type <- ggplot(FT, aes(x = type, fill = isFraud)) +
  geom_bar(position = "fill") +
  labs(
    title = "Proportion of Fraud by Transaction Type",
    x = "Transaction Type",
    y = "Proportion",
    fill = "Fraud"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("type_fraud.png", p_type, width = 8, height = 5)


# Transaction type vs amount 

p_amount <-  ggplot(FT, aes(x = isFraud, y = amount)) +
  geom_boxplot(outlier.alpha = 0.1, fill = "lightblue") +
  scale_y_log10(labels = scales::comma) +
  labs(
    title = "Transaction Amount by Fraud Status",
    x = "Fraud Status",
    y = "Amount (log scale)"
  )

ggsave("amount_fraud.png", p_amount, width = 8, height = 5)

# Fraud by amount category

p_cat <- ggplot(FT, aes(x = Amount.Category, fill = isFraud)) +
  geom_bar(position = "fill") +
  labs(
    title = "Proportion of Fraud by Amount Category",
    x = "Amount Category",
    y = "Proportion",
    fill = "Fraud"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
ggsave("amount_category.png", width = 8, height = 5)


#  Outlier Detection & Visualization



Q1 <- quantile(FT$amount, 0.25)
Q3 <- quantile(FT$amount, 0.75)
IQR_val <- Q3 - Q1

lower_bound <- Q1 - 1.5 * IQR_val
upper_bound <- Q3 + 1.5 * IQR_val

outliers <- FT$amount < lower_bound | FT$amount > upper_bound
sum(outliers)

# (c) Multicollinearity Analysis

num_vars <- FT %>%
  select(amount,
         oldbalanceOrg,
         newbalanceOrig,
         oldbalanceDest,
         newbalanceDest)

cor_matrix <- cor(num_vars)

png("correlation_plot.png", width = 800, height = 600)
corrplot(cor_matrix, method = "color")
dev.off()



# Remove identifier variables
FT_model <- FT %>%
  select(-nameOrig, -nameDest)


# Encoding & Scaling


# Convert categorical to factor
FT_model$type <- as.factor(FT_model$type)
FT_model$Amount.Category <- as.factor(FT_model$Amount.Category)
FT_model$isFraud <- as.factor(FT_model$isFraud)

# Scale numerical variables
num_cols <- sapply(FT_model, is.numeric)
FT_model[num_cols] <- scale(FT_model[num_cols])


# Final Dataset for Modelling


str(FT_model)
summary(FT_model)


# MODELLING SECTION


library(caret)
library(pROC)
library(e1071)      # Naive Bayes
library(rpart)      # Decision Tree
library(randomForest)


# Train/Test Split


set.seed(123)

trainIndex <- createDataPartition(FT_model$isFraud, p = 0.7, list = FALSE)

train_data <- FT_model[trainIndex, ]
test_data  <- FT_model[-trainIndex, ]


# Cross-Validation Setup


ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# 1. Logistic Regression


model_lr <- train(
  isFraud ~ .,
  data = train_data,
  method = "glm",
  family = "binomial",
  trControl = ctrl,
  metric = "ROC"
)

# ============================================
# 2. K-Nearest Neighbours
# ============================================

model_knn <- train(
  isFraud ~ .,
  data = train_data,
  method = "knn",
  trControl = ctrl,
  tuneLength = 5,
  metric = "ROC"
)

# ============================================
# 3. Naïve Bayes
# ============================================

model_nb <- train(
  isFraud ~ .,
  data = train_data,
  method = "nb",
  trControl = ctrl,
  metric = "ROC"
)


# 4. Decision Tree


model_dt <- train(
  isFraud ~ .,
  data = train_data,
  method = "rpart",
  trControl = ctrl,
  tuneLength = 5,
  metric = "ROC"
)


# 5. Random Forest


model_rf <- train(
  isFraud ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  tuneLength = 5,
  metric = "ROC"
)

# Compare Models (CV Results)


results <- resamples(
  list(
    Logistic_Regression = model_lr,
    KNN = model_knn,
    Naive_Bayes = model_nb,
    Decision_Tree = model_dt,
    Random_Forest = model_rf
  )
)

summary(results)
bwplot(results, metric = "ROC")

# TEST SET EVALUATION

# Function to evaluate models
evaluate_model <- function(model, test_data) {
  probs <- predict(model, test_data, type = "prob")[, 2]
  preds <- predict(model, test_data)

  cm <- confusionMatrix(preds, test_data$isFraud)

  roc_obj <- roc(test_data$isFraud, probs)
  auc_val <- auc(roc_obj)

  return(list(
    confusion = cm,
    AUC = auc_val,
    ROC = roc_obj
  ))
}

# Evaluate all models
eval_lr  <- evaluate_model(model_lr, test_data)
eval_knn <- evaluate_model(model_knn, test_data)
eval_nb  <- evaluate_model(model_nb, test_data)
eval_dt  <- evaluate_model(model_dt, test_data)
eval_rf  <- evaluate_model(model_rf, test_data)


# AUC Comparison Table


model_performance <- data.frame(
  Model = c(
    "Logistic Regression",
    "KNN",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest"
  ),

  AUC = c(
    eval_lr$AUC,
    eval_knn$AUC,
    eval_nb$AUC,
    eval_dt$AUC,
    eval_rf$AUC
  )
)

print(model_performance)


# Plot ROC Curves


plot(eval_lr$ROC, col = "blue", main = "ROC Curves")
lines(eval_knn$ROC, col = "red")
lines(eval_nb$ROC, col = "green")
lines(eval_dt$ROC, col = "purple")
lines(eval_rf$ROC, col = "black")

legend(
  "bottomright",
  legend = c("Logistic", "KNN", "NB", "DT", "RF"),
  col = c("blue", "red", "green", "purple", "black"),
  lwd = 2
)
