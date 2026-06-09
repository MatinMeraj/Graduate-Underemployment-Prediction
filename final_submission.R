## =========================
## XGBoost pipeline (binary AUC) for this competition
## Paste into an R script (or an R chunk in your .qmd)
## Assumes train.csv and test.csv are in your working directory
## =========================

# Packages
pkgs <- c("tidyverse", "Matrix", "xgboost", "pROC", "forcats")
to_install <- pkgs[!pkgs %in% installed.packages()[, "Package"]]
if (length(to_install)) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

# Load data
train_raw <- readr::read_csv("train.csv", show_col_types = FALSE)
test_raw  <- readr::read_csv("test.csv", show_col_types = FALSE)

# --- Confirm the target column is present ---
target_col <- "overqualified"
if (!(target_col %in% names(train_raw))) {
  stop("Target column '", target_col, "' not found in train.csv.")
}

# --- Basic checks ---
if (!("id" %in% names(train_raw)) || !("id" %in% names(test_raw))) {
  stop("Column 'id' not found in train/test.")
}

# Make all predictors factors and encode missing as an explicit level "Missing"
make_missing_factor <- function(df) {
  df %>%
    mutate(across(everything(), ~ as.factor(.))) %>%
    mutate(across(everything(), ~ forcats::fct_explicit_na(.x, na_level = "Missing")))
}

train_raw <- make_missing_factor(train_raw)
test_raw  <- make_missing_factor(test_raw)

# Separate target and features
y <- as.integer(as.character(train_raw[[target_col]]))  # should be 0/1
if (any(is.na(y))) stop("Target contains NA after conversion. Check target coding.")
if (!all(y %in% c(0L, 1L))) stop("Target must be 0/1. Found values: ", paste(sort(unique(y)), collapse=", "))

train_x <- train_raw %>% select(-all_of(target_col), -id)
test_x  <- test_raw  %>% select(-id)

# One-hot encode with identical columns for train and test
all_x <- bind_rows(train_x, test_x)
X_all <- sparse.model.matrix(~ . - 1, data = all_x)

n_train <- nrow(train_x)
X_train <- X_all[1:n_train, ]
X_test  <- X_all[(n_train + 1):nrow(X_all), ]

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = X_train, label = y)

# XGBoost parameters (strong baseline)
set.seed(42)
ratio <- sum(y == 0) / sum(y == 1)
ratio

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.06,
  max_depth = 8,
  min_child_weight = 2,
  subsample = 0.9,
  colsample_bytree = 0.9,
  # Classes are mildly imbalanced (~1.6 negatives per positive).
  # scale_pos_weight up-weights the positive ("overqualified") class.
  # We use 0.8 * ratio rather than the full ratio so the correction is
  # gentle -- the full ratio tended to over-predict the positive class.
  scale_pos_weight = 0.8 * ratio
)

# Cross-validation to find best number of rounds (like gbm.perf)
cv <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 6000,
  nfold = 5,
  early_stopping_rounds = 100,
  maximize = TRUE,
  verbose = 1
)


# Extract best iteration safely
if (!is.null(cv$best_iteration)) {
  best_nrounds <- cv$best_iteration
} else {
  best_nrounds <- which.max(cv$evaluation_log$test_auc_mean)
}

cat("Best nrounds:", best_nrounds, "\n")
cat("Best CV AUC:", max(cv$evaluation_log$test_auc_mean), "\n")


# Train final model
final_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)

# Feature importance (top 20)
imp <- xgb.importance(model = final_model)
print(head(imp, 20))
# Optional plot:
xgb.plot.importance(imp, top_n = 20)

# Diagnostic: training AUC (not your real score, just sanity check)
train_pred <- predict(final_model, dtrain)
train_auc <- auc(roc(response = y, predictor = train_pred))
cat("Training AUC (diagnostic):", as.numeric(train_auc), "\n")

# Predict test
dtest <- xgb.DMatrix(data = X_test)
test_pred <- predict(final_model, dtest)

submission <- tibble(
  id = test_raw$id,
  overqualified = test_pred
)

# Sanity check: predicted probabilities should spread across [0, 1],
# not collapse to all-0 or all-1.
hist(
  submission$overqualified,
  breaks = 30,
  main = "Predicted Underemployment Probabilities",
  xlab = "Probability"
)

readr::write_csv(submission, "submission.csv")


## ============================================================
## HONEST EVALUATION
## ------------------------------------------------------------
## The model above (final_model) is trained on ALL training rows
## and used to produce the competition submission. We canNOT
## measure its real performance on test.csv because test.csv has
## no labels. So below we train a SEPARATE model on an 80% slice
## and evaluate it on the held-out 20% it never saw -- these are
## the numbers we can actually defend (AUC, confusion matrix,
## precision / recall / F1).
## ============================================================

set.seed(42)

# ---- 80/20 stratified held-out split on the training rows ----
# (stratified = keep the same ~38% positive rate in both halves)
pos_idx <- which(y == 1)
neg_idx <- which(y == 0)
val_pos <- sample(pos_idx, round(0.20 * length(pos_idx)))
val_neg <- sample(neg_idx, round(0.20 * length(neg_idx)))
val_rows <- c(val_pos, val_neg)

X_tr_eval <- X_train[-val_rows, ]
X_val     <- X_train[ val_rows, ]
y_tr_eval <- y[-val_rows]
y_val     <- y[ val_rows]

dtrain_eval <- xgb.DMatrix(data = X_tr_eval, label = y_tr_eval)
dval        <- xgb.DMatrix(data = X_val,     label = y_val)

# Train on the 80% only, using the same params + best_nrounds found above
model_eval <- xgb.train(
  params  = params,
  data    = dtrain_eval,
  nrounds = best_nrounds,
  verbose = 0
)

# ---- Held-out probabilities and AUC ----
val_pred <- predict(model_eval, dval)
val_auc  <- as.numeric(auc(roc(response = y_val, predictor = val_pred)))
cat("\n==== HELD-OUT EVALUATION (20% never seen in training) ====\n")
cat(sprintf("Held-out AUC: %.3f\n", val_auc))

# ---- Confusion matrix + precision / recall / F1 at threshold 0.5 ----
thr    <- 0.5
pred01 <- as.integer(val_pred >= thr)

TP <- sum(pred01 == 1 & y_val == 1)
TN <- sum(pred01 == 0 & y_val == 0)
FP <- sum(pred01 == 1 & y_val == 0)
FN <- sum(pred01 == 0 & y_val == 1)

precision <- TP / (TP + FP)
recall    <- TP / (TP + FN)
f1        <- 2 * precision * recall / (precision + recall)
accuracy  <- (TP + TN) / length(y_val)

cat(sprintf("\nConfusion matrix @ threshold %.2f\n", thr))
cat("            pred=0   pred=1\n")
cat(sprintf("actual=0  %7d  %7d\n", TN, FP))
cat(sprintf("actual=1  %7d  %7d\n", FN, TP))
cat(sprintf("\nAccuracy:  %.3f\n", accuracy))
cat(sprintf("Precision: %.3f\n", precision))
cat(sprintf("Recall:    %.3f\n", recall))
cat(sprintf("F1:        %.3f\n", f1))

# ---- ROC curve plot (save to file for the README) ----
roc_obj <- roc(response = y_val, predictor = val_pred)
png("roc_curve.png", width = 900, height = 900, res = 150)
plot(roc_obj, col = "#c9743d", lwd = 2.5,
     main = "ROC Curve - Held-out Validation Set",
     legacy.axes = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray50")
legend("bottomright",
       legend = sprintf("XGBoost (AUC = %.3f)", val_auc),
       col = "#c9743d", lwd = 2.5, bty = "n")
dev.off()
cat("\nSaved roc_curve.png\n")
