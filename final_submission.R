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
