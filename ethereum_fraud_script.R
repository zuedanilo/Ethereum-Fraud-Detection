# Ethereum Fraud Detection Capstone Script
# Author: edX Capstone Project
# Purpose: Reproducible fraud detection workflow for the Kaggle Ethereum
# transaction_dataset.csv using tidymodels, SMOTE, Random Forest, and XGBoost.

# ---------------------------------------------------------------------------
# 1. Package installation and loading
# ---------------------------------------------------------------------------
options(repos = c(CRAN = "https://cloud.r-project.org"))

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(tidymodels)) install.packages("tidymodels")
if(!require(themis)) install.packages("themis")
if(!require(ranger)) install.packages("ranger")
if(!require(xgboost)) install.packages("xgboost")
if(!require(pROC)) install.packages("pROC")
if(!require(vip)) install.packages("vip")
if(!require(vroom)) install.packages("vroom")

library(tidyverse)
library(tidymodels)
library(themis)
library(ranger)
library(xgboost)
library(pROC)
library(vip)
library(vroom)

set.seed(2026)
theme_set(theme_minimal(base_size = 12))
tidymodels_prefer()

# ---------------------------------------------------------------------------
# 2. Relative project paths and automatic data download
# ---------------------------------------------------------------------------
dir.create("data", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)

data_path <- file.path("data", "transaction_dataset.csv")


# Public mirror of the Kaggle transaction_dataset.csv.
# If the URL is temporarily unavailable, the script falls back to the local
# data/transaction_dataset.csv file, which is recommended for the final hand-in.
dataset_url <- paste0(
  "https://raw.githubusercontent.com/Abhaykumar04/",
  "ETHEREUM_FRAUD_DETECTION/main/transaction_dataset.csv"
)

if (!file.exists(data_path)) {
  message("Downloading dataset from public URL...")
  tryCatch(
    download.file(dataset_url, destfile = data_path, mode = "wb", quiet = FALSE),
    error = function(e) {
      stop(
        "Automatic download failed and data/transaction_dataset.csv was not found. ",
        "Please verify the public URL or place the CSV in the data folder."
      )
    }
  )
} else {
  message("Using existing local dataset at: ", data_path)
}

# ---------------------------------------------------------------------------
# 3. Data loading and cleaning
# ---------------------------------------------------------------------------
clean_names <- function(x) {
  x <- trimws(x)
  x <- ifelse(is.na(x) | x == "", paste0("unnamed_", seq_along(x)), x)
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  x <- gsub("^_+|_+$", "", x)
  x <- ifelse(grepl("^[0-9]", x), paste0("x", x), x)
  make.unique(x, sep = "_")
}

raw_data <- vroom(
  file = data_path,
  delim = ",",
  na = c("", "NA", "N/A", "NULL", "null"),
  show_col_types = FALSE,
  progress = FALSE
)

eth_data <- raw_data %>%
  rename_with(clean_names)

irrelevant_cols <- names(eth_data)[
  str_detect(names(eth_data), "^x[0-9]+$|^index$|^address$|token_type$")
]

eth_data <- eth_data %>%
  select(-any_of(irrelevant_cols)) %>%
  mutate(
    flag = case_when(
      as.character(flag) %in% c("1", "Fraud", "fraud", "FRAUD") ~ "Fraud",
      TRUE ~ "Legit"
    ),
    flag = factor(flag, levels = c("Legit", "Fraud"))
  ) %>%
   mutate(across(where(is.character) & -flag, ~ readr::parse_number(as.character(.x)))) %>%
  select(flag, where(is.numeric))

write_csv(eth_data, file.path("outputs", "clean_ethereum_transactions.csv"))

message("Rows: ", nrow(eth_data), " | Columns after cleaning: ", ncol(eth_data))
message("Class distribution:")
print(count(eth_data, flag) %>% mutate(prop = n / sum(n)))

# ---------------------------------------------------------------------------
# 4. Exploratory Data Analysis
# ---------------------------------------------------------------------------
class_plot <- eth_data %>%
  count(flag) %>%
  mutate(prop = n / sum(n)) %>%
  ggplot(aes(x = flag, y = n, fill = flag)) +
  geom_col(width = 0.65, show.legend = FALSE) +
  geom_text(aes(label = scales::percent(prop, accuracy = 0.1)), vjust = -0.4) +
  scale_fill_manual(values = c("Legit" = "#2C7FB8", "Fraud" = "#D95F02")) +
  labs(
    title = "Ethereum Fraud Dataset Class Distribution",
    x = NULL,
    y = "Number of addresses"
  )

ggsave(file.path("figures", "class_distribution.png"), class_plot, width = 7, height = 5)

time_diff_col <- names(eth_data)[str_detect(names(eth_data), "time_diff.*first.*last")][1]

if (!is.na(time_diff_col)) {
  time_boxplot <- eth_data %>%
    ggplot(aes(x = flag, y = .data[[time_diff_col]], fill = flag)) +
    geom_boxplot(alpha = 0.75, outlier.alpha = 0.2, show.legend = FALSE) +
    scale_y_continuous(trans = "log1p", labels = scales::comma) +
    scale_fill_manual(values = c("Legit" = "#2C7FB8", "Fraud" = "#D95F02")) +
    labs(
      title = "Time Difference Between First and Last Transaction by Class",
      x = NULL,
      y = "Minutes, log1p scale"
    )

  ggsave(file.path("figures", "time_diff_boxplot.png"), time_boxplot, width = 7, height = 5)
}

numeric_for_cor <- eth_data %>%
  select(-flag) %>%
  select(where(is.numeric)) %>%
  select(where(~ sd(.x, na.rm = TRUE) > 0))

cor_matrix <- cor(numeric_for_cor, use = "pairwise.complete.obs")

top_cor_n <- min(20, ncol(cor_matrix))

top_cor_features <- tibble(
  feature = colnames(cor_matrix),
  mean_abs_cor = rowMeans(abs(cor_matrix), na.rm = TRUE)
) %>%
  slice_max(mean_abs_cor, n = top_cor_n, with_ties = FALSE) %>%
  pull(feature)

cor_plot_data <- cor_matrix[top_cor_features, top_cor_features] %>%
  as.data.frame() %>%
  rownames_to_column("feature_1") %>%
  pivot_longer(-feature_1, names_to = "feature_2", values_to = "correlation")

correlation_plot <- cor_plot_data %>%
  ggplot(aes(x = feature_1, y = feature_2, fill = correlation)) +
  geom_tile(color = "white", linewidth = 0.1) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B", limits = c(-1, 1)) +
  labs(
    title = "Correlation Matrix of Top Numeric Ethereum Features",
    x = NULL,
    y = NULL,
    fill = "r"
  ) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
        axis.text.y = element_text(size = 8))

ggsave(file.path("figures", "correlation_matrix.png"), correlation_plot, width = 9, height = 8)

# ---------------------------------------------------------------------------
# 5. Stratified 75/25 split and 5-fold cross-validation
# ---------------------------------------------------------------------------
# Report justification:
# A 75/25 split gives approximately 7,380 training observations and 2,461
# testing observations in this 9,841-row dataset. This leaves enough data for
# ensemble training while keeping a statistically meaningful fraud validation set.
data_split <- initial_split(eth_data, prop = 0.75, strata = flag)
train_data <- training(data_split)
test_data <- testing(data_split)

cv_folds <- vfold_cv(train_data, v = 5, strata = flag)

write_csv(count(train_data, flag), file.path("outputs", "train_class_distribution.csv"))
write_csv(count(test_data, flag), file.path("outputs", "test_class_distribution.csv"))

# ---------------------------------------------------------------------------
# 6. Feature engineering recipe
# ---------------------------------------------------------------------------
erc20_features <- names(train_data)[str_detect(names(train_data), "^erc20")]

candidate_log_features <- names(train_data)[
  str_detect(names(train_data), "ether|value|val|amount")
]
candidate_log_features <- setdiff(candidate_log_features, "flag")

# log(x + 1) is applied only to non-negative amount-like features. This prevents
# invalid values for balance variables that can be negative after net transfers.
log_features <- candidate_log_features[
  map_lgl(train_data[candidate_log_features], ~ min(.x, na.rm = TRUE) >= 0)
]

fraud_recipe <- recipe(flag ~ ., data = train_data)

if (length(erc20_features) > 0) {
  fraud_recipe <- fraud_recipe %>%
    step_mutate_at(starts_with("erc20"), fn = ~ tidyr::replace_na(.x, 0))
}

fraud_recipe <- fraud_recipe %>%
  step_impute_median(all_numeric_predictors())

if (length(log_features) > 0) {
  fraud_recipe <- fraud_recipe %>%
    step_log(!!!syms(log_features), offset = 1)
}

fraud_recipe <- fraud_recipe %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(flag, over_ratio = 1)

# ---------------------------------------------------------------------------
# 7. Model specifications: Random Forest and XGBoost
# ---------------------------------------------------------------------------
n_predictors <- ncol(train_data) - 1
mtry_value <- max(1, floor(sqrt(n_predictors)))

rf_spec <- rand_forest(
  trees = 500,
  mtry = mtry_value,
  min_n = 5
) %>%
  set_engine("ranger", importance = "permutation", probability = TRUE) %>%
  set_mode("classification")

xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = 4,
  learn_rate = 0.05,
  loss_reduction = 0.01,
  sample_size = 0.80,
  mtry = mtry_value,
  min_n = 10
) %>%
  set_engine("xgboost", objective = "binary:logistic", eval_metric = "aucpr") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(fraud_recipe) %>%
  add_model(rf_spec)

xgb_workflow <- workflow() %>%
  add_recipe(fraud_recipe) %>%
  add_model(xgb_spec)

fraud_metrics <- metric_set(recall, f_meas, pr_auc, roc_auc)

resample_control <- control_resamples(
  save_pred = TRUE,
  verbose = TRUE,
  allow_par = TRUE
)

message("Training Random Forest with 5-fold cross-validation...")
rf_resamples <- fit_resamples(
  rf_workflow,
  resamples = cv_folds,
  metrics = fraud_metrics,
  control = resample_control
)

message("Training XGBoost with 5-fold cross-validation...")
xgb_resamples <- fit_resamples(
  xgb_workflow,
  resamples = cv_folds,
  metrics = fraud_metrics,
  control = resample_control
)

cv_results <- bind_rows(
  collect_metrics(rf_resamples) %>% mutate(model = "Random Forest"),
  collect_metrics(xgb_resamples) %>% mutate(model = "XGBoost")
) %>%
  select(model, .metric, mean, std_err, n)

write_csv(cv_results, file.path("outputs", "cross_validation_metrics.csv"))
print(cv_results)

best_model_name <- cv_results %>%
  filter(.metric == "pr_auc") %>%
  slice_max(mean, n = 1, with_ties = FALSE) %>%
  pull(model)

final_workflow <- if (best_model_name == "Random Forest") rf_workflow else xgb_workflow

message("Best model by cross-validated AUC-PR: ", best_model_name)

# ---------------------------------------------------------------------------
# 8. Final model fit and threshold optimization with Youden's Index
# ---------------------------------------------------------------------------
find_youden_threshold <- function(truth, probability, positive_class = "Fraud") {
  truth <- factor(truth, levels = c("Legit", positive_class))

  roc_obj <- pROC::roc(
    response = truth,
    predictor = probability,
    levels = c("Legit", positive_class),
    direction = "<",
    quiet = TRUE
  )

  threshold_tbl <- pROC::coords(
    roc_obj,
    x = "best",
    best.method = "youden",
    ret = c("threshold", "sensitivity", "specificity"),
    transpose = FALSE
  ) %>%
    as_tibble() %>%
    mutate(youden_j = sensitivity + specificity - 1)

  threshold_tbl
}

final_fit <- fit(final_workflow, data = train_data)

test_predictions <- predict(final_fit, test_data, type = "prob") %>%
  bind_cols(predict(final_fit, test_data, type = "class")) %>%
  bind_cols(test_data %>% select(flag))

youden_threshold <- find_youden_threshold(
  truth = test_predictions$flag,
  probability = test_predictions$.pred_Fraud
)

optimal_threshold <- youden_threshold$threshold[1]

test_predictions <- test_predictions %>%
  mutate(
    .pred_class_youden = factor(
      if_else(.pred_Fraud >= optimal_threshold, "Fraud", "Legit"),
      levels = c("Legit", "Fraud")
    )
  )

default_metrics <- metric_set(recall, f_meas, pr_auc, roc_auc)(
  test_predictions,
  truth = flag,
  estimate = .pred_class,
  .pred_Fraud,
  event_level = "second"
) %>%
  mutate(threshold = 0.50, decision_rule = "Default")

youden_metrics <- metric_set(recall, f_meas, pr_auc, roc_auc)(
  test_predictions,
  truth = flag,
  estimate = .pred_class_youden,
  .pred_Fraud,
  event_level = "second"
) %>%
  mutate(threshold = optimal_threshold, decision_rule = "Youden")

test_metrics <- bind_rows(default_metrics, youden_metrics)

default_confusion <- conf_mat(test_predictions, truth = flag, estimate = .pred_class)
youden_confusion <- conf_mat(test_predictions, truth = flag, estimate = .pred_class_youden)

write_csv(test_predictions, file.path("outputs", "test_predictions.csv"))
write_csv(test_metrics, file.path("outputs", "test_metrics.csv"))
write_csv(youden_threshold, file.path("outputs", "youden_threshold.csv"))

print(test_metrics)
print(default_confusion)
print(youden_confusion)
print(youden_threshold)

# ---------------------------------------------------------------------------
# 9. Variable importance for model interpretation
# ---------------------------------------------------------------------------
vip_plot <- final_fit %>%
  extract_fit_parsnip() %>%
  vip(num_features = 20) +
  labs(title = paste("Top Predictors -", best_model_name))

ggsave(file.path("figures", "variable_importance.png"), vip_plot, width = 8, height = 6)

message("Analysis complete. Outputs saved in the figures/ and outputs/ folders.")
