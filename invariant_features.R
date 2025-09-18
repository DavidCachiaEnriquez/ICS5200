library(future)
library(InvariantCausalPrediction)

inv_pred_recola <- function(input_file, output_file, max_inv_features) {
  data <- read.csv(input_file, check.names = FALSE)

  selected_columns <- grep("ComPar|audio_speech|VIDEO|Face_detection|ECG|EDA", names(data), value = TRUE) #nolint

  # OBTAIN FEATURES
  features <- data[, selected_columns]
  features <- as.matrix(features)

  # CLASS LABEL
  if ("Class_Label_Arousal" %in% names(data)) {
    class_label <- data$Class_Label_Arousal
  } else if ("Class_Label_Valence" %in% names(data)) {
    class_label <- data$Class_Label_Valence
  }

  # PARTICIPANT IDS
  environments <- data$Participant_Number

  # OBTAIN INVARIANT CAUSAL PREDICTIONS
  if (length(unique(environments)) > 1) { # STANDARD METHOD
    max_no_variables <- as.numeric(max_inv_features)

    # result <- suppressWarnings(ICP(features, class_label, environments, alpha=0.01, selection = c("boosting"), maxNoVariables = max_no_variables , maxNoVariablesSimult = max_no_variables)) #nolint

    plan(multisession, workers = 20)
    result <- suppressWarnings(
      ICP(features, class_label, environments,
          alpha = 0.1,
          selection = "boosting",
          maxNoVariables = max_no_variables,
          maxNoVariablesSimult = max_no_variables,
          showAcceptedSets = FALSE,
          showCompletion = FALSE)
    )
    plan(sequential)

    invariant_feature_indices <- unique(unlist(result$acceptedSets))
    invariant_feature_names <- colnames(features)[invariant_feature_indices]
  } else { # IF THERE IS ONLY 1 ENVIRONMENT
    invariant_mask <- apply(features, 2, function(col) length(unique(col)) == 1)
    invariant_feature_names <- colnames(features)[invariant_mask]
  }

  # RETURN
  invariant_feature_df <- data.frame(invariant_feature_names)
  write.csv(invariant_feature_df, output_file, row.names = FALSE, quote = FALSE) #nolint
}

inv_pred_again <- function(input_file, output_file, max_inv_features) {
  data <- read.csv(input_file, check.names = FALSE)

  selected_columns <- setdiff(names(data), c("[control]player_id","[control]genre","[control]game","[output]arousal","Binary_Arousal_Class")) #nolint
  selected_columns <- intersect(selected_columns, names(data))
  selected_columns <- selected_columns[-1]

  # OBTAIN FEATURES
  features <- data[, selected_columns]
  features <- as.matrix(features)

  # CLASS LABEL
  class_label <- data$Binary_Arousal_Class

  # PARTICIPANT IDS
  environments <- data[["[control]game"]]

  # OBTAIN INVARIANT CAUSAL PREDICTIONS
  if (length(unique(environments)) > 1) { # STANDARD METHOD
    max_no_variables <- as.numeric(max_inv_features)

    result <- suppressWarnings(ICP(features, class_label, environments, alpha=0.01, selection = c("boosting"), maxNoVariables = max_no_variables , maxNoVariablesSimult = max_no_variables, showAcceptedSets = FALSE, showCompletion = FALSE)) #nolint
    print(result)
    invariant_feature_indices <- unique(unlist(result$acceptedSets))
    invariant_feature_names <- colnames(features)[invariant_feature_indices]
  } else { # IF THERE IS ONLY 1 ENVIRONMENT
    invariant_mask <- apply(features, 2, function(col) length(unique(col)) == 1)
    invariant_feature_names <- colnames(features)[invariant_mask]
  }

  # RETURN
  invariant_feature_df <- data.frame(invariant_feature_names)
  write.csv(invariant_feature_df, output_file, row.names = FALSE, quote = FALSE) #nolint
}

# OBTAINING ARGUMENTS FROM PYTHON FILE
args <- commandArgs(trailingOnly = TRUE)

# ARGUMENT HANDLER TO CALL A FUNCTION
if (length(args) > 0) {
  func_name <- args[1]
  func_args <- args[-1]  # all remaining args after function name
  do.call(func_name, as.list(func_args))
}