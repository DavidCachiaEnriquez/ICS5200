library(InvariantCausalPrediction)

inv_pred_recola <- function(input_file, output_file) {
  data <- read.csv(input_file, check.names = FALSE)

  column_names <- colnames(data)
  selected_columns <- column_names[grepl("ComPar|audio_speech|VIDEO|Face_detection|ECG|EDA", column_names)] #nolint

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
  max_no_variables <- 12
  result <- suppressWarnings(ICP(features, class_label, environments, alpha=0.01, selection = c("boosting"), maxNoVariables = max_no_variables , maxNoVariablesSimult = max_no_variables, showAcceptedSets = FALSE, showCompletion = FALSE)) #nolint
  invariant_feature_indices <- unique(unlist(result$acceptedSets))
  invariant_feature_names <- colnames(features)[invariant_feature_indices]

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
} else {
  cat("No function name provided.\n")
}
