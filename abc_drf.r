# script.R
library(drf)
library(Matrix)

generate_DRF_weights <- function(nb_iter, i, results_folder) {
  statistics_target_filename <- paste0(results_folder, "/statistics_targets_", i,  ".csv")
  statistics_target <- read.csv(statistics_target_filename)
  if (nb_iter == 1) {
  Xdrf_filename <- paste0(results_folder, "prior_", nb_iter, "/Xdrf_", i, "_prior_", nb_iter,  ".csv")
  Ydrf_filename <- paste0(results_folder, "prior_", nb_iter, "/Ydrf_", i, "_prior_", nb_iter, ".csv")
  
} else {
  Xdrf_filename <- paste0(results_folder, "prior_", nb_iter, "/Xdrf_", i, "_prior_", nb_iter, ".csv")
  Ydrf_filename <- paste0(results_folder, "prior_", nb_iter, "/Ydrf_", i, "_prior_", nb_iter, ".csv")
}
  Xdrf <- read.csv(Xdrf_filename)
  Ydrf <- read.csv(Ydrf_filename)
  
  # Train DRF model
  print("Loaded Xdrf, Ydrf")
  drfmodel <- drf(Xdrf, Ydrf)
  print("DRF model initialized")
  
  # Obtain sample weights
  DRF_weights <- as.vector(get_sample_weights(drfmodel, statistics_target))
  print("weight OK")
  
  # Save the weights in a CSV file
  if (nb_iter == 1) {
  output_filename <- paste0(results_folder,"prior_", nb_iter, "/DRF_weights_", i, "_prior_", nb_iter, ".csv")
  
} else {
  output_filename <- paste0(results_folder,"prior_", nb_iter, "/DRF_weights_", i, "_prior_", nb_iter, ".csv")
}
  write.csv(DRF_weights, output_filename, row.names = FALSE)
  
  print("DRF weights saved successfully to DRF_weights.csv")
}
