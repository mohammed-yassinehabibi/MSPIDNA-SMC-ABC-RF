# script.R
library(drf)
library(Matrix)

generate_DRF_weights <- function(nb_iter, perturbation, i, DRF_folder, custom_exp_name, model_custom_exp_name) {
  statistics_target_filename <- paste0(DRF_folder, "/statistics_targets_", i, custom_exp_name, model_custom_exp_name, ".csv")
  statistics_target <- read.csv(statistics_target_filename)
  if (nb_iter == 1) {
  Xdrf_filename <- paste0(DRF_folder, "prior_", nb_iter, "/Xdrf_", i, "_prior_", nb_iter, custom_exp_name, model_custom_exp_name, ".csv")
  Ydrf_filename <- paste0(DRF_folder, "prior_", nb_iter, "/Ydrf_", i, "_prior_", nb_iter, custom_exp_name, model_custom_exp_name, ".csv")
  
} else {
  Xdrf_filename <- paste0(DRF_folder, "prior_", nb_iter, "/Xdrf_", i, "_prior_", nb_iter, "_", perturbation, custom_exp_name, model_custom_exp_name, ".csv")
  Ydrf_filename <- paste0(DRF_folder, "prior_", nb_iter, "/Ydrf_", i, "_prior_", nb_iter, "_", perturbation, custom_exp_name, model_custom_exp_name, ".csv")
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
  output_filename <- paste0(DRF_folder,"prior_", nb_iter, "/DRF_weights_", i, "_prior_", nb_iter, custom_exp_name, model_custom_exp_name, ".csv")
  
} else {
  output_filename <- paste0(DRF_folder,"prior_", nb_iter, "/DRF_weights_", i, "_prior_", nb_iter, "_", perturbation, custom_exp_name, model_custom_exp_name, ".csv")
}
  write.csv(DRF_weights, output_filename, row.names = FALSE)
  
  print("DRF weights saved successfully to DRF_weights.csv")
}
