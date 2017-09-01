#!/bin/bash

cd $1
python3 ../plot_epochs.py *.log "training RMSE" --save training_rmse.png
python3 ../plot_epochs.py *.log "testing RMSE" --save testing_rmse.png
python3 ../plot_epochs.py *.log "likelihoods percentile 50" --save likelihood_training_set_median.png
python3 ../plot_epochs.py *.log "likelihoods validation set percentile 50" --save likelihood_validation_set_median.png
python3 ../plot_epochs.py *.log "objs percentile 50" --save objs_training_set_median.png
python3 ../plot_epochs.py *.log "objs validation set percentile 50" --save objs_validation_set_median.png
python3 ../plot_epochs.py *.log "mean_total_kls_per_dim percentile 5" --save mean_total_kls_per_dim_percentile_5.png
python3 ../plot_epochs.py *.log "mean_total_kls_per_dim percentile 50" --save mean_total_kls_per_dim_median.png
python3 ../plot_epochs.py *.log "mean_total_kls_per_dim percentile 95" --save mean_total_kls_per_dim_percentile_95.png
