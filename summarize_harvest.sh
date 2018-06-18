#!/bin/bash

HARVESTS=$(bash list_harvest.sh |  awk '{print}' ORS=" ")
     
echo HARVESTS: $HARVESTS
python3 logs_summarizer.py $HARVESTS \
    -f "" \
    -s best_testing_rmse \
    -c lr,best_testing_rmse,best_training_rmse,minibatch_size,regularization_type,regularization_lambda,n_hid_layers,hid_dim,K,harvest_dir,max_epoch,upd,g_hid,regression_type,nan,dropout_p,input_dropout_p,TK,r_KL
    #--twod=lr,regularization_lambda
