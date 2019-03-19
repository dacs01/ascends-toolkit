rm output/*

# The following command 

./train_regression.py data/creep.csv output/creep_default LMP --ignore_col id Name logRT --model_type SVM --auto_tune False --save_test_chart True --scaler StandardScaler

# will train SVM regressor with all input columns except id, Name, logRT

# Data will be scaled using StandardScaler.
# Auto hyperparameter tuning will not be performed, and cross-validation test result chart will be saved.
# You will get something close to * (SVM)	 RMSE =  989.911, R2 =   -0.013 via 5-fold cross validation 
# which means, SVM does not perform well without tuning

# The following commend will perform training SVM but this time with hyperparameter tuning and top-10 feature selection using MIC criterion 
# Also, it will save the tuned parameters, correlation analysis result in csv format, and correlation analysis charts
# For example, "creep_autotuned,MIC (target_col = 'LMP').png" will have the MIC values for all input columns against a target column LMP
./train_regression.py data/creep.csv output/creep_autotuned LMP --ignore_col id Name logRT --model_type SVM --auto_tune True --auto_tune_iter 1000 --save_test_chart True --scaler StandardScaler --save_corr_report False --save_corr_chart False --feature_selection MIC --num_of_features 10 --save_auto_tune True

# You will get something close to * (SVM)	 RMSE =  374.405, R2 =    0.855 via 5-fold cross validation 
# which is a dramatic improvement
# The tuned parameters will be saved to where the original input data is located, in this case, "data/creep.csv,Model=SVM,Scaler=StandardScaler.tuned.prop"
# Without having to perform hyperparameters everytime, you can just load the prop file as follows
./train_regression.py data/creep.csv output/creep_autotuned_file_loaded LMP --ignore_col id Name logRT --model_type SVM --hyperparameter_file data/creep.csv,Model=SVM,Scaler=StandardScaler.tuned.prop --save_test_chart True --feature_selection MIC --num_of_features 10

# The result will be slightly different due to different random_state, you can fix random_state by adding --random_state option

# To see more options, please see
# ./training_regression.py -h 

# Enjoy !