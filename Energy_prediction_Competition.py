import os, sys,site,pyodbc , pandas as pd,numpy as np , imp,sys,yaml , matplotlib.pyplot as plt , gc ,csv
from dateutil import parser
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta

from catboost import CatBoostRegressor, Pool
from pylab import rcParams
from datetime import datetime
from copy import deepcopy
from pathlib import Path

#path ="/Boris/Projects/Brain/Energy_prediction_Competition-Kasun/Code_submitted"
path = Path(__file__).parents[0].absolute() # parents[0] => go one back ( like ../) to folder where the current __file__ is located 
os.chdir(path)

print(path)

import Aux_Energy_prediction_Competition as Aux_Energy
imp.reload(Aux_Energy)


################################################
#Set required col names ########################
################################################
		
	
def set_config(is_to_plot = True,is_to_standardize=False):

	###############################
	#Set configuration object
	############################### 

	agg_level={"months":0,'days':1}
	
	item_col ="meter_id" ; y_label ="energy_agg"  ;  date_column ='date_only' ; pred_col = "prediction"
	y_label_original_col = y_label #+ "_original"
	# key_id_col = "key_ID"
	
	key_columns = [item_col]
			
	#Horize is 14 days ahead !
	total_future_UNITS_to_forecast = horizon = 365
	

	losses_function_optimize = "Quantile:alpha=0.45"#"RMSE"		

	is_residual_based_model =  False
	

	#Create	configuration object with all column setups
	conf_obj = Aux_Energy.Configuration_class(agg_level = agg_level.copy(), item_col =item_col  ,\
											  date_column=date_column, pred_col=pred_col, y_label = y_label, y_label_original_col = y_label_original_col,
											  key_columns =key_columns ,
											  total_future_UNITS_to_forecast = total_future_UNITS_to_forecast,											
											  losses_function_optimize = losses_function_optimize,
											  is_residual_based_model = is_residual_based_model,
											  is_to_plot = is_to_plot,is_to_standardize=is_to_standardize)
	
			
	return(conf_obj)




#####################################################################
#Predict the test set
#####################################################################
#def predict_on_test(all_given_data_df,test_data_to_predict,CB_model,x_columns_numeric_to_remain , x_CAT_cols_to_remain , conf_obj)	:
def predict_on_test(all_given_data_df,test_data_to_predict,CB_model,x_columns_all, cat_features , conf_obj)	:
	
	
	#Get months name
	all_given_data_df['month_name'] = all_given_data_df[conf_obj.date_column].map(lambda x : x.strftime('%b'))#Get months name
	
	#################################
	#Predict on all given (trained)
	#################################

	all_given_data_to_predict_pool = Pool(   data  = all_given_data_df[x_columns_all ].copy(), cat_features = cat_features )
	predictions = CB_model.predict(all_given_data_to_predict_pool)


	#No negative predictions - BUT for residuals based solution there are !
	if  conf_obj.is_residual_based_model == False:
		predictions[predictions < 0] = 0
	
	#Set predictions for the TRAIN set		
	all_given_data_df.loc[:,conf_obj.pred_col] = predictions
	
	##################
	#Predict on test
	#################

	test_data_to_predict_pool = Pool(   data  = test_data_to_predict[x_columns_all ].copy(), cat_features = cat_features )
	predictions = CB_model.predict(test_data_to_predict_pool)
	
	predictions[predictions < 0] = 0
	#Set predictions for the TEST set	
	test_data_to_predict.loc[:,conf_obj.pred_col] = predictions
	
	###################################################################
	#Run Shap go get shaply values per each sample in data to predict
	###################################################################

	#Set_of_TS_Models.run_shap_models(CB_model , x_train = df_full[x_columns_numeric + CAT_cols ])
	(shaply_values,shaply_values_as_DF,shap_explainer) = \
													Aux_Energy.get_shaply_values_from_Catboost(CB_model ,\
													x_train = test_data_to_predict[x_columns_all ].copy(),\
													x_train_pool  = test_data_to_predict_pool , is_to_plot = False)
	
	assert(len(shaply_values)==len(test_data_to_predict)),"Must have shaply value per row in shap_values !"	
	assert(len(shaply_values_as_DF)==len(test_data_to_predict)),"Must have shaply value per row in shap_values !"	
	
								

	#######################################################################
	#Add key columns to shaply values df - to be able to do group by !
	#######################################################################
	#shaply_values_as_DF.loc[:,conf_obj.key_columns + [conf_obj.date_column] + [conf_obj.pred_col]] = test_data_to_predict[conf_obj.key_columns + [conf_obj.date_column] + [conf_obj.pred_col]]
	
	cols = list(shaply_values_as_DF.columns) + conf_obj.key_columns + [conf_obj.date_column] + [conf_obj.pred_col]
	shaply_values_as_DF = pd.concat([shaply_values_as_DF , test_data_to_predict[conf_obj.key_columns + [conf_obj.date_column] + [conf_obj.pred_col]] ],axis=1 , ignore_index=True)
	shaply_values_as_DF.columns = cols
	assert( np.all(shaply_values_as_DF[conf_obj.pred_col]==test_data_to_predict[conf_obj.pred_col]) ) ,"Sanity for prediction assignments was failed !"

	shaply_values_as_DF.loc[:,"month_name"] = test_data_to_predict[conf_obj.date_column].map(lambda x : x.strftime('%b'))#Get months name
	
	#Verify that I have the same number of smaples to predict per each meter id & the number is EQUAL to the horizon length 
	assert(shaply_values_as_DF.groupby(by="meter_id")[conf_obj.date_column].count().sort_values(ascending =False).value_counts().nunique() ==1)
	assert(shaply_values_as_DF.groupby(by="meter_id")[conf_obj.date_column].count().value_counts().index[0] ==conf_obj.total_future_UNITS_to_forecast)
	
	#shaply_values_as_DF.to_csv("shaply_values_as_DF.csv",index=False)
	
	###################################################################
	#Remove day/energy_mean of month from the shap values
	###################################################################
	"""
	Based in "feature_importance_based_on_Shaply_values" day of month (1,2,3,....31) & energy_mean of last year 
	have high impact on daily prediction. BUT we are aggregating by month and year
	so day of month isn't intersting'
	"""
	
	#shaply_values_as_DF_bck = shaply_values_as_DF.copy()
	x_columns_all_filtered = x_columns_all.copy()
	cols_to_remove =["day","energy_mean"]
	for col_to_drop  in cols_to_remove :
		if col_to_drop not in x_columns_all:
			continue
		
		print(col_to_drop , "was dropped from text explanation generator  ! ")
		shaply_values_as_DF.drop([col_to_drop],axis=1,inplace=True) # Remove more than one attribute
		x_columns_all_filtered.remove(col_to_drop) #remove from x_columns as well
	
	######################################################################
	#Columns that can be adapted to decrease/increase energy consumptions
	######################################################################
	#Numeric attributes (taken from "DataDescription-FUZZ.pdf"  file
	# which explains each attribute in the data input)
	cols_that_changing_them_may_influnce_on_energy_consumptions = ['dishwasher', 'freezer', 'fridge_freezer', 'refrigerator', 'tumble_dryer', \
																'washing_machine', 'game_console', 'laptop', 'pc', 'router', 'set_top_box', 'tablet', 'tv']
		
	cols_that_changing_them_may_influnce_on_energy_consumptions_df = pd.DataFrame(cols_that_changing_them_may_influnce_on_energy_consumptions,columns=["changeable_cols"])	
	######################################################################
	#Generate results
	######################################################################

	#Get sum of energy consumption per each type of household
	meta_data_cols = ["dwelling_type","num_occupants","month_name"]
	sum_energy_consumption_observed_per_month_per_similar_households = all_given_data_df.groupby(by=conf_obj.key_columns + meta_data_cols)['energy_agg'].sum().reset_index() #get sum of usage
	sum_energy_consumption_observed_per_month_per_similar_households = sum_energy_consumption_observed_per_month_per_similar_households.groupby(by= meta_data_cols)['energy_agg'].mean().reset_index() #get mean of all sums

	sum_energy_consumption_observed_per_month_per_similar_households[:4]
	# 	  dwelling_type  num_occupants month_name  energy_agg
	#       bungalow          1.000        Aug     145.682
	#       bungalow          1.000        Dec     113.216
	#       bungalow          1.000        Jul     137.695
	#       bungalow          1.000        Nov     151.553

	cols = ["meter_id" ,"AnnualPRED","AnnualEXP","JanPRED","JanEXP","FebPRED","FebEXP","MarPRED","MarEXP","AprPRED","AprEXP","MayPRED","MayEXP","JunPRED","JunEXP","JulPRED","JulEXP","AugPRED","AugEXP","SepPRED","SepEXP","OctPRED","OctEXP","NovPRED","NovEXP","DecPRED","DecEXP"]	
	#cols = cols.replace("\t",",")
	

	#pandarallel.initialize(nb_workers= int(os.cpu_count())-1, use_memory_fs = False ,progress_bar = False) #set num of cores	 ; parallel_apply	
	ans_df = all_given_data_df.groupby(by = conf_obj.key_columns,as_index=False).apply(generate_results_per_meter_id,shaply_values_as_DF,\
																					cols_that_changing_them_may_influnce_on_energy_consumptions_df,
																					sum_energy_consumption_observed_per_month_per_similar_households, meta_data_cols,\
																					conf_obj=conf_obj,x_columns = x_columns_all_filtered,top_n_features =3).reset_index(drop=True)

	ans_df.columns = cols
	assert(all_given_data_df.meter_id.nunique() == len(ans_df))
	#string_cols = [col  for col in cols if "EXP" in col]
	#ans_df[string_cols] = ans_df[string_cols].astype(str)
	#ans_df.to_csv("results_to_submit_21.csv",index = False,quoting=csv.QUOTE_NONNUMERIC)
	return(ans_df)
		
		
	
###################################################################
######################## Generate results per meter id ############
###################################################################


def generate_results_per_meter_id(sub_df_per_meter_id , shaply_values_as_DF ,cols_that_changing_them_may_influnce_on_energy_consumptions_df,\
								  sum_energy_consumption_observed_per_month_per_similar_households,meta_data_cols, conf_obj ,x_columns,top_n_features = 3) :

	#Get examined meter_id
	meter_id = sub_df_per_meter_id[conf_obj.item_col].unique()[0]
	assert(sub_df_per_meter_id[conf_obj.item_col].nunique()==1),"Only ONE unique meter-id MUST be here !"
	#print(meter_id)

	##########################
	#Shap get yearly values
	##########################
	shap_sub_df = shaply_values_as_DF[shaply_values_as_DF[conf_obj.item_col] ==meter_id].copy()
	shap_importance_yearly = Aux_Energy.get_importance_per_meter_id(shap_sub_df,conf_obj, x_columns = x_columns) #Per all sub_df NOT per month
	shap_importance_yearly = shap_importance_yearly.sort_values(by="importance_mean",ascending=False) # sort from high to low "importance_mean"

	####################################
	#Actual values yearly and monthly
	####################################
	sum_energy_consumption_observed_per_month = sub_df_per_meter_id.groupby(by='month_name')['energy_agg'].sum() #actual monthly total energy consumption	
	# 	month_name
	# Aug   130.809
	# Dec   202.446
	# Jul   140.212
	# Jun   144.041
	# May   116.811
	# Nov   268.866
	# Oct   153.194
	# Sep   116.803
	
	sum_energy_consumption_observed_per_year = sum_energy_consumption_observed_per_month.values.sum()#actual yearly total energy consumption
	#809
	
	########################################
	#Model's predictions yearly and monthly
	########################################	
	sum_energy_consumption_predicted_per_month = shap_sub_df.groupby(by='month_name')[conf_obj.pred_col].sum()	 #predicted monthly total energy consumption	
	num_of_observed_unique_months = len(sum_energy_consumption_observed_per_month) #num of historical months
	
	# 	month_name
	# Apr   213.894
	# Aug   179.024
	# Dec   210.261
	# Feb   246.805
	# Jan   275.645
	# Jul   184.403
	# Jun   182.607
	# Mar   241.389
	# May   143.624
	# Nov   240.507
	# Oct   177.348
	# Sep   168.336
	
	sum_energy_consumption_predicted_per_year =  sum_energy_consumption_predicted_per_month.values.sum() #predicted  yearly total energy consumption
	#2463.843
	
	################################################################################
	#For meter ids without full data cycle (no 12 months in the history)
	#so use similar arhouseholds
	################################################################################	

	#To get usage for the similar meta data families
	sum_energy_consumption_observed_per_month_per_similar_households_selected =\
		 sum_energy_consumption_observed_per_month_per_similar_households.merge(sub_df_per_meter_id[meta_data_cols].drop_duplicates(), left_on = meta_data_cols, right_on = meta_data_cols, how='inner').copy()

	assert(len(sum_energy_consumption_observed_per_month_per_similar_households_selected) > 0 ),"must be data after the inner join!"
	
	#Get predictions + auto generated text for yearly aggregations
	yearly_result = Aux_Energy.generate_yearly_prediction_with_auto_generated_message(sum_energy_consumption_observed_per_year,sum_energy_consumption_predicted_per_year,\
											shap_importance_yearly ,cols_that_changing_them_may_influnce_on_energy_consumptions_df, top_n_features ,num_of_observed_unique_months = num_of_observed_unique_months)


	#List of month names
	#list_of_month_names = shaply_values_as_DF['month_name'].unique()
	list_of_month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',   'Oct', 'Nov', 'Dec']
	
	#Get predictions + auto generated text for monthly aggregations
	monthly_result = Aux_Energy.generate_monthly_prediction_with_auto_generated_message(sum_energy_consumption_observed_per_month , sum_energy_consumption_predicted_per_month,\
															sum_energy_consumption_observed_per_month_per_similar_households_selected ,\
															cols_that_changing_them_may_influnce_on_energy_consumptions_df.copy(),\
															shap_sub_df,top_n_features,conf_obj,x_columns ,list_of_month_names )
		
		
	#Combine between yearly & monthly
	yearly_result.extend(monthly_result)
	yearly_result = [meter_id] + yearly_result
	yearly_result_df = pd.DataFrame(yearly_result).T
	return(yearly_result_df)
	
	

def main():
	
	#Set font size
	rcParams['figure.figsize'] = 16, 4 #14, 5 
			
	#Set configuration object
	conf_obj = set_config(is_to_plot =True ,is_to_standardize =True )
	
	#Load the data	
	(X_train, X_valid, y_train, y_valid,df_combined,test_data_to_predict,x_columns,cat_features,conf_obj)  = Aux_Energy.load_data(path,conf_obj)

	#Train a model
	CB_last_model =  Aux_Energy.execute_the_modeling_process(X_train, X_valid, y_train, y_valid,test_data_to_predict,x_columns,cat_features,conf_obj)
	
	#Shaply + explain the results
	results_to_submit_df = predict_on_test(X_train,test_data_to_predict, CB_last_model, x_columns , cat_features,conf_obj)
	
	#Store the results to csv
	results_to_submit_df.to_csv("results_to_submit_ver_X.csv",index = False,quoting=csv.QUOTE_NONNUMERIC)

	
if __name__ == "__main__":
	main()
