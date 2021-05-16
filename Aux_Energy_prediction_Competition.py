import pandas as pd,numpy as np
import os,sys,imp
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from copy import deepcopy

from pathlib import Path
import catboost as catboost


import shap


####################################
# Configuration Class
####################################
	
class Configuration_class(object):
	"""
	Setup  Class
	"""

	def __init__(self, agg_level, item_col , date_column\
				, losses_function_optimize ,y_label ,y_label_original_col, key_columns  , total_future_UNITS_to_forecast,pred_col,\
				  is_residual_based_model =False , is_to_plot =False ,is_to_standardize =False ):
		
		self.agg_level = agg_level
		self.item_col = item_col
		self.date_column = date_column
		self.pred_col = pred_col
		self.y_label = y_label
		self.y_label_original_col = y_label_original_col
		self.key_columns = key_columns
		self.total_future_UNITS_to_forecast = total_future_UNITS_to_forecast
		self.is_residual_based_model = is_residual_based_model
		self.is_to_plot = is_to_plot
		self.is_to_standardize = is_to_standardize
		
		self.losses_function_optimize = losses_function_optimize #The loss function is used to optimize the neural network


##################################
#Load data 
##################################

def load_data(path=None,conf_obj=None):


	path = Path(__file__).parents[1].absolute()

	# Model Training File
	df_train = pd.read_feather(os.path.join(path,"Data_2",'df_train.feather'))

	df_imputed = pd.read_csv(os.path.join(path,"Data_2","weekday_mean_season_imputed.csv"))

	df_train.head(5)

	# Replacing with the Imputed Energy Consumption (Boris - ???)
	df_imputed = df_imputed[['energy_agg']]
	df_train = df_train.drop(columns = ['energy_agg'])
	df_train  = pd.concat([df_train, df_imputed], axis = 1)


	# Reading the file relevant to test file (file o make prediction for \ the file to submit)
	df_test = pd.read_feather(os.path.join(path,"Data_2",'df_test.feather'))
	# Setting dependent variable to zero (test set)
	df_test['energy_agg'] = 0

	# Reading the estimated temperature file.
	#df_temp = pd.read_csv('all_temp_boostrap_median.csv')
	df_temp = pd.read_csv(os.path.join(path,"Data_2",'all_temp_boostrap_median.csv'))
	# Selecting the required temperature columns.
	df_temp = df_temp[['avg_temp', 'max_temp', 'min_temp']]


	# Removing the existing temperature columns from the test dataframe.
	df_test = df_test.drop(columns = ['avg_temp', 'max_temp','min_temp'])
	# Replace the temperature columns with the estimated.
	df_test = pd.concat([df_test, df_temp], axis = 1)


	# Combining with the training file.
	df_test.rename(columns={'ds': 'date_only'}, inplace=True)

	#Set split type col :
	df_train['split_type'] ="train" ; df_test['split_type'] ="test" 
	df_combined = pd.concat([df_train, df_test], axis = 0)

	# Replace empty fields as NAs
	df_combined.fillna(value=np.nan, inplace=True)

	df_combined.head(5)

	# Fill categorical NA values with Unknowns
	df_combined['dwelling_type'] = df_combined['dwelling_type'].cat.add_categories('Unknown')
	df_combined['dwelling_type'].fillna('Unknown', inplace =True) 
	df_combined['heating_fuel'] = df_combined['heating_fuel'].cat.add_categories('Unknown')
	df_combined['heating_fuel'].fillna('Unknown', inplace =True)
	df_combined['hot_water_fuel'] = df_combined['hot_water_fuel'].cat.add_categories('Unknown')
	df_combined['hot_water_fuel'].fillna('Unknown', inplace =True)
	df_combined['boiler_age'] = df_combined['boiler_age'].cat.add_categories('Unknown')
	df_combined['boiler_age'].fillna('Unknown', inplace =True) 
	df_combined['loft_insulation'] = df_combined['loft_insulation'].cat.add_categories('Unknown')
	df_combined['loft_insulation'].fillna('Unknown', inplace =True) 
	df_combined['wall_insulation'] = df_combined['wall_insulation'].cat.add_categories('Unknown')
	df_combined['wall_insulation'].fillna('Unknown', inplace =True) 
	df_combined['heating_temperature'] = df_combined['heating_temperature'].cat.add_categories('Unknown')
	df_combined['heating_temperature'].fillna('Unknown', inplace =True) 
	df_combined['efficient_lighting_percentage'] = df_combined['efficient_lighting_percentage'].cat.add_categories('Unknown')
	df_combined['efficient_lighting_percentage'].fillna('Unknown', inplace =True)

	# Fill numerical NA values with a predefined number.
	df_combined['num_occupants'].fillna(-999,inplace=True)
	df_combined['num_bedrooms'].fillna(-999,inplace=True)
	df_combined['dishwasher'].fillna(-999,inplace=True)
	df_combined['freezer'].fillna(-999,inplace=True)
	df_combined['fridge_freezer'].fillna(-999,inplace=True)
	df_combined['refrigerator'].fillna(-999,inplace=True)
	df_combined['tumble_dryer'].fillna(-999,inplace=True)
	df_combined['washing_machine'].fillna(-999,inplace=True)
	df_combined['game_console'].fillna(-999,inplace=True)
	df_combined['laptop'].fillna(-999,inplace=True)
	df_combined['pc'].fillna(-999,inplace=True)
	df_combined['router'].fillna(-999,inplace=True)
	df_combined['set_top_box'].fillna(-999,inplace=True)
	df_combined['tablet'].fillna(-999,inplace=True)
	df_combined['tv'].fillna(-999,inplace=True)


	# Define categorical features
	cat_features = ['dwelling_type','heating_fuel', 'hot_water_fuel', 'boiler_age', 'loft_insulation', 'wall_insulation', 'heating_temperature', 'efficient_lighting_percentage']
				
	
	#Boris : this not required !
	# Categorical Feature Embedding:
	from sklearn import preprocessing
	lbl = preprocessing.LabelEncoder()
	for col in cat_features:
		#col ="dwelling_type"
		df_combined[col] = lbl.fit_transform(df_combined[col].astype(str))
	
	

	# Temperature Scaling (0-1)
	min_max_scaler = preprocessing.MinMaxScaler()
	df_combined[["max_temp", "avg_temp", "min_temp"]] = min_max_scaler.fit_transform(df_combined[["max_temp", "avg_temp", "min_temp"]])	


	# Encoding month and day-of-the-week as integer variables.
	df_combined['month'] = df_combined['month'].astype('int8')
	df_combined['day_of_week'] = df_combined['day_of_week'].astype('int8')


	# Splitting the training and testing data.
	df_train_only = df_combined[df_combined.split_type =='train'].copy()
	df_test_only =  df_combined[df_combined.split_type =='test'].copy()
	

	x_columns = [i for i in df_train_only.columns if i not in \
						  [conf_obj.date_column ,conf_obj.y_label,conf_obj.pred_col ,"split_type" ] + conf_obj.key_columns ]

	
	#Drop my 'split_type' cols
	del df_train_only['split_type'] ; del df_test_only['split_type'] 

	df_test_only.shape #(1185520, 30)

	#Split the train to train & validation	
	X_train, X_valid, y_train, y_valid = train_test_split(df_train_only, df_train_only[conf_obj.y_label], test_size = 0.1, random_state = 28)


	#return(X_train, X_valid, y_train, y_valid,df_train_only,df_test_only,df_test,x_columns,cat_features)
	return(X_train, X_valid, y_train, y_valid,df_combined,df_test_only,x_columns,cat_features,conf_obj)



#################################################################
# Prepare  data frame to predict
#################################################################


def  prepare_test_data_to_predict(test_data_to_predict,df_full,df_weather_all,df_weather_boostrap_future,x_columns_to_remain,conf_obj):

	""" Enrich the test data with all required data to be able to run prediction in top of trained model """
	
	#########################################
	#Impute future Weather values (horizon)
	#########################################
	
	len_before = len(test_data_to_predict)
	
		
	test_data_to_predict =  pd.merge(test_data_to_predict , df_weather_boostrap_future, left_on =[conf_obj.item_col,conf_obj.date_column] ,\
			   right_on=[conf_obj.item_col,conf_obj.date_column],how ='left',suffixes=('','_drop') )

	#########################################
	#Add missing meta data to the test data
	#########################################
	missing_cols = list(set(df_full[x_columns_to_remain].columns)- set(test_data_to_predict.columns))		
	missing_cols_df = df_full[conf_obj.key_columns+ missing_cols].drop_duplicates().copy()
	assert(len(missing_cols_df) == df_full[conf_obj.key_columns].nunique()[0]),"prepare_test_data_to_predict() -> missing_cols_df must have only 1 rows per sample ! "
	
	
	test_data_to_predict =  pd.merge(test_data_to_predict, missing_cols_df , left_on =[conf_obj.item_col] , right_on=[conf_obj.item_col],how ='left')
	#######################################		
			

	assert( len(test_data_to_predict) == len_before), "prepare_test_data_to_predict() -> ERROR : Duplicated samples after merge ! please check why !"
	#adding "temp_bootsraped" as additional col to 'x_columns_to_remain'
	
	
	return(test_data_to_predict )
	
##############################################
#Get _importance_per_meter_id	
##############################################
def get_importance_per_meter_id(sub_df , conf_obj,x_columns, top_n =3):

	assert(sub_df[conf_obj.key_columns].nunique()[0] ==1),"get_importance_per_meter_id() => each sub dataframe's must have ONE meter-id"
	
	shap_values = sub_df[x_columns]
	
	feature_importance_mean_vals  = (shap_values).mean(axis = 0) # mean on abs shap values - per col
	feature_importance_vals = np.abs(shap_values).mean(axis = 0) # mean on abs shap values - per col
	vals_normalized = feature_importance_vals / np.sum(feature_importance_vals) #between 0 to 1		
	
	feature_importance_df_tmp = pd.DataFrame(list(zip( x_columns ,feature_importance_mean_vals,feature_importance_vals,vals_normalized)),columns=['feature','importance_mean','importance_abs','normalized_importance'])		
	feature_importance_df_tmp.sort_values(by=['normalized_importance'],ascending=False,inplace=True)
	
	return(feature_importance_df_tmp)


##############################################
#Auto messages generator 
##############################################

def generate_yearly_prediction_with_auto_generated_message(sum_energy_consumption_observed_per_year,sum_energy_consumption_predicted_per_year,\
	shap_importance_yearly, cols_that_changing_them_may_influnce_on_energy_consumptions_df ,top_n_features , num_of_observed_unique_months):

	""" Function to generate YEARLY predction + auto generated messgae """
	
		
	if num_of_observed_unique_months == 12 :
		
		#General message form
		meesage_form = "The estimation of your energy consumption for next year is {} {} because of the following attributes : {}"
	
		#Genearte messages
		if (sum_energy_consumption_predicted_per_year >= 1.1 * sum_energy_consumption_observed_per_year) &\
		   (sum_energy_consumption_predicted_per_year <= 1.2 * sum_energy_consumption_observed_per_year) :
		   AnnualEXP_message = meesage_form.format("slightly","higher", str(shap_importance_yearly[:top_n_features]['feature'].values))
	
		elif (sum_energy_consumption_predicted_per_year >= 1.2 * sum_energy_consumption_observed_per_year) : 
			AnnualEXP_message = meesage_form.format("much","higher", str(shap_importance_yearly[:top_n_features]['feature'].values))
	
		elif (sum_energy_consumption_predicted_per_year <= 0.9 * sum_energy_consumption_observed_per_year) &\
			   (sum_energy_consumption_predicted_per_year >= 0.8 * sum_energy_consumption_observed_per_year) :
			   AnnualEXP_message = meesage_form.format("slightly","lower", str(shap_importance_yearly[-top_n_features:]['feature'].values))
	
		elif (sum_energy_consumption_predicted_per_year < 0.8 * sum_energy_consumption_observed_per_year) :
			AnnualEXP_message = meesage_form.format("much","lower", str(shap_importance_yearly[-top_n_features:]['feature'].values))
	
		else:		
			AnnualEXP_message = "The estimation of your energy consumption for next year is similar to previous year"
		
	else:
		meesage_form = "The estimation of your energy consumption for next year is mostly influenced by the following attributes : {}"
		
		shap_importance_yearly_ordered_by_importance_abs_value = shap_importance_yearly.sort_values(by="importance_abs",ascending=False) # sort from high to low "importance_abs"
		AnnualEXP_message = meesage_form.format( str(shap_importance_yearly_ordered_by_importance_abs_value[: top_n_features]['feature'].values))



	#####################################################################################
	#Getting attributes that by changing them we may control the energy consumptions
	#####################################################################################

	shap_importance_yearly_changing_them_may_influnce_on_energy_consumptions_df =  pd.merge(shap_importance_yearly, cols_that_changing_them_may_influnce_on_energy_consumptions_df ,\
																						  left_on =['feature'] , right_on=['changeable_cols'],how ='left').dropna()
		
	shap_importance_yearly_changing_them_may_influnce_on_energy_consumptions_df = \
										shap_importance_yearly_changing_them_may_influnce_on_energy_consumptions_df.sort_values(by="importance_abs",ascending=False) # sort from high to low "importance_abs"		


	#Additional Message - some recommended action
	if len(shap_importance_yearly_changing_them_may_influnce_on_energy_consumptions_df)> 0:
		additional_message_form =  "Your consumption may reduced by controlling the following devices : {}"
		additional_message = additional_message_form.format(  str(shap_importance_yearly_changing_them_may_influnce_on_energy_consumptions_df[: top_n_features]['feature'].values) )  
		
		#Append to the message
		AnnualEXP_message = AnnualEXP_message + os.linesep + additional_message

	#####################################################################################	
		
		
	#Get Annual predction	
	AnnualPRED = sum_energy_consumption_predicted_per_year	

	return([AnnualPRED,AnnualEXP_message])


def generate_monthly_prediction_with_auto_generated_message(sum_energy_consumption_observed_per_month , sum_energy_consumption_predicted_per_month,\
															sum_energy_consumption_observed_per_month_per_similar_households_selected ,\
															cols_that_changing_them_may_influnce_on_energy_consumptions_df,\
															shap_sub_df ,\
															top_n_features,conf_obj,x_columns,list_of_month_names ):
	
	""" Function to generate MONTHLY predction + auto generated messgae """
	results_list = []
	for month  in list_of_month_names :
		#month = "Jan"
		
		
		#print(get_importance_per_meter_id(shap_sub_df[shap_sub_df.month_name==month],conf_obj, x_columns = x_columns_numeric_to_remain + x_CAT_cols_to_remain)[:10])
		shap_importance_per_month = get_importance_per_meter_id(shap_sub_df[shap_sub_df.month_name == month],conf_obj, x_columns = x_columns)
		shap_importance_per_month = shap_importance_per_month.sort_values(by="importance_mean",ascending=False) # sort from high to low "importance_mean"
			
		#Get sum of PREDICTED consumption per month
		sum_energy_consumption_predicted_per_month_selected = sum_energy_consumption_predicted_per_month[sum_energy_consumption_predicted_per_month.index == month][0]		
				
		############################################################################
		#Get OBSERVED data per month - if no such data so get usage of similar households
		############################################################################
		
		sum_energy_consumption_observed_per_month_selected = sum_energy_consumption_observed_per_month[sum_energy_consumption_observed_per_month.index == month]	

		#IF we DO have historical data per this month for current examined meter-id
		if len(sum_energy_consumption_observed_per_month_selected) >0 :
			sum_energy_consumption_observed_per_month_selected = sum_energy_consumption_observed_per_month_selected[0]
			meesage_form =  "In {}, your energy consumption will be {} {} because of the following  attributes : {}"
		else:
			# If there NO historical data for this month then try to Get usage of similar households
			sum_energy_consumption_observed_per_month_selected  =\
				 sum_energy_consumption_observed_per_month_per_similar_households_selected[sum_energy_consumption_observed_per_month_per_similar_households_selected['month_name'] == month]['energy_agg'].values
		
			if len(sum_energy_consumption_observed_per_month_selected) > 0 :
				meesage_form =  "In {}, compared to similar households , your energy consumption will be {} {} because of the following  attributes : {}"
				sum_energy_consumption_observed_per_month_selected = sum_energy_consumption_observed_per_month_selected[0] 
			else: 	
				# If there is NO historical data for this month to similar households as well then just take averge of OBSERVED energy consumption (of all given months)
				sum_energy_consumption_observed_per_month_selected = np.mean(sum_energy_consumption_observed_per_month.values)
				meesage_form =  "In {}, based on mean of all your given months , your energy consumption will be {} {} because of the following  attributes : {}"

		#Genearte messages
		if (sum_energy_consumption_predicted_per_month_selected >= 1.1 * sum_energy_consumption_observed_per_month_selected) &\
		   (sum_energy_consumption_predicted_per_month_selected <= 1.2 * sum_energy_consumption_observed_per_month_selected) :		  
		   Monthly_EXP_message =meesage_form.format(month,"slightly","higher", str(shap_importance_per_month[:top_n_features]['feature'].values))


		elif (sum_energy_consumption_predicted_per_month_selected >= 1.2 * sum_energy_consumption_observed_per_month_selected) : 
			Monthly_EXP_message = meesage_form.format(month,"much","higher", str(shap_importance_per_month[:top_n_features]['feature'].values))

		elif (sum_energy_consumption_predicted_per_month_selected <= 0.9 * sum_energy_consumption_observed_per_month_selected) &\
			   (sum_energy_consumption_predicted_per_month_selected >= 0.8 * sum_energy_consumption_observed_per_month_selected) :
			   Monthly_EXP_message = meesage_form.format(month,"slightly","lower", str(shap_importance_per_month[-top_n_features:]['feature'].values))

		elif (sum_energy_consumption_predicted_per_month_selected <= 0.8 * sum_energy_consumption_observed_per_month_selected) :			
			Monthly_EXP_message = meesage_form.format(month,"much","lower", str(shap_importance_per_month[-top_n_features:]['feature'].values))

		else:		
			meesage_form= "In {},  the estimation of your energy consumption is similar to the same month from your previous usage"
			Monthly_EXP_message = meesage_form.format(month)
			
	

		#####################################################################################
		#Getting attributes that by changing them we may control the energy consumptions
		#####################################################################################

		
		shap_importance_per_month_changing_them_may_influnce_on_energy_consumptions_df =  pd.merge(shap_importance_per_month, cols_that_changing_them_may_influnce_on_energy_consumptions_df ,\
																							  left_on =['feature'] , right_on=['changeable_cols'],how ='left').dropna()
			
		shap_importance_per_month_changing_them_may_influnce_on_energy_consumptions_df = \
											shap_importance_per_month_changing_them_may_influnce_on_energy_consumptions_df.sort_values(by="importance_abs",ascending=False) # sort from high to low "importance_abs"		


		#Additional Message - some recommended action
		if len(shap_importance_per_month_changing_them_may_influnce_on_energy_consumptions_df)> 0:
			additional_message_form =  "Your consumption may reduced by controlling the following devices and what is related to them : {}"
			additional_message = additional_message_form.format(  str(shap_importance_per_month_changing_them_may_influnce_on_energy_consumptions_df[: top_n_features]['feature'].values) )  
			
			#Append to the message
			Monthly_EXP_message = Monthly_EXP_message + os.linesep + additional_message

		#####################################################################################		

		#Combine prediction with an explain
		results_list.extend([sum_energy_consumption_predicted_per_month_selected,Monthly_EXP_message])		
	
	return(results_list)


##################################
#Train a model   
##################################


def execute_the_modeling_process(X_train, X_valid, y_train, y_valid,test_data_to_predict,x_columns,cat_features,conf_obj):	
	
	# Defining the Catboost Object.
	cb_model = CatBoostRegressor(iterations=700,
								 learning_rate=0.04,
								 depth=12,
								 loss_function = 'Quantile:alpha=0.45',
								 eval_metric = 'RMSE',
								 random_seed = 30,
								 bagging_temperature = 0.4,
								 od_type='Iter',
								 metric_period = 75,
								 od_wait=100)

	# Training the Catboost model.
	model = cb_model.fit(X_train[x_columns], y_train,
			 eval_set=(X_valid[x_columns],y_valid),
			 cat_features=cat_features,
			 use_best_model=True,
			 verbose=True)


	# Getting the predictions for the testset.
	predictions = model.predict(test_data_to_predict[x_columns])
	predictions[predictions < 0] = 0 # !!! Boris

	test_data_to_predict.loc[:,'prediction'] = predictions
		


	return(model)


################################################################
### Get_shaply_values_from_Catboost ############################
################################################################	


def get_shaply_values_from_Catboost(model, x_train, x_train_pool, is_to_plot=True):
	

	shap_explainer = None
	try:

		# explain the model's predictions using SHAP
		shap_explainer = shap.TreeExplainer(model)
		shap_values = shap_explainer.shap_values(x_train)

		# Extracting the base value from shap explainer
		base_value = shap_explainer.expected_value

		# Transform to DF and set col names
		shaply_values_as_DF = pd.DataFrame(shap_values);
		shaply_values_as_DF.columns = list(x_train.columns)

	except:
		print("Boris Exception ! : using Catboost's internal Shap values ,instead of 'shap.TreeExplainer' ! ")
	
		shap_values_with_base_value = model.get_feature_importance(x_train_pool,  type=catboost.EFstrType.ShapValues) 
		

		# The last column is the base value
		base_value = shap_values_with_base_value[:, -1]  
		

		# Rest of them are Shaply values
		shap_values = shap_values_with_base_value[:, :-1].copy()  # The x_cols's Shaply values

		# Transform to DF and set col names
		shaply_values_as_DF = pd.DataFrame(shap_values);
		shaply_values_as_DF.columns = list(x_train.columns)
		return (shap_values, shaply_values_as_DF, shap_explainer)

	if is_to_plot == True:

		# load JS visualization code to notebook
		shap.initjs()

		
		# summarize the effects of all the features
		shap.summary_plot(shap_values, x_train)
		shap.summary_plot(shap_values, x_train, plot_type="bar")  # Feature importance

		# shap_interaction_values
		try:
			shap_interaction_values = shap_explainer.shap_interaction_values(x_train.iloc[:2000, :])
			shap.summary_plot(shap_interaction_values, x_train.iloc[:2000, :])
		except  AttributeError as error:
			print(" 'shap_interaction_values' generates the following error  (boris : this is known bug in shap) :",
				  error)

	return (shap_values, shaply_values_as_DF, shap_explainer)