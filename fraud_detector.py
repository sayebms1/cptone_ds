#------------------------------------IMPORTS-----------------------------------#
import os
import json
import wget #for programmatically downloading the data
import pickle
import requests
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import preprocessing
from urllib.request import urlopen
from collections import defaultdict
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             plot_precision_recall_curve, roc_auc_score)



import keras
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential


#-------------------------------SETTINGS(GLOBAL)-------------------------------#
PATH = './data/' #change this if input data are elsewhere
FILE_URL = "https://github.com/CapitalOneRecruiting/DS/blob/master/transactions.zip?raw=true"
FNAME = "transactions.txt"
ZIP_FNAME = "transactions.zip"
RAND_STATE = 42


#------------------------MAIN CLASS FOR FRAUD DETECTION------------------------#
class FraudDetection:    

    def __init__(self):
        
        self.raw_transactions = self.parse()
        self.cleaned_transactions = self.drop_extra_cols(self.raw_transactions)
        
        self.reversal_trans =  self.find_reverse_transactions(self.raw_transactions)
        self.multiswipe_trans = self.find_multiswipe_transactions(self.raw_transactions, dt_threshold = 120)        
        

    def download_and_unzip(self, url, extract_to='.'):
        """
        This function programmatically downloads the files and unzips them from
        the github url
        """
        print ("Downloading and un-zipping the file")
        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=extract_to)    
        
    def parse(self):
        
        """
        This functions reads the json files into a dictionary and convert the d-
        ictionary to a pandas dataframe. It's used in the __init__ function for 
        initialization purposes.
        
        ---ARGUMENT---
        
        ---RETURN---
        The dataframe created from the json file. 

        
        """
        
        #check if the file exists if not download it
        print ('Parsing data...')
        print ('This will take approximately 80 seconds if the file is already downloaded')        
        if not os.path.exists(PATH+FNAME):
            print ("File does not exist. Downloading file")
            self.download_and_unzip(FILE_URL, extract_to=PATH)
            # print (f"file donwloaded and extracted to: {os.getcwd()+PATH}")
            
        with open(PATH+FNAME, "r") as file:
            dict_list = []
            for line in file:
                dict_list.append(json.loads(line))
        dict_comb = defaultdict(list)

        for d in dict_list: # you can list as many input dicts as you want here
            for key, value in d.items():
                dict_comb[key].append(value)
        
        return pd.DataFrame(dict_comb)
    

    
    def separate_datetime(self, transactions):
        
        """
        Convert the datetime column to two separate date and time column. This is needed for 
        the identification of multiswipe transactions. By definition the multiswipe transactions
        happen on the same day but not necessarily the same time
        
        ---ARGUMENT---
        transactions (dataframe):          The dataframe
        
        ---RETURN---
        Returns the dataframe with transactionDateTime column removed and two new columns created
        with date and time separately on them
        """
        
        transactions_ = transactions.copy()
        transactions_['date'] = pd.to_datetime(transactions_['transactionDateTime']).dt.date
        transactions_['time'] = pd.to_datetime(transactions_['transactionDateTime']).dt.time
        transactions_ = transactions_.drop(columns=['transactionDateTime'])        
        
        return transactions_
    

    def find_reverse_transactions(self, transactions):
        """
        This function uses the flag that already exists to find the reverse transactions
        This option can be further explored by looking at the time difference between 
        transactions. 
        
        ---ARGUMENTS---
        transactions (dataframe)  :                  The trasanction data farame         
        
        
        ---RETURNS---
        Returns the dataframe of REVERSAL transactions         
        """
        
        reversal_trans = transactions[transactions['transactionType']=='REVERSAL']
        
        return reversal_trans
    
    
    def find_multiswipe_transactions(self, transactions, dt_threshold = 120, return_table=None, verbose=False):

        """
        This function finds multi-swipe transactions based on the difference between 
        trasaction times. This difference can be changed in dt_threshold.
        Looks at the repeated transactions.
        Assumptions for repeated transactions:
        1- Happend on the same day
        2- Happened consecutively (difference in time between transaction is determined by dt_threshold argument)
        3- Transaction amount is the same
        4- Merchant name is the same
        5- transaction amount is non-zero (the zeros are for verification purposes)
        6- account number is the same 
        
        ---ARGUMENTS---
        transactions (dataframe)  :                  The trasanction data farame 
        dt_threshold (int seconds):                  Time difference in seconds between consecutive transactions
                                                     for it to be considered multiswipe default values is 120 seconds
        return_table (str)        :                  Flag to return all duplicated transactions, multiswipe transactions
                                                     and duplicate transactions with delta t column
        verbose (bool)            :                  Prints some extra information (older version of the code)
        
        ---RETURNS---
        Returns the dataframe of multiswipe transactions (iv return_table='all' then it returns the intermediate dataframes
        also: dataframe of duplicates, dataframe of duplicates with delta_time column)
        """
        
        #First split trasactiondatetime into date and time columns and remove trasactionDateTime from df
        transactions = self.separate_datetime(transactions)
        
        #first consider columns that are not the same accross multiswipe transactions 
        cols_to_consider = ['accountNumber', 'date', 'transactionAmount', 'merchantName']

        #----------------------------------------older code----------------------------------------#
        # cols_to_consider = []    
        # for column_name in transactions.columns:
        #     #condition excludes the columns that are potentially different between duplicates
        #     condition = column_name not in ['time', 'transactionType', 'availableMoney', 'currentBalance']
        #     if condition:
        #         print (f'\nentered if condition for: {column_name}')
        #         cols_to_consider.append(column_name)
        # if verbose:
        #     print (cols_to_consider)
        #------------------------------------------------------------------------------------------

        #create a deep copy of the dataframe
        transactions_copy = transactions.copy(deep=True)
        
        #take only relevant columns
        transactions_no_time = transactions_copy[cols_to_consider]

        #take both the original and the repeated values and nonzero transaction amount
        duplic_idx = np.where((transactions_no_time.duplicated(keep=False).values==True) & 
                              (transactions_no_time['transactionAmount'].values!=0.0))[0]
        if verbose:            
            print (f"shape of duplic_idx: {duplic_idx.shape}")
            print (f"shape of unique duplic_idx: {np.unique(duplic_idx).shape}")
        
        #Take only the relevant rows
        duplicated_transactions = transactions_copy.loc[duplic_idx]
        
        #exclude reversals (upto this point we have excluded transaction for address approval)
        #now our dataset only has PURCHASE and REVERSAL flags. Since reversal's are already flagged
        #we only consider purchase for multiswipes 
        duplicated_transactions = duplicated_transactions[duplicated_transactions['transactionType']=='PURCHASE']

        #create a datetime column from date and time columns
        datetime_col = pd.to_datetime(duplicated_transactions['date'].astype(str)+' '+duplicated_transactions['time'].astype(str))
        
        #convert all the values to seconds in this column. This will be used for tracking time elapsed since last transaction
        min_datetime = datetime_col.min()
        datetime_col = datetime_col.apply(lambda x: (x-min_datetime).total_seconds() )
        
        #adding this as a column of duplic_trans and call it time_seconds
        duplicated_transactions['tranRelativeSeconds'] = datetime_col   
        
        #Now group by account number, then for each account number order the transactions.
        #aftrerwards take the difference of the transaction times and append to the df
        #concatenate the dataframe for different account numbers 
        
        j=0
        for i in duplicated_transactions['accountNumber'].unique():
            temp_df = duplicated_transactions[duplicated_transactions['accountNumber']==i]
            temp_df = temp_df.sort_values(by=['tranRelativeSeconds'])
            temp_df["dtTransaction"] = temp_df['tranRelativeSeconds'].diff(1)
            if j==0:
                duplic_trans_with_dt = temp_df
            else:
                duplic_trans_with_dt = pd.concat([duplic_trans_with_dt, temp_df])
            j+=1
        
        #apply dt threshold to filter out the multiswipe transactions (based on our definition
        #of being only a few secondsa apart. Here we have 120seconds as the default value)
        multiswipe_trans = duplic_trans_with_dt[duplic_trans_with_dt['dtTransaction']<=dt_threshold]
        
        #condition on what to return. 
        if return_table != None:
            return multiswipe_trans, duplicated_transactions, duplic_trans_with_dt
        else:
            return multiswipe_trans    


    def reversal_multiswipe_info(self, transactions, fraud_info = False, *args, **kwargs):
        
        try:
            reversal_trans = self.reversal_trans
            multiswipe_trans = self.multiswipe_trans
        except:
            print('Creating reversal_trans and multiswipe_trans')
            self.reversal_trans =  self.find_reverse_transactions(transactions)
            self.multiswipe_trans = self.find_multiswipe_transactions(transactions, dt_threshold = 120)
        print ('\n'+30*'-'+'Transaction numbers'+30*'-'+'\n')
        print(f"total number of multiswipe trasactions: {multiswipe_trans.shape[0]}")
        print(f"total number of reversal trasactions: {reversal_trans.shape[0]}")

        print(f"percentage of multiswipe trasactions: {100*multiswipe_trans.shape[0]/transactions.shape[0]:.2f}")
        print(f"percentage of reversal trasactions: {100*reversal_trans.shape[0]/transactions.shape[0]:.2f}")
        
        
        print ('\n'+30*'-'+'Transaction dollar amounts'+30*'-'+'\n')
        print (f"Total dollar amount of multiswipe trasnactions: {multiswipe_trans['transactionAmount'].sum():.2f}")
        print (f"Total dollar amount of reverse trasnactions: {reversal_trans['transactionAmount'].sum():.2f}")

        print (f"Percentage dollar amount of multiswipe trasnactions: \
               {100*multiswipe_trans['transactionAmount'].sum()/transactions['transactionAmount'].sum():.2f}")
        print (f"Percentage dollar amount of reverse trasnactions: \
               {100*reversal_trans['transactionAmount'].sum()/transactions['transactionAmount'].sum():.2f}")        
        print('')
        if fraud_info:
            reversal_vals = reversal_trans['isFraud'].value_counts().values
            multiswipe_vals = multiswipe_trans['isFraud'].value_counts().values
            alltrans_vals = transactions['isFraud'].value_counts().values
            print ('\n'+30*'-'+'Percentage of Fraud in each transaction'+30*'-'+'\n')
            print(f"Reversal: {reversal_trans['isFraud'].value_counts().index} , values: {reversal_vals}, fraud percentage: \
                  {100*reversal_vals[1]/reversal_vals.sum():.2f}")
            print(f"Multiswipe: {multiswipe_trans['isFraud'].value_counts().index}, values: {multiswipe_vals}, fraud percentage: \
                  {100*multiswipe_vals[1]/multiswipe_vals.sum():.2f}")
            print(f"All transactions: {transactions['isFraud'].value_counts().index}, values: {alltrans_vals}, fraud percentage: \
                  {100*alltrans_vals[1]/alltrans_vals.sum():.2f}")            


    def drop_singleval_cols(self, transactions):
        """
        This function is redundant as we have the drop_extra_cols function.
        Perform cleaning of the data and removing the columns that have no impa-
        ct(columns that are either empty or have the same unique value acroos t-
        he whole data).  
        
        ---ARGUMENTS---
        transactions (dataframe)  :                  The trasanction data farame         
        
        
        ---RETURNS---
        This function does not return anything. it creates a new attribute for 
        the class called transactions_no1s.
        
        """
        
        print ('Finding single valued columns to remove later')
        self.single_valued = transactions.nunique()[transactions.nunique()<2]
        self.transactions_no1s = transactions.copy(deep=True).drop(columns=self.single_valued.index)
        return
    
    def plot_fraud_in_rev_mult(self):
        
        """
        Creates a two panel plot showing the fraud and non-fraud distribution 
        in normal scale distribution and log scaled distribution for the reversal
        transactions and multiswipe transactions
        
        
        ---ARGUMENTS---
        
        
        ---RETURNS---
        prints the plot of fraud and non-fraud transactions in for reversal and 
        multiswipe categories.
        
        """

        x = ["reversal", "mutiswipe"]
        y_True = [self.reversal_trans['isFraud'].value_counts().values[1], self.multiswipe_trans['isFraud'].value_counts().values[1]]
        y_False = [self.reversal_trans['isFraud'].value_counts().values[0], self.multiswipe_trans['isFraud'].value_counts().values[0]]
        fix, ax = plt.subplots(1,2, figsize=(15,5))

        # plot bars in stack manner
        ax[0].set_title('normal scale distribution')
        ax[0].bar(x, y_True, color='red', alpha=0.5)
        ax[0].bar(x, y_False, bottom=y_True, color='green', alpha=0.5)
        ax[0].legend([ "Fraud", "Non-Fraud" ])

        ax[1].set_title('log scaled distribution')
        ax[1].bar(x, y_True, color='red', alpha=0.5)
        ax[1].bar(x, y_False, bottom=y_True, color='green', alpha=0.5)
        ax[1].set_yscale('log')
        ax[1].set_ylim([10**0, 10**5])
        ax[1].legend([ "Fraud", "Non-Fraud" ])
        plt.show()
        return
    
    def plot_transaction_amount(self):  
        
        """
        Cretes a two panel plot of transaction amount histograms. One in normal 
        scale and another in logscale
        
        ---ARGUMENTS---
        
        
        
        ---RETURNS---
        prints the plot
        
        """
        
        fig, ax = plt.subplots(1,2,figsize=(17,7))
        nbins = 20
        #---------------------------------------------------------------------
        hist_arr = ax[0].hist(self.raw_transactions["transactionAmount"], bins=nbins)#.hist(ax = ax[0], bins=20)
        ax[0].set_title('histogram of transactions unscaled', size=20)
        ax[0].set_xlabel('Transaction amount', size=20)
        ax[0].set_ylabel('number of transaction', size=20)
        #---------------------------------------------------------------------
        ax[1].hist(self.raw_transactions["transactionAmount"], bins=nbins)
        ax[1].set_title('histogram of transactions log scaled', size=20)
        ax[1].set_xlabel('Transaction amount', size=20)
        ax[1].set_ylabel('number of transaction', size=20)
        ax[1].set_yscale('log')
        plt.show()    
        
        return

    
    def plot_dt_distrib(self):
        """
        Incompolete, might come back to it later
        """
        pass
    
        log_bins = np.logspace(start=np.log10(1), stop=np.log10(10**10), num=50)
        plt.title('Time difference between consecutive transactions')
        reversal_new_duplic['dtTransaction'].hist(label = 'reversal', histtype='step', bins=log_bins)
        purchase_new_duplic['dtTransaction'].hist(label = 'purchase', histtype='step', bins=log_bins)

        # fraud_trans['dtTransaction'][fraud_trans['dtTransaction']<40000000000].hist(label = 'fraud', histtype='step', bins=log_bins)
        # non_fraud_trans['dtTransaction'][non_fraud_trans['dtTransaction']<4000000000].hist(label = 'non fraud', histtype='step', bins=log_bins)

        fraud_trans['dtTransaction'].hist(label = 'fraud', histtype='step', bins=log_bins)
        non_fraud_trans['dtTransaction'].hist(label = 'non fraud', histtype='step', bins=log_bins)
        # unmarked_new_duplic['dtTransaction'].hist(label = 'unmarked', histtype='step')
        plt.legend()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('dt(s) between consecutive transactions')    
        
        
        
    def check_equiv_cols(self):
        
        """
        ---ARGUMENTS---
        
        ---RETURNS---
        prints if columns are oneto one or not
        
        """
        
        print ("Is acqCountry and merchantCountryCode one to one: ",transactions['acqCountry'].equals(transactions['merchantCountryCode']))
        print ("Is customerId and accountNumber one to one: ",transactions["customerId"].equals(transactions["accountNumber"]))        
        return



    def bool_to_int(self, transactions):
        """
        Converts all boolean values to integers 0 and 1
        
        ---ARGUMENTS---
        transactions (dataframe):                the transaction dataframe
        
        ---RETURNS---
        transactions with bool converted to int 
        
        """
        bool_cols = transactions.select_dtypes(include='bool').columns
        for col in bool_cols:
            transactions[col] = transactions[col].astype(int) 
        return transactions


    def drop_extra_cols(self,transactions):
        
        """
        Drops one to one columns and signle valued columns
        
        ---ARGUMENTS---
        transactions (dataframe):                the transaction dataframe
        
        ---RETURNS---
        cleaned_trasactions that have only 22 columns instead of 29 columns
        in the raw transactions file
        
        """
        #note the function check_equiv_cols shows that customerId and accountNumber
        #are equals 
        print ('Dropping columns: customerId, and single-valued columns')
        col_to_drop = ['customerId']
        uniques = transactions.nunique()[transactions.nunique()<2]
        for i in range(len(uniques)):
            if uniques[i]==1:
                # print (uniques[i], uniques.index[i])
                col_to_drop.append(uniques.index[i])
        formatted_transactions = transactions.copy(deep=True).drop(columns=col_to_drop)
        return formatted_transactions

    def dates_to_datetime(self, transactions, return_col_names=False, verbose=False):
        """
        Converts all the date columns with string values in them to 
        datetime values. Coversion to datetime makes it easier to convert to 
        oordinal and numerical variables that are later used by our modeling section
        
        ---ARGUMENTS---
        transactions (dataframe)[MxN]:                transaction date frame               
        return_col_names (bool)      :                flag to return column names 
                                                      that are dated
        verbose (bool)               :                flag to print extra information 
        
        ---RETURNS---
        transactions with all the column with date entries converted to datetime
        objects
        
        """
        
        transactions_ = transactions.copy(deep=True)
        #Let's find columns that have datetime like values in them. These will later be
        #converted to seconds
        col_with_dates = []
        for i in range(len(transactions_.columns)):
            
            #conditions to programmatically find columns with date time like values
            #these columns either have date in them and their elements are longer 
            #than 2 in length e.g. 12/2/3
            condition = ('date' in transactions_.columns[i].lower()) & (len(str(transactions_[transactions_.columns[i]][0]))>2)
            
            if condition:
                col_with_dates.append(transactions_.columns[i])
                transactions_[transactions_.columns[i]] = pd.to_datetime(transactions_[transactions_.columns[i]])
                
        if verbose:
            print (col_with_dates)
        if return_col_names:
            return transactions_, col_with_dates
        return transactions_


    def datetime_to_ordinal(self, transactions):
        """
        Converts the datetime columns to ordinal variables. To do this this fun-
        ction find the global minimum of all the dates in the dataframe and sub-
        tracts it from all the other timedate values. Then it replaces those va-
        lues with the total number of seconds they correspond to. This is done 
        to get numerical values that can be scaled and input into the model.
        
        ---ARGUMENTS---
        transactions (dataframe):                   dataframe of the data
        
        
        ---RETURNS---
        transactions data with the dattime columns converted to total seconds
        
        """


        _, col_with_date = self.dates_to_datetime(transactions, return_col_names=True, verbose=False)

        min_datetime = transactions[col_with_date].min().min()
        for i in range(len(col_with_date)):
            transactions[col_with_date[i]] = transactions[col_with_date[i]].apply(lambda x: (x-min_datetime).total_seconds() )   
        return transactions


    def treat_empty_rows(self, transactions, impute=True, verbose=True):

        """
        This function looks at the rows with empty values and performs imputati-
        on of the data or deletion of the rows with missing values. 
        
        ---ARGUMENTS---
        transactions (dataframe):                   dataframe of the data
        impute (bool)           :                   imputation flag, if True the missing values are imputed
                                                    if False, the rows with missing values are dropped
        verbose(bool)           :                   If True prints extra information
         
        ---RETURNS---
        transactions with empty values either imputed or dropped, and array of e-
        mpty values indices.
        
        """
        #to print warning for when missin values are larger than 5%
        import warnings
        
        
        #indices of empty values
        empty_vals_idx = np.where(transactions.applymap(lambda x: x == ''))
        
        #create a copy of the data frame
        transactions_ = transactions.copy(deep=True)
        
        if verbose:        

            print(empty_vals[0].shape, empty_vals_idx[1].shape, fd.transactions.shape)
            print (f"percentage of empty rows: {len(empty_vals_idx[0])/ fd.transactions.shape[0]}")
            print( f"unique values of 0th element: {np.unique(empty_vals_idx[0])}")
            print( f"unique values of 1st element: {np.unique(empty_vals_idx[1])}")

        if impute:

            print ("Imputing the missing values with the Mode")

            cols_to_impute = transactions_.iloc[:,np.unique(empty_vals_idx[1])]
            name_cols_to_impute = cols_to_impute.columns

            for name_col in name_cols_to_impute:

                    if (transactions_[name_col]=='').mean()*100>5:
                        
                        #the following warning is if imputation is being done on entries that constitute more
                        #than 5% of the total population. If this is the case it might bias the model towards
                        #the mode of data
                        warnings.warn("Warning: Missing values are more than 5% of population, imputation might impact results")        

                    print (name_col)
                    transactions_[name_col] = transactions_[name_col].replace('', transactions_[name_col].mode()[0])

        else:
            #drop the rows that contain an empty value
            transactions_ = transactions_.drop(np.unique(empty_vals_idx[0]), axis=0)

        return transactions_, empty_vals_idx


    def stringint_to_int(self, transactions):
        
        """
        convert the strings-integers to integers which are specifically in found
        four columns
        
        ---ARGUMENTS---
        transactions (dataframe):                   dataframe of the data
        verbose(bool)           :                   If True prints extra information
         
        ---RETURNS---
        transactions with columns that have string-integers converted to integers        
        """
        
        stringint_cols = ["cardLast4Digits", "cardCVV", "enteredCVV", "accountNumber"]
        for column in stringint_cols:
            transactions[column] = transactions[column].astype(int) 
        return transactions

    
    def do_label_encoding (self, imputed_trasactions_int):
        
        """
        Label encoding fom scikit-learn is used to encode the object type columns
        with large number of entries (columns= nerchantName, merchantCategoryCode)
        
        ---ARGUMENTS---
        imputed_transactions_int (dataframe):         dataframe of the data with all the string-integers
                                                      converted to integers and imputation or deletion of
                                                      rows with empty values already implemented.
         
        ---RETURNS---
        returns the transaction data with label encoding of for merchantName and 
        merchantCategoryCode columns
        """        
        
        le = preprocessing.LabelEncoder()
        cols = ['merchantName', 'merchantCategoryCode']
        imputed_trasactions_int[cols] = imputed_trasactions_int[cols].apply(le.fit_transform)        
        
        return imputed_trasactions_int
    
    def one_hot_enc(self, label_enc_transactions):

        """
        One hot encoding is used to encode the object type columns with smaller 
        number of entries ("acqCountry", "merchantCountryCode", "posEntryMode", 
        "posConditionCode", "transactionType") into new one-hot-encoded columns
        
        ---ARGUMENTS---
        label_enc_transactions (dataframe):         dataframe of the data with all the string-integers
                                                    converted to integers and imputation or deletion of
                                                    rows with empty values already implemented and label
                                                    encoding already done on large categorical columns
         
        ---RETURNS---
        returns the transaction data frame with one hot encoding done on the co-
        lumns mentioned above. 
        """        
        one_hot_enc_transactions = label_enc_transactions.copy(deep=True)
        ohenc = preprocessing.OneHotEncoder()
        ohenc_cols = ["acqCountry", "merchantCountryCode", "posEntryMode", "posConditionCode", "transactionType"]
        # imputed_trasactions_int[cols] = 
        for col in ohenc_cols:
            dummies = pd.get_dummies(one_hot_enc_transactions[col])
            one_hot_enc_transactions = one_hot_enc_transactions.drop(columns=[col])
            one_hot_enc_transactions = pd.concat([one_hot_enc_transactions, dummies], axis=1)
        return one_hot_enc_transactions

    
    
    def scale_data(self, transactions_numeric):
        """
        Its always good practice to scale the data before putting it into an ML
        algorithm
        
        ---ARGUMENTS---
        transactions_numeric (dataframe):         dataframe with all the entries
                                                  converted to numeric values
         
        ---RETURNS---
        scaled values of the dataframe feature set. Note: this excludes the targets 

        """
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

        X = transactions_numeric.drop('isFraud', axis=1)
        y = transactions_numeric['isFraud']

        scaled_X = scaler.fit_transform(X)
        return scaled_X, y


    def train_test_val(self, transactions_numeric, **kwargs):
        
        """
        Uses the scale_data function to scale the data first and then creates t-
        rain, test, and validation sets. 
        
        ---ARGUMENTS---
        transactions_numeric (dataframe):         dataframe with all the entries
                                                  converted to numeric values
         
        ---RETURNS---
        scaled values of the dataframe feature set. Note: this excludes the targets 

        """
        
        
        # In the first step we will split the data in training and remaining dataset
        X, y = self.scale_data(transactions_numeric)
        X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)

        # Now since we want the valid and test size to be equal (10% each of overall data). 
        # we have to define valid_size=0.5 (that is 50% of remaining data)
        test_size = 0.5
        X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

        if kwargs.get('verbose'):
            print (f"X_train: {X_train.shape}")
            print (f"X_valid: {X_valid.shape}")
            print (f"X_test: {X_test.shape}")
            print (f"y_train: {y_train.shape}")
            print (f"y_valid: {y_valid.shape}")
            print (f"y_test: {y_test.shape}")



        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    
    
    

    def basic_regression(self, X_train, y_train):
        """
        This function takes as input formatted transaction where everything is 
        numerical. It perfomrs simple logistic regression on the data
        
        ---ARGUMENTS---
        X_train (ndarray)[MxN]:                Training feature set
        y_train (ndarray)[Mx1]:                Training targets
         
        ---RETURNS---
        Logistic regression fitted pipeline

        """

        pipeline = Pipeline([
            ('selector', VarianceThreshold()),
            ('classifier', LogisticRegression(random_state=RAND_STATE, n_jobs=-1))
            ])

        pipeline.fit(X_train, y_train)
        return pipeline    

    
    
    def fine_tuned_gboost(self, X_train, y_train):
        """
        This function takes as input formatted transaction where everything is 
        numerical. It perfomrs simple logistic regression on the data
        
        ---ARGUMENTS---
        X_train (ndarray)[MxN]:                Training feature set
        y_train (ndarray)[Mx1]:                Training targets
         
        ---RETURNS---
        Fitted gradient boosted classifier model

        """        
        
        gboost_model = GradientBoostingClassifier(random_state=RAND_STATE)
        gboost_pipeline = Pipeline([
            ('classifier', gboost_model),
            ])
        gboost_params = {
            'classifier__max_depth': [3, 5, 6],
            'classifier__max_features': [3, 4]
        }
        gboost_grid = GridSearchCV(gboost_pipeline, gboost_params, scoring='recall', cv=2, n_jobs=1, refit=True, verbose=10)
        gboost_grid.fit(X_train, y_train)

        return gboost_grid
    
    def simple_ann(self,  X_train, y_train, X_valid, y_valid, **kwargs):   
        
        
        #define model
        model=Sequential([
            Dense(units=16, input_dim=35, activation='relu'),
            Dense(units=24, activation='relu'), 
            Dropout (0.5), 
            Dense(units=20, activation='relu'), 
            Dense(units=24, activation='relu'),
            Dense(units=1 , activation='sigmoid')
            ])
        
        from tensorflow.keras.optimizers import Adam
        opt = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=kwargs.get('metrics'))      
        model.fit(
            X_train,
            y_train,
            batch_size=64,
            epochs=5,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(X_valid, y_valid),
            )
        return model
    
    def prep_data_all(self, transactions):
        """
        Does all the data preparation steps and prepares the data for input to
        the models
        
        ---ARGUMENTS---
        transactions(dataframe)[MxN]:                 Training feature set
         
        ---RETURNS---
        transaction data that goes through all the encoding steps
        """
        transactions = self.raw_transactions.copy(deep=True)
        print ('Converting bools to integer')
        print (transactions.shape)
        transactions = self.bool_to_int(transactions)
        print ('Dropping single valued and empty columns')
        print (transactions.shape)
        transactions = self.drop_extra_cols(transactions)
        print ('Coverting date columns to datetime')
        print (transactions.shape)
        transactions_datetime = self.dates_to_datetime(transactions, verbose=False)
        print ('Convertind datetime to seconds')
        transactions_seconds = self.datetime_to_ordinal(transactions_datetime)
        print ('Imputing and encoding')
        imputed_transactions, _ = self.treat_empty_rows(transactions_seconds, impute=False, verbose=False) 
        imputed_trasactions_int = self.stringint_to_int(imputed_transactions)
        label_enc_transactions = self.do_label_encoding (imputed_trasactions_int)
        one_hot_enc_transactions = self.one_hot_enc(label_enc_transactions)
        self.processed_data = one_hot_enc_transactions
        
        return 
        
    
    def data_sampler(self, X_train, y_train, sampler_type='over sample'):
        """
        Performs undersampling, oversampling, or SMOTE sampling 
        """

        X = np.asarray(X_train)
        y = np.asarray(y_train)
        
        if sampler_type=='under sample':
            print ('under sampling')
            print('Original dataset shape %s' % Counter(y))
            # Original dataset shape Counter({1: 900, 0: 100})
            rus = RandomUnderSampler(random_state=RAND_STATE)
            X_res, y_res = rus.fit_resample(X, y)
            print('Resampled dataset shape %s' % Counter(y_res))
            # Resampled dataset shape Counter({0: 100, 1: 100})

        elif sampler_type=='over sample':
            print ('over sampling')
            print('Original dataset shape %s' % Counter(y_train))
            # Original dataset shape Counter({1: 900, 0: 100})
            ros = RandomOverSampler(random_state=RAND_STATE)
            X_res, y_res = ros.fit_resample(X_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_res))
            # Resampled dataset shape Counter({0: 100, 1: 100})
        elif sampler_type=='SMOTE':
            print ('SMOTE sampling')
            print('Original dataset shape %s' % Counter(y_train))
            sm = SMOTE(random_state=RAND_STATE)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            print('Resampled dataset shape %s' % Counter(y_res))
        elif sampler_type == 'no sampling':  
            X_res, y_res = X_train, y_train
            
        return X_res, y_res

       
    def permform_modelling(self, transactions, sampler= 'SMOTE', model_name = 'gboost'):
        """
        Implements one of the mdoels as indicated by model name
        """
        transactions = self.raw_transactions.copy(deep=True)
        try:
            data = self.processed_data  
        except:
            self.prep_data_all(self.raw_transactions)
            data = self.processed_data
        self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test = self.train_test_val(data, verbose=True)
        print (f'The training data is imbalanced, performing {sampler}')
                
        self.X_res, self.y_res = self.data_sampler(self.X_train, self.y_train, sampler_type=sampler)
        
        if model_name=='gboost':
            gboost = self.fine_tuned_gboost(self.X_res, self.y_res)
            return gboost
        elif model_name=='reg':
            basic_regression = self.basic_regression(self.X_res, self.y_res)
            return basic_regression
        elif model_name=='ann':  
            simple_ann = self.simple_ann (self.X_train, self.y_train, self.X_valid, self.y_valid)
            return simple_ann
            
        
        return
        

    def model_results(self, trained_model, X_train=None, X_test=None, y_train=None, y_test=None):
        """
        Run this only after perform_modelling module to get the model results 
        """
        try:
            print(f"Best Parameters: {trained_model.best_params_}")
        except AttributeError:
            pass
        
        if (X_train ==None).all() | (X_test==None).all() | (y_train==None).all() | (y_test==None).all():                
            X_train = self.X_train
            X_valid = self.X_valid
            X_test = self.X_test
            y_train = self.y_train
            y_test = self.y_test
            y_valid = self.y_valid

        print(
        f"training score: {trained_model.score(X_train, y_train)}",
        f"testing score: {trained_model.score(X_test, y_test)}"
        )
        y_pred_prob = trained_model.predict_proba(X_test)[:, 1]
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_prob)}")
        # Uses AP for my threshold
        threshold = average_precision_score(y_test, y_pred_prob)
        print(f"{threshold=}")
        y_pred = trained_model.predict(X_test)
        predictions = np.where(y_pred_prob > threshold, 1, 0)
        print("y_test distribution", np.bincount(y_test))
        print("prediction distribution", np.bincount(predictions == 1))

        # Plot PR Curve
        plot_precision_recall_curve(trained_model, X_test, y_test)
        plt.show()

        # Plotting confusion Matrices
        f, axes = plt.subplots(1, 2, sharey=True, figsize=[15, 5])
        axes[0].set_title("Confusion Matrix 50/50")
        try:
            sns.heatmap(confusion_matrix(y_test.to_numpy(), y_pred), annot=True, fmt='g', ax=axes[0])
        except:
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g', ax=axes[0])
        axes[1].set_title("Confusion Matrix - Custom Threshold")
        try:
            sns.heatmap(confusion_matrix(y_test.to_numpy(), predictions), annot=True, fmt='g', ax=axes[1])
        except:
            sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='g', ax=axes[1])
        plt.show()    
        
        return
        
        
if '__name__'=='__main__':
    print ('Running directly')
else:
    print ("fraud_detector is imported")
    

    
    
    
    
    
    
    
#-------------------------------------------------------------------------------- 
        
# old code
# class FraudPlots(FraudDetection):
    
#     def __init__(self, **kwargs):
#         # this allows all the FraudDetection class vars to be defined here as self
#         super().__init__(**kwargs) 
        
        
#     def fraud_in_rev_mult(self):

#         x = ["reversal", "mutiswipe"]
#         y_True = [self.reversal_trans['isFraud'].value_counts().values[1], self.multiswipe_trans['isFraud'].value_counts().values[1]]
#         y_False = [self.reversal_trans['isFraud'].value_counts().values[0], self.multiswipe_trans['isFraud'].value_counts().values[0]]
#         fix, ax = plt.subplots(1,2, figsize=(15,5))

#         # plot bars in stack manner
#         ax[0].set_title('normal scale distribution')
#         ax[0].bar(x, y_True, color='red', alpha=0.5)
#         ax[0].bar(x, y_False, bottom=y_True, color='green', alpha=0.5)
#         ax[0].legend([ "Fraud", "Non-Fraud" ])

#         ax[1].set_title('log scaled distribution')
#         ax[1].bar(x, y_True, color='red', alpha=0.5)
#         ax[1].bar(x, y_False, bottom=y_True, color='green', alpha=0.5)
#         ax[1].set_yscale('log')
#         ax[1].set_ylim([10**0, 10**5])
#         ax[1].legend([ "Fraud", "Non-Fraud" ])
#         plt.show()
#         return
        
#--------------------------------------------------------------------------------            


#     def treat_empt_vals(self, dataframe, drop=False):

#         #find indices of cells with empty values        
#         self.empty_vals = np.where(fd.transactions.applymap(lambda x: x == ''))
#         if drop:
#             dropped_empty_rows = fd.transactions.drop(self.empty_vals[0], axis=0
#                                                      )
#             return dropped_empty_rows
            
#         else:
#             print ("imputing the data with the most common value for categoric\
#                    al and average for continuous data")
#             #fill this out later        

    

# def isOneToOne(df, col1, col2):
#     """
#     This function sees if two columns are on to one. It does so by using groupb-
#     y functionality on both of columns separately and then comparing their count
    
    
#     --- Argument ---
#     df: the data frame
#     col1: first column that needs to be checked for one to one-ness
#     col2: second column to be compared to col1 for one-to-one-ness
    
#     --- Return ---
#     returns true or false depending on col1 and col2 bine one to one 
#     """
#     first = df.groupby(col1)[col2].count().max()
#     second = df.groupby(col2)[col1].count().max()
#     return first + second == 2
#--------------------------------------------------------------------------------