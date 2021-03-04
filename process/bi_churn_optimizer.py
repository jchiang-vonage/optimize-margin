#from upsell.configuration import config
import logging
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import keras as ks
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
import time
import joblib
from keras import backend as K
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA, TruncatedSVD
import collections
import snowflake.connector #!pip install snowflake-connector-python
import pandas as pd
import numpy as np
from pyramid.arima import auto_arima #!pip install pyramid.arima
# from pyramid import auto_arima
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class BiChurn:
    def __init__(self):
        self.data_file_directory = 'data_file/'
        self.data_file_monthly = 'df_churn_2021_jan_mx.csv'
        self.data_file_yearly = 'df_churn_yearly_mx.csv'
        self.train_test_ratio=0.8
        self.look_back=30
        self.num_prediction = 14

    def featureSelection(self,datain, target, tops):
        model = LogisticRegression() #SVR(kernel="linear")
        print("RFE")
        #rfe = RFE(model, tops)
        rfe = RFECV(estimator=DecisionTreeClassifier())
        rfe = rfe.fit(datain, target.astype(int))
        print(rfe.ranking_)

    # def yb_featureSelection(self,datain, target):
    #     # Create a dataset with only 3 informative features
    #     # Instantiate RFECV visualizer with a linear SVM classifier
    #     visualizer = RFECV(SVC(kernel='linear', C=1))
    #     visualizer.fit(datain, target.astype(int))  # Fit the data to the visualizer
    #     visualizer.show()  # Finalize and render the figure
    #
    def chi2Selection(self,datain, target):  # higher score is better.
        # feature extraction
        print("Chi2")
        test = SelectKBest(score_func=chi2)
        # print(datain)
        fit = test.fit(datain, target.astype(int))
        # summarize scores
        np.set_printoptions(precision=3, suppress=True)
        print("scores:",fit.scores_)
        print("pvalues:",fit.pvalues_)
        features = fit.transform(datain)
        # Transformed array. only look at first two rows here
        #print(features[0:2, :])

    def show_scores(self,Y_validation, predictions):
        print("Accuracy score: ", accuracy_score(Y_validation, predictions))
        classes = [0,1,2,3,4, 5, 6, 7,8,9]
        print("Confusion Matrix:")
        print(classes)
        print(confusion_matrix(Y_validation, predictions, labels=classes))
        # tp -true positive/correctly predicts the positive class(Wolf).
        # #tn -true negative/correctly predicts the negative class.
        # #precision =tp/tp+fp) #recall = tp/(tp+fn)
        # #F1 = 2 x (precision x recall)/(precision + recall)
        # #F1 score is the harmonic average of the precision and recall,
        print("Classification Report:")
        print(classification_report(Y_validation, predictions))

    def pcaSelection(self,datain, num=None):
        print("PCA")
        pca = PCA(n_components=num)
        fit = pca.fit(datain)
        print('pca.explained_variance_', pca.explained_variance_)#The amount of variance explained by each of the selected components.
        #print('pca.explained_variance_ratio_', pca.explained_variance_ratio_)#Percentage of variance explained by each of the selected components.
        #print(fit.components_)

    def get_key(self,dict,val):
        for key, value in dict.items():
            if val == value:
                return key
        return "key doesn't exist"

    #"111" means"1x1 grid, first subplot" and "234" means "2x3 grid, 4th subplot".
    def plot_all_dimensions(self,country):
        # Visualizing 3-D numeric data with Scatter Plots
        # dataset = pd.read_csv('../data_file/' + 'tw_2020.csv')
        dataset = pd.read_csv(self.data_file_directory+self.data_file)
        dataset = dataset[dataset.GATEWAY.notnull()]
        dataset = dataset[dataset.NETWORK_NAME.notnull()]
        dataset = dataset[dataset.COST.notnull()]
        if country != 'ALL':
            dataset = dataset[dataset.COUNTRY == country]
        fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(221, projection='3d')
        xs = dataset['REVENUE']
        ys = dataset['VOLUME']
        zs = dataset['MARGIN']
        # ax.scatter(xs, ys, zs, s=20, alpha=0.6, edgecolors='w')
        # plt.title("Year 2020 for "+country)
        # ax.set_xlabel('REVENUE($)')
        # ax.set_ylabel('VOLUME')
        # ax.set_zlabel('MARGIN')

        fig.add_subplot(222)
        plt.scatter(xs, ys,s=5)
        plt.title("2020 for "+country)
        plt.xlabel("REVENUE")
        plt.ylabel("VOLUME")

        fig.add_subplot(223)
        plt.scatter(xs, zs,s=5)
        #plt.title("2020 REVENUE/MARGIN for "+country)
        plt.xlabel("REVENUE($)")
        plt.ylabel("MARGIN")

        fig.add_subplot(224)
        plt.scatter(ys, zs,s=5)
        #plt.title("2020 VOLUME/MARGIN for "+country)
        plt.xlabel("VOLUME")
        plt.ylabel("MARGIN")

        plt.savefig(('../data_file/' + country+'_2020.png'))
        #plt.show()

    def plot_3_dimensions(self,country):
        # Visualizing 3-D numeric data with Scatter Plots
        #dataset = pd.read_csv('../data_file/' + 'tw_2020.csv')
        dataset = pd.read_csv(self.data_file_directory+self.data_file)

        if country != 'ALL':
            dataset = dataset[dataset.COUNTRY == country]
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        xs = dataset['REVENUE']
        ys = dataset['VOLUME']
        zs = dataset['MARGIN']
        ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors='w')
        ax.set_xlabel('REVENUE')
        ax.set_ylabel('VOLUME')
        ax.set_zlabel('MARGIN')
        plt.show()

        # fig = plt.figure()
        # fig.add_subplot(111)
        # plt.scatter(xs, ys)
        # plt.show()
        #
        # plt.scatter(xs, zs)
        # plt.show()
        #
        # plt.scatter(ys, zs)
        # plt.show()

    # b: blue    # g: green    # r: red    # c: cyan
    # m: magenta    # y: yellow    # k: black    # w: white
    def plot_LSA(self):
        #dataset = pd.read_csv('../data_file/' + 'tw_2020.csv')
        dataset = pd.read_csv(self.data_file_directory+self.data_file)
        lsa = TruncatedSVD(n_components=2)
        lsa.fit(dataset[['REVENUE','VOLUME','MARGIN']])
        lsa_scores = lsa.transform(dataset[['REVENUE','VOLUME','MARGIN']])
        color_mapper = {label: idx for idx, label in enumerate(set(dataset['NETWORK_NAME']))}
        color_column = [color_mapper[label] for label in dataset['NETWORK_NAME']]
        print('color_mapper:', color_mapper)
        counter = collections.Counter(color_column)
        print('counter.most_common(11):', counter.most_common(11))
        colors = ['green', 'black', 'red', 'orange', 'blue', 'yellow', 'cyan', 'pink', 'gray', 'purple','white']
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=color_column,
                    cmap=matplotlib.colors.ListedColormap(colors))
        cnt = 0
        a = [0] * len(colors)
        for key, value in color_mapper.items():
            a[cnt] = mpatches.Patch(color=colors[value], label=key)
            cnt = cnt + 1
        plt.legend(handles=[a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],a[10]], prop={'size': 30})

    def dynamic_select(self):
        print('real time connection to snowflake')
        country= "mx"
        sql_a = '''SELECT
	b.MAIN_ACCOUNT_ID_IN_SOURCE AS "Api Key",
	b."Main Account Name" AS "Company Name",
	cm.REGION AS "Country Region",
	COALESCE(acct2.PFD_ON_SMS_C, 'No') AS "PFD Customer",
	acct.PRICING_FX_RATE_C AS "Fixed FX Rate",
	COALESCE(er.CONVERSION_RATE,1) as "Finance Rate",
	COALESCE(b.ACCOUNT_REGION, 'NA') AS "Account Region",
	sms_daily."Sub Key",
	sms_daily."Network Code",
	sms_daily."Network Name",
	sms_daily."Country Code",
	sms_daily."Country Name",
	sms_daily."Gateway Name",
	sms_daily."Usage Date",
	sms_daily."Volume",
	sms_daily."Margin (EUR)",
	sms_daily."Margin (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Margin (USD)",
	sms_daily."Revenue (EUR)",
	sms_daily."Revenue (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Revenue (USD)",
	sms_daily."Cost (EUR)",
	sms_daily."Cost (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Cost (USD)",
	sms_daily."Cost SMS (EUR)",
	sms_daily."Cost SMS (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Cost SMS (USD)",
	sms_daily."Cost HLR (EUR)",
	sms_daily."Cost HLR (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Cost HLR (USD)",
	CASE WHEN "PFD Customer" = 'Yes' AND sms_daily."Volume" <> 0
		THEN (sms_daily."Revenue (EUR)" / sms_daily."Volume") * dlr."Delivered" 
		ELSE sms_daily."Revenue (EUR)" END AS "Revenue (EUR) pFD",
	("Revenue (EUR) pFD" * COALESCE(acct.PRICING_FX_RATE_C /er.CONVERSION_RATE ,1)) AS "Revenue (EUR) pFD Adjusted",
	("Revenue (EUR) pFD" * COALESCE(acct.PRICING_FX_RATE_C/er.CONVERSION_RATE ,1) - sms_daily."Cost (EUR)") AS "Margin (EUR) pFD Adjusted",
	"Revenue (EUR) pFD" * coalesce(to_number(er_usd.CONVERSION_RATE,20,4),1) AS "Revenue (USD) pFD",
	"Revenue (EUR) pFD Adjusted" * coalesce(to_number(er_usd.CONVERSION_RATE,20,4),1) AS "Revenue (USD) pFD Adjusted",
	"Margin (EUR) pFD Adjusted" * coalesce(to_number(er_usd.CONVERSION_RATE,20,4),1) AS "Margin (USD) pFD Adjusted",
	dlr."Sent",
	dlr."Passed To Gateway",
	dlr."Delivered",
	dlr."Undelivered",
	dlr."Failed",
	dlr."Expired",
	dlr."Rejected",
	dlr."Deleted",
	'Outbound' AS "Direction",
	COALESCE(acct.FX_RATE_TYPE_C , 'NA') AS "Customer Type"
FROM (SELECT 
	ACCOUNT_ID AS "Sub Key",
	"NETWORK" AS "Network Code",
	NETWORK_NAME AS "Network Name",
	COUNTRY AS "Country Code",
	COUNTRY_NAME AS "Country Name",
	GATEWAY AS "Gateway Name",
	to_date(DTTM) AS "Usage Date",
	sum("VOLUME") AS "Volume",
	sum(MARGIN) AS "Margin (EUR)",
	sum(REVENUE) AS "Revenue (EUR)",
	sum(COST) AS "Cost (EUR)",
	sum(COST_SMS) AS "Cost SMS (EUR)",
	sum(COST_HLR) AS "Cost HLR (EUR)"
FROM edw.DMART.NXM_DETAIL_SMS_DAILY a
WHERE to_date(dttm) >= '2020-01-01' AND product_detail <> 'Inbound SMS'
--	AND "Sub Key" = 'f13b21d9'
GROUP BY 1,2,3,4,5,6,7) sms_daily
	LEFT JOIN (
	SELECT 
		ACCOUNTID AS "Sub Key",
		"NETWORK" AS  "Network Code",
		GATEWAY AS "Gateway Name",
		to_date(DLR_DATE) AS "Usage Date",
		COUNTRY AS "Country Code",
		sum(COALESCE(REQUESTED,0)) AS "Sent",
		sum(COALESCE(PASSED_TO_GATEWAY,0)) AS "Passed To Gateway",
		sum(COALESCE(DELIVERED,0)) AS "Delivered",
		sum(COALESCE(UNDELIVERED,0)) AS "Undelivered",
		sum(COALESCE(FAILED,0)) AS "Failed",
		sum(COALESCE(EXPIRED,0)) AS "Expired",
		sum(COALESCE(REJECTED,0)) AS "Rejected",
		sum(COALESCE(DELETED,0)) AS "Deleted"
	FROM EDW.DMART.NXM_DLR
	WHERE to_date(DLR_DATE) >= '2020-01-01' 
--	and to_date(DLR_DATE) <= '2020-01-31'
--		AND "Sub Key" = 'f13b21d9
	GROUP BY 1,2,3,4,5) dlr
		ON sms_daily."Usage Date" = dlr."Usage Date"
			AND sms_daily."Sub Key" = dlr."Sub Key"
			AND sms_daily."Network Code" = dlr."Network Code"
			AND sms_daily."Gateway Name" = dlr."Gateway Name"
			AND sms_daily."Country Code" = dlr."Country Code"
	LEFT JOIN edw.BASEDW.v_NXM_ZO_ACCOUNT acct
		ON sms_daily."Sub Key" = acct.ACCOUNT_NUMBER 
			AND COALESCE(acct.FX_RATE_TYPE_C , 'NA') = 'Fixed'
	LEFT JOIN edw.BASEDW.v_NXM_ZO_ACCOUNT acct2
		ON sms_daily."Sub Key" = acct2.ACCOUNT_NUMBER 
	JOIN edw.TABLEAU.V_NXM_ACCOUNT_INFORMATION b
		ON sms_daily."Sub Key" = b.ACCOUNT_ID_IN_SOURCE 
	LEFT JOIN (SELECT DISTINCT country_code, region FROM edw.BASEDW.EDW_API_COUNTRY_REGION_MAP) cm
		ON sms_daily."Country Code" = cm.COUNTRY_CODE 
	LEFT JOIN edw.TABLEAU.V_EXCHANGE_RATE er
		ON er.FROM_CURRENCY = 'EUR'
			AND er.TO_CURRENCY = acct.CURRENCY 
			AND to_date(sms_daily."Usage Date") BETWEEN er.RECORD_START_DATE AND er.RECORD_END_DATE 
	LEFT JOIN edw.TABLEAU.V_EXCHANGE_RATE er_usd
		ON er_usd.FROM_CURRENCY = 'EUR'
			AND er_usd.TO_CURRENCY = 'USD' 
			AND to_date(sms_daily."Usage Date") BETWEEN er_usd.RECORD_START_DATE AND er_usd.RECORD_END_DATE 
WHERE "Direction" = 'Outbound' AND sms_daily."Country Code" = 'MX'
   '''
        sql_b = '''SELECT
	b.MAIN_ACCOUNT_ID_IN_SOURCE AS "Api Key",
	b."Main Account Name" AS "Company Name",
	cm.REGION AS "Country Region",
	COALESCE(acct2.PFD_ON_SMS_C, 'No') AS "PFD Customer",
	acct.PRICING_FX_RATE_C AS "Fixed FX Rate",
	COALESCE(er.CONVERSION_RATE,1) as "Finance Rate",
	COALESCE(b.ACCOUNT_REGION, 'NA') AS "Account Region",
	sms_daily_inbound."Sub Key",
	sms_daily_inbound."Network Code",
	sms_daily_inbound."Network Name",
	sms_daily_inbound."Country Code",
	sms_daily_inbound."Country Name",
	sms_daily_inbound."Gateway Name",
	sms_daily_inbound."Usage Date",
	sms_daily_inbound."Volume",
	sms_daily_inbound."Margin (EUR)",
	sms_daily_inbound."Margin (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Margin (USD)",
	sms_daily_inbound."Revenue (EUR)",
	sms_daily_inbound."Revenue (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Revenue (USD)",
	sms_daily_inbound."Cost (EUR)",
	sms_daily_inbound."Cost (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Cost (USD)",
	sms_daily_inbound."Cost SMS (EUR)",
	sms_daily_inbound."Cost SMS (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Cost SMS (USD)",
	sms_daily_inbound."Cost HLR (EUR)",
	sms_daily_inbound."Cost HLR (EUR)" * COALESCE(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Cost HLR (USD)",
	sms_daily_inbound."Revenue (EUR)" AS "Revenue (EUR) pFD",
	("Revenue (EUR) pFD" * COALESCE(acct.PRICING_FX_RATE_C /er.CONVERSION_RATE ,1)) AS "Revenue (EUR) pFD Adjusted",
	("Revenue (EUR) pFD" * COALESCE(acct.PRICING_FX_RATE_C/er.CONVERSION_RATE ,1) - sms_daily_inbound."Cost (EUR)") AS "Margin (EUR) pFD Adjusted",
	"Revenue (EUR) pFD" * coalesce(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Revenue (USD) pFD", 
	"Revenue (EUR) pFD Adjusted" * coalesce(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Revenue (USD) pFD Adjusted",
	"Margin (EUR) pFD Adjusted" * coalesce(to_number(er_usd.CONVERSION_RATE,20,4), 1) AS "Margin (USD) pFD Adjusted",
	0 AS "Sent",
	0 AS "Passed To Gateway",
	0 AS "Delivered",
	0 AS "Undelivered",
	0 AS "Failed",
	0 AS "Expired",
	0 AS "Rejected",
	0 AS "Deleted",
	'Inbound' AS "Direction",
	COALESCE(acct.FX_RATE_TYPE_C , 'NA') AS "Customer Type"
FROM (SELECT 
	ACCOUNT_ID AS "Sub Key",
	"NETWORK" AS "Network Code",
	NETWORK_NAME AS "Network Name",
	COUNTRY AS "Country Code",
	COUNTRY_NAME AS "Country Name",
	GATEWAY AS "Gateway Name",
	to_date(DTTM) AS "Usage Date",
	sum("VOLUME") AS "Volume",
	sum(MARGIN) AS "Margin (EUR)",
	sum(REVENUE) AS "Revenue (EUR)",
	sum(COST) AS "Cost (EUR)",
	sum(COST_SMS) AS "Cost SMS (EUR)",
	sum(COST_HLR) AS "Cost HLR (EUR)"
FROM edw.DMART.NXM_DETAIL_SMS_DAILY 
WHERE to_date(dttm) >= '2020-01-01' AND product_detail = 'Inbound SMS'
GROUP BY 1,2,3,4,5,6,7) sms_daily_inbound
	LEFT JOIN edw.BASEDW.v_NXM_ZO_ACCOUNT acct
		ON sms_daily_inbound."Sub Key" = acct.ACCOUNT_NUMBER 
			AND COALESCE(acct.FX_RATE_TYPE_C , 'NA') = 'Fixed'
	LEFT JOIN edw.BASEDW.v_NXM_ZO_ACCOUNT acct2
		ON sms_daily_inbound."Sub Key" = acct2.ACCOUNT_NUMBER 
	JOIN edw.TABLEAU.V_NXM_ACCOUNT_INFORMATION b
		ON sms_daily_inbound."Sub Key" = b.ACCOUNT_ID_IN_SOURCE 
	LEFT JOIN (SELECT DISTINCT country_code, region FROM edw.BASEDW.EDW_API_COUNTRY_REGION_MAP) cm
		ON sms_daily_inbound."Country Code" = cm.COUNTRY_CODE 
	LEFT JOIN edw.TABLEAU.V_EXCHANGE_RATE er
		ON er.FROM_CURRENCY = 'EUR'
			AND er.TO_CURRENCY = acct.CURRENCY 
			AND to_date(sms_daily_inbound."Usage Date") BETWEEN er.RECORD_START_DATE AND er.RECORD_END_DATE 
	LEFT JOIN edw.TABLEAU.V_EXCHANGE_RATE er_usd
		ON er_usd.FROM_CURRENCY = 'EUR'
			AND er_usd.TO_CURRENCY = 'USD' 
			AND to_date(sms_daily_inbound."Usage Date") BETWEEN er_usd.RECORD_START_DATE AND er_usd.RECORD_END_DATE
WHERE "Direction" = 'Outbound' AND sms_daily_inbound."Country Code"= 'MX'
                '''
        conn = snowflake.connector.connect(host="ft32896.us-east-1.snowflakecomputing.com",
                                           user="ap_tableau_jchiang", password="x6WtwTrs6%ptrHFk",
                                           account="ft32896.us-east-1")
        df_a = pd.read_sql(sql_a , conn)
        df_b = pd.read_sql(sql_b , conn)
        df=pd.concat([df_a, df_b])
        #df.to_csv('../data_file/df_churn_'+country+'.csv')
        df.to_csv(self.data_file_directory+self.data_file_yearly)
        return df

    def data_analysis(self):
        print('data_analysis')
        dataset = pd.read_csv(self.data_file)
        dataset = dataset[dataset.COUNTRY == 'TW']
        dataset = dataset[dataset.MARGIN.notnull()]
        #print(dataset.head())
        #print(dataset.info())
        print(dataset.REVENUE.describe())

        plt.subplot(2, 2, 1)
        (dataset['MARGIN']).plot.hist(bins=50, figsize=(12, 6), edgecolor='white', range=[0, 250])
        plt.xlabel('MARGIN', fontsize=12)
        plt.title('MARGIN Distribution', fontsize=12)
        plt.subplot(2, 2, 2)
        np.log(dataset['MARGIN']+1).plot.hist(bins=50, figsize=(12, 6), edgecolor='white')
        plt.xlabel('log(MARGIN)', fontsize=12)
        plt.title('MARGIN Distribution', fontsize=12)
        plt.subplot(2, 2, 3)
        (dataset['VOLUME']).plot.hist(bins=50, figsize=(12, 6), edgecolor='white', range=[0, 250])
        plt.xlabel('VOLUME', fontsize=12)
        plt.title('VOLUME Distribution', fontsize=12)
        plt.subplot(2, 2, 4)
        np.log(dataset['VOLUME']+1).plot.hist(bins=50, figsize=(12, 6), edgecolor='white')
        plt.xlabel('log(VOLUME)', fontsize=12)
        plt.title('VOLUME Distribution', fontsize=12)
        plt.show()

        sns.boxplot(x='NETWORK_NAME', y=dataset['COST']/dataset['VOLUME'], data=dataset,
                    palette=sns.color_palette('RdBu', 5))
        plt.show()

        return dataset

    def svm(self):
        print('svm')
        dataset = pd.read_csv(self.data_file_directory+self.data_file_yearly)
        dataset = dataset[dataset.COUNTRY == 'TW']
        dataset = dataset[dataset.VOLUME.notnull()]
        from sklearn.svm import SVC
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report

        dataset['DTTM'] = pd.to_datetime(dataset['DTTM'])
        X_train, X_test, y_train, y_test = train_test_split(dataset['DTTM'], dataset['MARGIN'], test_size=0.20, shuffle=True)

        clf = SVC(gamma='auto')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_test, y_pred))
        plt.plot(y_test)
        plt.plot(y_pred, color='red')
        plt.show()

    def bi_plot(self,df):
        plt.plot(df)
        plt.xlabel('TIME')
        plt.ylabel(df.name+ '$')
        plt.plot()
        plt.show()

    def arima(self,file_name):
        print('arima')
        dataset = pd.read_csv(file_name)
        #dataset = dataset[dataset.COUNTRY == 'TW']
        dataset = dataset[['Usage Date', 'Margin (EUR) pFD Adjusted', 'Volume', 'Cost SMS (EUR)', 'Revenue (EUR) pFD Adjusted']]
        #dataset = dataset.groupby('Usage Date',as_index=False).agg({'Margin (EUR) pFD Adjusted':"sum"})
        dataset = dataset.groupby(['Usage Date']).sum()[['Margin (EUR) pFD Adjusted', 'Volume', 'Cost SMS (EUR)', 'Revenue (EUR) pFD Adjusted']]
        #one column becomes series
        # df_series = dataset.groupby(['DTTM']).sum()['MARGIN']
        # dataset=df_series.to_frame()
        #print(dataset.head(2))
        # self.bi_plot(dataset['Margin (EUR) pFD Adjusted'])
        # self.bi_plot(dataset['Volume'])
        # self.bi_plot(dataset['Cost SMS (EUR)'])
        # self.bi_plot(dataset['Revenue (EUR) pFD Adjusted'])

        #train, test = dataset['MARGIN'][:292], dataset['MARGIN'][292:]
        train_size = int(len(dataset['Margin (EUR) pFD Adjusted']) * self.train_test_ratio)
        test_size = len(dataset['Margin (EUR) pFD Adjusted']) - train_size
        train, test = dataset['Margin (EUR) pFD Adjusted'][:train_size], dataset['Margin (EUR) pFD Adjusted'][train_size:]
        whole_size = int(len(dataset))
        print(whole_size, train_size, test_size)
        #dataset.iloc[0:train_size,1] #1 means second column of data is MARGIN
        #train, test = dataset.iloc[0:train_size, 1], dataset.iloc[train_size:len(dataset['MARGIN']), 1]
        print(train.shape,test.shape)

        # Fit a simple auto_arima model
        model = auto_arima(train, start_p=1, start_q=1, start_P=1, start_Q=1,
                          max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=False,
                          stepwise=False, suppress_warnings=True, D=10, max_D=10,
                          error_action='ignore')
        # Create predictions for the future, evaluate on test
        preds, conf_int = model.predict(n_periods=test.shape[0], return_conf_int=True)
        # preds = modelb.predict(n_periods=test.shape[0], return_conf_int=False, alpha=0.05)
        # print(preds)
        r2score = r2_score(test, preds)
        print(r2score)
        print("ARIMA Test RMSE: %.3f" % np.sqrt(mean_squared_error(test, preds)))
        preds, conf_int = model.predict(n_periods=train.shape[0], return_conf_int=True)
        print("ARIMA Train RMSE: %.3f" % np.sqrt(mean_squared_error(train, preds)))
        # preds, conf_int = model.predict(n_periods=test.shape[0] + 60, return_conf_int=True)
        # # Plot the points and the forecasts
        # x_axis = np.arange(train.shape[0] + preds.shape[0])
        # x_years = x_axis + 1  # Year starts at
        # plt.plot(x_years[x_axis[:train.shape[0]]], train, alpha=0.75)
        # plt.plot(x_years[x_axis[train.shape[0]:]], preds, alpha=0.75)  # Forecasts
        # plt.scatter(x_years[x_axis[train.shape[0]:]], test.append([0] * 60), alpha=0.4, marker='x')  # Test data

        preds, conf_int = model.predict(n_periods=test.shape[0], return_conf_int=True)
        # Plot the points and the forecasts
        x_axis = np.arange(train.shape[0] + preds.shape[0])
        x_years = x_axis + 1  # starts at
        plt.plot(x_years[x_axis[:train.shape[0]]], train, alpha=0.75)
        plt.plot(x_years[x_axis[train.shape[0]:]], preds, alpha=0.75)  # Forecasts
        #plt.scatter(x_years[x_axis[train.shape[0]:]], test, alpha=0.4, marker='x')  # Test data
        plt.plot(x_years[x_axis[train.shape[0]:]], test, alpha=0.4)  # Test data
        #plt.fill_between(x_years[x_axis[-preds.shape[0]:]], conf_int[:, 0], conf_int[:, 1],alpha=0.1, color='b')
        plt.title("Margin (EUR) pFD Adjusted (ARIMA)")
        plt.xlabel("Time step (Test)")
        plt.plot()
        plt.show()

        # train_preds, train_conf_int = model.predict(n_periods=train.shape[0], return_conf_int=True)
        # # Plot the points and the forecasts
        # x_axis = np.arange(train.shape[0] + preds.shape[0])
        # x_years = x_axis + 1  # Year starts at
        # plt.plot(x_years[x_axis[:train.shape[0]]], train_preds, alpha=0.75)
        # plt.plot(x_years[x_axis[train.shape[0]:]], preds, alpha=0.75)  # Forecasts
        # plt.scatter(x_years[x_axis[train.shape[0]:]], test, alpha=0.4, marker='x')  # Test data
        # plt.fill_between(x_years[x_axis[-preds.shape[0]:]], conf_int[:, 0], conf_int[:, 1], alpha=0.1, color='b')
        # plt.title("Margin (EUR) pFD Adjusted forecasts")
        # plt.xlabel("Time step (Train)")
        # plt.plot()
        # plt.show()


    def create_time_series_dataset(self,dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
            # print(X,Y)
            # if i==10 : exit(0)
        return np.array(X), np.array(Y)

    def lstm(self,file_name):
        print('lstm')
        dataset = pd.read_csv(file_name)
        dataset = dataset[['Usage Date', 'Margin (EUR) pFD Adjusted', 'Volume', 'Cost SMS (EUR)', 'Revenue (EUR) pFD Adjusted']]
        dataset = dataset.groupby(['Usage Date']).sum()[['Margin (EUR) pFD Adjusted', 'Volume', 'Cost SMS (EUR)', 'Revenue (EUR) pFD Adjusted']]
        # self.bi_plot(dataset['Margin (EUR) pFD Adjusted'])

        dataset.sort_values(by=['Usage Date'])
        dataset_y=dataset['Margin (EUR) pFD Adjusted'].array
        dataset_y = np.reshape(dataset_y, (-1, 1))# -1 is unknown dimenstion
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_y = scaler.fit_transform(dataset_y)
        train_size = int(len(dataset_y) * self.train_test_ratio)
        test_size = len(dataset_y) - train_size
        train, test = dataset_y[0:train_size, :], dataset_y[train_size:len(dataset_y), :]
        whole_size = int(len(dataset_y))
        print(whole_size,train_size,test_size)

        X_train, Y_train = self.create_time_series_dataset(train, self.look_back)
        X_test, Y_test = self.create_time_series_dataset(test, self.look_back)
        X_whole, Y_whole = self.create_time_series_dataset(dataset_y, self.look_back)

        # reshape input to be [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
        X_whole = np.reshape(X_whole, (X_whole.shape[0], 1, X_whole.shape[1]))
        print(X_train.shape[0],X_train.shape[1], X_train.shape[2])

        #input shape=look_back time step=1, features=325
        print(X_train.shape[1], X_train.shape[2])
        model = Sequential()
        model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))#
        #model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

        history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=0, shuffle=False)
        # print(model.summary())
        # plt.figure(figsize=(8, 4))
        # plt.plot(history.history['loss'], label='Train Loss')
        # plt.plot(history.history['val_loss'], label='Test Loss')
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epochs')
        # plt.legend(loc='upper right')
        # plt.show();

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        whole_predict = model.predict(X_whole)
        # invert predictions
        train_predict = scaler.inverse_transform(train_predict)
        Y_train = scaler.inverse_transform([Y_train])
        test_predict = scaler.inverse_transform(test_predict)
        Y_test = scaler.inverse_transform([Y_test])
        whole_predict = scaler.inverse_transform(whole_predict)
        Y_whole = scaler.inverse_transform([Y_whole])

        print('LSTM Train RMSE:', np.sqrt(mean_squared_error(Y_train[0], train_predict[:, 0])))
        print('LSTM Test RMSE:', np.sqrt(mean_squared_error(Y_test[0], test_predict[:, 0])))

        # aa = [x for x in range(test_size-self.look_back-1)]
        # plt.figure(figsize=(8, 4))
        # plt.plot(aa, Y_test[0][:test_size], marker='.', label="actual")
        # plt.plot(aa, test_predict[:, 0][:test_size], 'r', label="prediction")
        # # plt.tick_params(left=False, labelleft=True) #remove ticks
        # plt.tight_layout()
        # sns.despine(top=True)
        # plt.subplots_adjust(left=0.07)
        # plt.ylabel('Margin (EUR) pFD Adjusted forecasts', size=15)
        # plt.xlabel('Time step (Test)', size=15)
        # plt.legend(fontsize=15)
        # plt.show();
        #
        # aa = [x for x in range(train_size-self.look_back-1)]
        # plt.figure(figsize=(8, 4))
        # plt.plot(aa, Y_train[0][:train_size], marker='.', label="actual")
        # plt.plot(aa, train_predict[:, 0][:train_size], 'r', label="prediction")
        # # plt.tick_params(left=False, labelleft=True) #remove ticks
        # plt.tight_layout()
        # sns.despine(top=True)
        # plt.subplots_adjust(left=0.07)
        # plt.ylabel('Margin (EUR) pFD Adjusted forecasts', size=15)
        # plt.xlabel('Time step (Train)', size=15)
        # plt.legend(fontsize=15)
        # plt.show();

        #
        # aa = [x for x in range(whole_size-self.look_back-1)]
        # plt.figure(figsize=(8, 4))
        # plt.plot(aa, Y_whole[0][:whole_size], marker='.', label="actual")
        # plt.plot(aa, whole_predict[:, 0][:whole_size], 'r', label="prediction")
        # # plt.tick_params(left=False, labelleft=True) #remove ticks
        # plt.tight_layout()
        # sns.despine(top=True)
        # plt.subplots_adjust(left=0.07)
        # plt.ylabel('Margin (EUR) pFD Adjusted (LSTM)', size=15)
        # plt.xlabel('Time step (whole)', size=15)
        # plt.legend(fontsize=15)
        # plt.show();

        # ARIM comarison the points and the forecasts
        x_axis = np.arange(whole_size)
        x_years = x_axis + 1  # Year starts at
        plt.plot(x_years[x_axis[:self.look_back]], [[0]]*self.look_back, alpha=0.75)
        plt.plot(x_years[x_axis[self.look_back:train_size-1]], Y_train[0][:train_size], alpha=0.75)
        plt.plot(x_years[x_axis[train_size:train_size+self.look_back]], [[0]]*self.look_back, alpha=0.75)  # 0
        plt.plot(x_years[x_axis[train_size+self.look_back+1:]], test_predict[:, 0][:test_size], alpha=0.75)  # Forecasts
        plt.plot(x_years[x_axis[train_size+self.look_back+1:]], Y_test[0][:test_size], alpha=0.4)  # Test data
        # plt.fill_between(x_years[x_axis[-preds.shape[0]:]], conf_int[:, 0], conf_int[:, 1],alpha=0.1, color='b')
        plt.title("Margin (EUR) pFD Adjusted (LSTM)")
        plt.xlabel("Time step (Test)")
        plt.plot()
        plt.show()

        prediction_result=[]
        dataset_future = dataset_y[-(self.look_back+2):,:]
        for i in range(self.num_prediction):
            X_future, Y_future = self.create_time_series_dataset(dataset_future, self.look_back)
            X_future = np.reshape(X_future, (X_future.shape[0], 1, X_future.shape[1]))
            future_predict = model.predict(X_future)
            future_predict_shape=np.reshape(future_predict, (1, -1))
            #future_predict_0=future_predict[0]
            dataset_future = np.append(dataset_future, future_predict_shape,axis=0)
            dataset_future=dataset_future[1:]
            future_predict_inverse = scaler.inverse_transform(future_predict)
            print("t",i,future_predict_inverse.reshape(-1,))
            #prediction_result.append(future_predict_inverse)      double array
            prediction_result.append(future_predict_inverse.reshape(-1,))

        #print(prediction_result)
        aa = [x for x in range(whole_size-self.look_back-1)]
        plt.figure(figsize=(8, 4))
        plt.plot(aa, Y_whole[0][:whole_size], marker='.', label="actual")
        #plt.plot(aa, whole_predict[:, 0][:whole_size], 'r', label="prediction")
        aa = [x for x in range(whole_size- self.look_back - 1, whole_size + self.num_prediction - self.look_back - 1)]
        plt.plot(aa, prediction_result, 'r', label="future")
        # plt.tick_params(left=False, labelleft=True) #remove ticks
        plt.tight_layout()
        sns.despine(top=True)
        plt.subplots_adjust(left=0.07)
        plt.ylabel('Margin (EUR) pFD Adjusted (LSTM)', size=15)
        plt.xlabel('Time step (future)', size=15)
        plt.legend(fontsize=15)
        plt.show();

    def margin_percent(self,percent):
        import pandas as pd
        import numpy as np
        from pyramid.datasets import load_lynx
        from pyramid.arima import auto_arima
        #from pyramid import auto_arima
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import r2_score

        print('margin_percent ',percent)
        dataset = pd.read_csv(self.data_file_directory+self.data_file_monthly)
        print(dataset.shape)
        #dataset = dataset[dataset.COUNTRY == 'TW']
        #dataset = dataset[['Usage Date','Api Key','Country Name','Network Name','Margin (EUR) pFD Adjusted','Revenue (EUR) pFD Adjusted']]
        dataset = dataset[['Country Name','Api Key','Network Name','Margin (EUR) pFD Adjusted','Revenue (EUR) pFD Adjusted']]
        dataset = dataset.groupby(['Country Name','Api Key','Network Name']).sum()[['Margin (EUR) pFD Adjusted','Revenue (EUR) pFD Adjusted']]
        dataset['Margin (EUR) pFD Adjusted in Percentage'] = dataset['Margin (EUR) pFD Adjusted'] / dataset['Revenue (EUR) pFD Adjusted']
        dataset.to_csv('../data_file/df_churn_0222_margin_'+str(percent)+'.csv')
        dataset=dataset[dataset['Margin (EUR) pFD Adjusted in Percentage'] < 0.2]
        dataset.to_csv('../data_file/df_churn_0222_margin_' + str(percent) + '_less_per_customer.csv')

    def gateway_optimizer(self,file_name):
        print('gateway_optimizer')
        dataset = pd.read_csv(file_name)
        print(dataset.shape)
        dataset = dataset[['Country Name','Network Name','Gateway Name','Api Key','Usage Date','Volume','Cost SMS (EUR)']]
        dataset = dataset.groupby(['Country Name','Network Name','Gateway Name','Api Key','Usage Date']).sum()[['Volume','Cost SMS (EUR)']]
        dataset['Unit Cost SMS (EUR)'] = dataset['Cost SMS (EUR)'] / dataset['Volume']
        dataset = dataset[dataset['Unit Cost SMS (EUR)'] != 0]
        dataset['Optimized Gateway Name']='same'
        dataset['Optimized Cost SMS (EUR)'] = 0.0
        #dataset.to_csv('../data_file/df_churn_0222_gateway_' + 'mx' + '.csv')
        dataset=dataset.reset_index()
        dataset_min_unit_cost = dataset.groupby(['Country Name', 'Network Name', 'Gateway Name'])['Unit Cost SMS (EUR)'].agg(['min'])
        dataset_min = dataset_min_unit_cost.loc[dataset_min_unit_cost.groupby(['Country Name', 'Network Name'])['min'].idxmin()]
        dataset_min.to_csv('../data_file/df_churn_0222_gateway_min' + 'mx' + '.csv')
        dataset_min = dataset_min.reset_index()
        for index,row in dataset.iterrows():
            df_recommend = pd.DataFrame()
            df_recommend = df_recommend.append({"Country Name": row['Country Name'],
                                                "Network Name": row['Network Name'],
                                                "Gateway Name": row['Gateway Name']},ignore_index=True)
            #if df_recommend.isin(dataset_min).all().all():
            check_gateway=df_recommend.isin(dataset_min)
            if (check_gateway["Gateway Name"]).bool():
                print(index, 'leave it')
            else: #print(index,'switch gateway')
                df_cost = pd.merge(df_recommend, dataset_min, on=['Country Name', 'Network Name'], how='inner')
                min_unit_cost = df_cost['min'] * row['Volume'] #series
                #min_unit_cost.values #ndarry  #.item(0) #item 0
                dataset.set_value(index, 'Optimized Cost SMS (EUR)', min_unit_cost.values.item(0))
                dataset.set_value(index, 'Optimized Gateway Name', df_cost['Gateway Name_y'].values[0])
        dataset.to_csv('../data_file/df_churn_0222_gateway_' + 'mx' + '_new.csv')


    def train_model(self):
        print(tf.__version__)
        print("------------train_model start time:", time.ctime())
        dataset = pd.read_csv(self.data_file_directory+ self.data_file_yearly)
        #print(dataset.info())
        # dataset = dataset[dataset.CLASSES != 'Paging Group']
        # print(dataset.CLASSES.value_counts())
        # classes = dataset['CLASSES'].unique()
        # print(classes)

        # #balance data to min - 3232 change 0.49 to 0.56
        # dataset = dataset.groupby('CLASSES')
        # dataset=dataset.apply(lambda x: x.sample(dataset.size().min()).reset_index(drop=True))
        # print(dataset.CLASSES.value_counts())
        # classes = dataset['CLASSES'].unique()
        # dataset = dataset[dataset.PRODUCT_NAME != 'Paging Group']

        print(dataset.PRODUCT_NAME.value_counts())
        classes = dataset['PRODUCT_NAME'].unique()
        print(classes)
        #dataset = dataset.drop(['ORDER_AMOUNT'], axis=1) #0.59 to 0.55

        # dataset = dataset.groupby('PRODUCT_NAME')
        # dataset=dataset.apply(lambda x: x.sample(dataset.size().min()).reset_index(drop=True))
        # print(dataset.PRODUCT_NAME.value_counts())
        # classes = dataset['PRODUCT_NAME'].unique()
        # print(classes)

        dataset = dataset.drop(['ACCT_ID'], axis=1)
        # dataset = dataset.drop(['PARENT_PARTNER_ACCOUNT_NAME'], axis=1)
        # dataset = dataset.drop(['PARTNER_ACCOUNT_NAME'], axis=1)
        print("numberOfFeatures: ", config.numberOfFeatures)

        dataset = dataset.apply(lambda x: x.fillna(0) if x.dtype.kind in 'iuf' else x.fillna("O"))
        labelencoderX = []
        for i in range(config.numberOfFeatures):
            labelencoderX.append(0)
        dataset = dataset.sample(frac=1)

        X = dataset.iloc[:, :config.numberOfFeatures].values
        y = dataset.iloc[:, -1].values

        totalRecords = len(y)
        print("The record counts in the model: ", totalRecords)
        #yOrg = y
        yOrg = y = np.array([self.MAP[yi] for yi in y])
        XOrg = X
        for i in range(len(config.isNumber)):
            if config.isNumber[i] == 0:
                labelencoderX[i] = LabelEncoder()
                X[:, i] = labelencoderX[i].fit_transform(X[:, i])
        # scaling
        scalerX = MinMaxScaler()
        scalerX.fit(X)
        scaler_filename = "../data_file/scalerX.save"
        joblib.dump(scalerX, scaler_filename)

        y = y.reshape(len(y), 1)
        X = scalerX.transform(X)

        self.featureSelection(X, y, config.numberOfFeatures)
        self.chi2Selection(X, y)
        self.pcaSelection(X)

        # X, y = make_classification(
        #     n_samples=1000, n_features=9, n_informative=3, n_redundant=2,
        #     n_repeated=0, n_classes=8, n_clusters_per_class=1, random_state=0
        # )
        # self.yb_featureSelection(X, y)

        # summarize components
        # y = scalerY.transform(y.reshape(len(y), 1))
        # split into train and test sets
        train_size = int(len(dataset) * 0.8)
        test_size = len(X) - train_size
        trainX, testX = X[0:train_size, :], X[train_size:len(X), :]
        trainY, testY = y[0:train_size, :], y[train_size:len(X), :]
        trainYorg = trainY

        yOrg_cat = ks.utils.np_utils.to_categorical(yOrg, num_classes=self.max_class)
        trainY_cat = ks.utils.np_utils.to_categorical(trainY, num_classes=self.max_class)
        testY_cat = ks.utils.np_utils.to_categorical(testY, num_classes=self.max_class)
        print('trainY_cat.shape', trainY_cat.shape)
        print('testY_cat.shape', testY_cat.shape)

        # smt = SMOTE()  #max
        # nr = NearMiss() #min
        # trainX, trainY = nr.fit_sample(trainX, trainY)
        # print('np.bincount(trainY)',np.bincount(trainY))

        print("XGBClassifier model ...")
        model = XGBClassifier()#learning_rate=0.1, n_estimators=29, min_child_weight=3, max_depth=13)
        model.fit(trainX, trainY)
        y_pred = model.predict(testX)
        joblib.dump(model, '../data_file/XGB.pkl', compress=1)
        print('saving the model to ../data_file/XGB.pkl')

        from sklearn.metrics import accuracy_score
        print(accuracy_score(testY, y_pred))
        # predictions = [round(value) for value in y_pred]
        # accuracy = accuracy_score(testY, predictions)
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))


        print("linear_model model ...") #0.2778
        from sklearn import linear_model
        model = linear_model.SGDClassifier()
        model.fit(trainX, trainY)
        y_pred = model.predict(testX)
        from sklearn.metrics import accuracy_score
        print(accuracy_score(testY, y_pred))

        print("RandomForestClassifier model ...") #0.3404
        from sklearn import linear_model
        model = RandomForestClassifier(n_estimators= 200, max_depth = 30 )
        model.fit(trainX, trainY)
        joblib.dump(model, '../data_file/RF.pkl', compress=1)
        print('saving the model to ../data_file/RF.pkl')

        #testing the saved model
        #model = joblib.load('../data_file/RF.pkl')
        y_pred = model.predict(testX)
        from sklearn.metrics import accuracy_score
        print(accuracy_score(testY, y_pred))

        matrix = confusion_matrix(y_pred, testY)
        print('confusion_matrix:')
        #print(matrix)
        sns.heatmap(matrix, annot=True, fmt="g", linewidths=.5, xticklabels=classes, yticklabels=classes)
        sns.set(font_scale=0.4)
        plt.xticks(rotation=90)
        plt.yticks(rotation=45)
        b, t = plt.ylim()  # discover the values for bottom and top
        b += 0.5  # Add 0.5 to the bottom
        t -= 0.5  # Subtract 0.5 from the top
        plt.ylim(b, t)  # update the ylim(bottom, top) values
        plt.show()
        self.show_scores(list(map(int, testY)), y_pred)

        # print("CNN model ...")
        # epochs = config.epochs
        # model = Sequential()
        # model.add(Dense(config.numberOfFeatures*2 + 1, input_dim=config.numberOfFeatures, activation='relu'))
        # model.add(Dense(trainY_cat.shape[1], activation='softmax'))
        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())
        # # tbCallBack = ks.callbacks.TensorBoard(log_dir='./Outbound', histogram_freq=0, batch_size=32, write_graph=True,         #                                       write_grads=False, write_images=False, embeddings_freq=0,         #                                       embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)         # history = model.fit(trainX, trainY_cat, epochs=epochs, verbose=2, callbacks=[tbCallBack])
        # history = model.fit(trainX, trainY_cat, epochs=epochs, verbose=2)
        # print(model.metrics_names)
        # scores = model.evaluate(testX, testY_cat, verbose=0)
        # print(scores)
        # model.save('../data_file/CNN.h5')
        # trainPredict = model.predict(trainX)
        # #print('trainPredict:', trainPredict[0])
        # # make predictions
        # yhat = model.predict_classes(testX)  # the most likely class only (highest probability value). Calls argmax on the output of predict.
        # #print(yhat)
        # #self.show_scores(list(map(int, testY)), yhat)
        # yhat = model.predict_proba(testX)
        # print(yhat[0])

    @staticmethod
    def predict_model_by_id(id,file):
        print('predict_model')
        #model = joblib.load('../data_file/XGB.pkl')
        model = joblib.load('../data_file/RF.pkl')

        dataset_persona = pd.read_csv(file)
        if 'PRODUCT_NAME' in dataset_persona.columns:
            dataset_persona = dataset_persona.drop(['PRODUCT_NAME'], axis=1)
            print('drop PRODUCT_NAME')
        dataset_persona = dataset_persona.apply(lambda x: x.fillna(0) if x.dtype.kind in 'iuf' else x.fillna("O"))
        if (id in dataset_persona.ACCT_ID.values):
            data=dataset_persona.loc[dataset_persona['ACCT_ID'] == id]
            data = data.drop(['ACCT_ID'], axis=1)
            X=BiLeast15.transformer(data)
            #bb = np.array([X])
            predict_value = model.predict(X)
            print('predict_value:',predict_value)
            for item in predict_value:
                print(id, BiLeast15().get_key(BiLeast15().MAP,item))
        K.clear_session()
        return

    @staticmethod
    def predict_model_by_file(file):
        print('predict_model')
        df_price = pd.read_csv('../data_file/' + BiLeast15().data_file_price)
        #model = joblib.load('../data_file/XGB.pkl')
        model = joblib.load('../data_file/RF.pkl')
        dataset_persona = pd.read_csv(file)
        if 'PRODUCT_NAME' in dataset_persona.columns:
            dataset_persona = dataset_persona.drop(['PRODUCT_NAME'], axis=1)
            print('drop PRODUCT_NAME')
        #dataset_persona['PREDICTED_PRODUCT'] = ''
        dataset_persona = dataset_persona.apply(lambda x: x.fillna(0) if x.dtype.kind in 'iuf' else x.fillna("O"))
        #df_out = pd.DataFrame(columns=['ACCT_ID', 'PREDICTED_PRODUCT'])

        data = dataset_persona.drop(['ACCT_ID'], axis=1)
        X=BiLeast15.transformer(data)
        predict_value = model.predict(X)
        name_list = []
        for item in predict_value:
            name_list.append(BiLeast15().get_key(BiLeast15().MAP,item))
        dataset_persona['PRODUCT_NAME'] = name_list #predict_value
        dataset_persona.to_csv('../data_file/df_details.csv')

        df_groupby=dataset_persona.groupby(['ACCT_ID', 'PRODUCT_NAME']).PRODUCT_NAME.agg('count').to_frame('SCORE').reset_index()
        df_groupby.to_csv('../data_file/df_groupby.csv')

        df_groupby_percentage=df_groupby.groupby(['ACCT_ID', 'PRODUCT_NAME']).agg({'SCORE':'sum'})
        df_groupby_percentage = df_groupby_percentage.groupby(level=0).apply(lambda x:100 * x / float(x.sum()))
        df_groupby_percentage.to_csv('../data_file/df_groupby_percentage.csv')
        idx=df_groupby_percentage.groupby(['ACCT_ID'])['SCORE'].transform(max) == df_groupby_percentage['SCORE']
        df_groupby_percentage[idx].to_csv('../data_file/df_groupby_max.csv')

        df_dataset = pd.read_csv('../data_file/df_groupby_max.csv')
        df_revenue = pd.merge(df_dataset, df_price, on='PRODUCT_NAME', how='inner')
        df_revenue = df_revenue[df_revenue.SCORE > BiLeast15().model_accuracy]
        df_revenue = df_revenue.rename(columns={'ACCT_ID': 'PARENT_ACCOUNT_IDENTIFIER'})
        df_revenue.to_csv('../data_file/df_groupby_max_revenue.csv')

        # idx=df_groupby.groupby(['ACCT_ID'])['SCORE'].transform(max) == df_groupby['SCORE']
        # df_groupby[idx].to_csv('../data_file/df_groupby_max.csv')
        # df_revenue = pd.merge(df_groupby[idx], df_price, on='PRODUCT_NAME', how='inner')
        # df_revenue.to_csv('../data_file/df_out_groupby_max_revenue.csv')

        df_pa = pd.read_csv('../data_file/' + BiLeast15().data_file_pa)
        df_recommended = pd.merge(df_revenue,df_pa,how='inner',on='PARENT_ACCOUNT_IDENTIFIER')
        df_recommended = df_recommended.drop(['SCORE','PRICE'], axis=1)
        df_recommended.to_csv('../data_file/df_groupby_max_revenue_recommended.csv')

        K.clear_session()
        return

    @staticmethod
    def transformer(data):
        labelencoderX = []
        for i in range(config.numberOfFeatures):
            labelencoderX.append(0)
        X = data.iloc[:, :config.numberOfFeatures].values
        for i in range(len(config.isNumber)):
            if config.isNumber[i] == 0:
                labelencoderX[i] = LabelEncoder()
                X[:, i] = labelencoderX[i].fit_transform(X[:,i])
        # scalerX = MinMaxScaler()
        # scalerX.fit(X)
        scalerX = joblib.load('../data_file/scalerX.save')
        X = scalerX.transform(X)
        return X

    def run_process(self):
        logging.debug("run_process")
        options = {0: self.train_model, 1: self.predict_model_by_id, 2: self.predict_model_by_file,
                   3: self.plot_all_dimensions, 5: self.dynamic_select,6:self.data_analysis,
                   11: self.arima,12: self.lstm,
                   21: self.margin_percent,22: self.gateway_optimizer}
        #options[5]()
        #options[21](20)
       # options[22](self.data_file_directory+self.data_file_monthly)
        options[11](self.data_file_directory+self.data_file_yearly)
        options[12](self.data_file_directory+self.data_file_yearly)

        #options[5]()
        #options[3]('ALL')
        #options[3]('CN')
        #options[6]()

        # fig = plt.figure(figsize=(12, 9))
        # self.plot_LSA()
        # plt.show()
        return True

if __name__ == '__main__':
    BiChurn().run_process()
# s3 -> s3 -> create bucket -> bucket name will be key
# import boto3
# import sagemaker
# s3 = boto3.client('s3')
# object = s3.get_object(Bucket='optimizebucket', Key=file_name)
# dataset = pd.read_csv(object['Body'])

#Error tokenizing data. C error: Expected 41 fields in line 34486179, saw 45
#data = pd.read_csv('file1.csv', error_bad_lines=False)
