# Here we are basically tracking the moving average.
# moving average basically smooths outliers.
# We could easily do the same thing with the raw data

# first check the std dev for the region pair and max and min values
# then check linear or nonlinear regression to see how current value compares

import logging as pylog
import os
import re
import json
import datetime
import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
#import pytz
from tabulate import tabulate

# from flask import escape
from google.cloud import bigquery
from google.cloud import logging
from google.cloud.logging.resource import Resource

from sklearn.model_selection import train_test_split 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn import metrics

os.environ['TEST'] = 'iperf'
os.environ['THRESHHOLD'] = '4'
os.environ['METRIC'] = 'Throughput'
os.environ['ABOVEBELOW'] = 'above'
os.environ['MACHINETYPE'] = 'n1-standard-16'
os.environ['HISTRANGEDAYS'] = '10'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service_key.json'
# GCP_PROJECT = os.environ.get('GCP_PROJECT')


long_term_throughput():
  # logClient = logging.Client()
  bqClient = bigquery.Client('smu-benchmarking')

  # setup bq 
  dataset = 'reporting'
  table = 'historic_throughput'
  # dst_table = 'pkb_alerts'
  # dst_dataset = 'reporting'

  hist_range_days = int(os.environ['HISTRANGEDAYS'])


  query = (
              f"SELECT * FROM `smu-benchmarking.reporting.historic_throughput` "
              f"WHERE receiving_region != sending_region "
              f"  AND hist_range_days={hist_range_days} "
              f"  AND entry_date > '2020-02-01' "
              f"ORDER BY entry_date "
          )

  # Query Config
  # job_config = bigquery.QueryJobConfig()
  # job_config.query_parameters = query_params
  # table_ref = bqClient.dataset('reporting').table('pkb_alerts')
  # job_config.destination = table_ref
  # job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND


  query_job = bqClient.query(
      query,
      # Location must match that of the dataset(s) referenced in the query.
      location="US"
      # job_config=job_config,
  )  # API request - starts the query

  # Print the results
  #for row in query_job:
  #    print("{}: \t{}".format(row.word, row.word_count))
  print("QUERY RESULTS")
  results = query_job.result().to_dataframe()
  assert query_job.state == "DONE"
  print(results)
  results['entry_date'] = pd.to_datetime(results['entry_date'])
  results = results.sort_values('entry_date')
  results = results.set_index('entry_date')
  gb = results.groupby(['sending_zone','receiving_zone', 'ip_type', 'sending_thread_count'])

  groups = list(gb.groups)
  print(groups)


  dataframe_to_insert_columns = ['entry_date',
                                 'sending_zone', 'receiving_zone', 
                                 'sending_region', 'receiving_region',
                                 'test', 'metric',
                                 'max', 'min', 'mean', 'median', 'stddev', 'unit',
                                 'most_recent_moving_average','lin_reg_coef', 'ip_type',
                                 'sending_thread_count', 'alert_string']

  df_to_insert = pd.DataFrame(columns = dataframe_to_insert_columns)


  # FOR EACH GROUP
  for group in groups:

      end_date = datetime.datetime.combine(datetime.date.today(), datetime.datetime.min.time())

      print(f"GROUP: {group}")
      print(gb.get_group(group))
      # print(gb.get_group(group).index)

      # print(f'SHAPE: {gb.get_group(group).shape}')

      # print(end_date)

      for timespan_days in [7,30,90,180]:
          start_date = end_date - datetime.timedelta(days=timespan_days)

          if gb.get_group(group).index.min() >= end_date:
              end_date = end_date + datetime.timedelta(days=1)
              continue

          temp_df = gb.get_group(group).loc[start_date:end_date]

          oldest_date = temp_df.index.min()
          if oldest_date > start_date:
              end_date = end_date + datetime.timedelta(days=1)
              continue
          # print(temp_df.index)
          # print(type(oldest_date))
          # print(type(start_date))
          if len(temp_df) == 0:
            end_date = end_date + datetime.timedelta(days=1)
            continue

          Y = temp_df.loc[:,['hist_avg']].values.reshape(-1, 1)
          X = temp_df.reset_index().index.values.reshape(-1, 1)


          y_max = np.max(Y)
          y_mean = np.mean(Y)
          y_median = np.median(Y)
          y_min = np.min(Y)
          y_std = np.std(Y)
          y_most_recent = Y[-1][0]
          y_initial = Y[0][0]

          linear_regressor = LinearRegression()
                
          linear_regressor.fit(X, Y)

          Y_pred = linear_regressor.predict(X)

          lin_reg_coef = linear_regressor.coef_[0][0]

          change_from_start = y_most_recent - Y[0][0]
          change_from_mean = y_most_recent - y_mean
          unit = temp_df['unit'][0]

          # if (abs(change_from_start) > 2*y_std) or (abs(change_from_mean) > 2*y_std):

          new_row = {'entry_date': end_date,
                     'start_date': start_date,
                     'timespan_days': timespan_days,
                     'sending_zone': temp_df['sending_zone'][0],
                     'receiving_zone': temp_df['receiving_zone'][0], 
                     'sending_region': temp_df['sending_region'][0],
                     'receiving_region': temp_df['receiving_zone'][0],
                     'test': 'iperf',
                     'metric': 'Throughput',
                     'max': y_max,
                     'min': y_min,
                     'mean': y_mean, 
                     'median': y_median,
                     'stddev': y_std,
                     'initial_moving_average': y_initial,
                     'most_recent_moving_average': y_most_recent,
                     'unit': unit,
                     'lin_reg_coef': lin_reg_coef,
                     'ip_type': temp_df['ip_type'][0],
                     'sending_thread_count': temp_df['sending_thread_count'][0],
                     'alert_string': f'Throughput moving average changed by {change_from_start:.3f} {unit} in the past {timespan_days} days'
                    }

          df_to_insert = df_to_insert.append(new_row, ignore_index=True)

          # print(df_to_insert)
          # print(df_to_insert['alert_string'][0])



          end_date = end_date + datetime.timedelta(days=1)


          # svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
          # svr_rbf.fit(X,Y)
          # Y_svr = svr_rbf.predict(X)

          # plt.scatter(X, Y)
          # plt.plot(X, Y_pred, color='red')
          # plt.plot(X, Y_svr, color='green')
          # plt.show()


          # # plt.plot(gb.get_group(groups[0])['hist_avg'])
          # # plt.show()


          # # plt.hist(gb.get_group(groups[0])['hist_avg'])
          # # plt.show()

          # print(gb.get_group(groups[0])['hist_avg'].std())


  insert_table_id = 'smu-benchmarking.reporting.long_term_trend_throughput'

  table = bqClient.get_table(insert_table_id)

  errors = [[]]

  errors = bqClient.insert_rows_from_dataframe(table, df_to_insert)
  # print(tabulate(df_to_insert[0:200], headers='keys', tablefmt='psql'))
  if errors[0] == []:
      print(f"{df_to_insert.shape[0]} new rows have been added.")
  else:
      print("Encountered errors while inserting rows: {}".format(errors))

long_term_throughput()