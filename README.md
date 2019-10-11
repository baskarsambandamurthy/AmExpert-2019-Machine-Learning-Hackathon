# AmExpert 2019 Machine Learning Hackathon
## Score
- Private LB Rank:  13th
- Private LB Score:  0.9273
- Public LB Rank: 18th
- Public LB Score:  0.9361
## Important points for top score achievement

 1. Understanding table relationships and performing various merging of the tables
 2. Merging customer transaction data inside cross validation instead of before cross validation.
 3. Performing Time Series Cross Validation Technique to prevent target leak.
 4. Treating categorical columns with huge values as text and generating  tfidf (term frequency inverse document frequency) features while merging.
 5. Test Predictions using full train data set with the iterations information retrieved from time series cross validation

## Data Preprocessing

### Label Encoding
Label encoding is performed on below categorical data.
 - _brand_type_ and _category_ columns in `item_data` table 
  - _marital_status_, _age_range_, _family_size_,   _no_of_children_ in `customer_demographics` table 
 - _campaign_type_ in `campaign_data` table

### Date Feature formatting
   The following columns that contain date are converted to pandas date time format for further date comparison and filtering.
- _date_ column in `customer_transaction_data` table 
- _start_date_ and _end_date_ columns in `campaign_data` table


## Merging Data
This competition contains data from multiple tables and hence proper merging of data need to be performed depending upon the type of relationship.

#### Understanding Table relationships
There are different types of table relationships possible:

| Relationship      |Description  |
|----------------------|-------------------------------|
|`one-to-one`        |Both tables can have only one record on either side of the relationship. Each primary key value relates to only one (or no) record in the related table 
|`one-to-many`        |The parent table (or primary key table) contains only one record that relates to none, one, or many records in the child table 
|`many-to-many`        |Each record in both tables can relate to any number of records (or no records) in the other table.

The following depicts the type of table relationships in this competition.

**one-to-many relationship tables**

| Tables|  Key
|----------------------|-------------------------------|
|`campaign_data` and `train` (or `test`)        |campaign_id
|`customer_demographics` and `train` (or `test`)          |customer_id
|`item_data`  and `customer_transaction_data`         |item_id
|`customer_demographics` and `customer_transaction_data`       |customer_id

**many-to-many relationship tables**

| Tables|  Key
|----------------------|-------------------------------|
|`train` (or `test`)   and `customer_transaction_data`          | customer_id
|`coupon_item_mapping` and `item_data`        | item_id
|`coupon_item_mapping`  and `customer_transaction_data`         |item_id

**Multi level relationship**
There are also following relationships which goes upto 2 levels.

 - `train` (or `test`)   ->`coupon_item_mapping` on coupon_id and `coupon_item_mapping` ->`customer_transaction_data` on item_id
- `train` (or `test`)   ->`coupon_item_mapping` on coupon_id and `coupon_item_mapping` ->`item_data` on item_id

Hence from the above multi-level relationship, it can be seen that there is an indirect `many-to-many` relationship between `train` (or `test`)   and `item_data` tables.

**Merging techniques**
Separate merging process will be applied for one-to-many and many-to-one relationships. 
- For `one-to-many`, simple merge of both tables will provide combined features and 
- for `many-to-many`, aggregation of columns such as mean, min, max etc need to be performed on the table that will be joined.


**Merging Process**
- Simple merge of `train` (or `test`)   table with `campaign_data` and `customer_demographics` are performed. 
- Then aggregates of `item_data` is generated from `coupon_item_mapping` parent table using coupon_id  key. All `item_data` columns are categorical, the aggregates performed for the categorical columns are *mode* and *nunique*.
- 
> **Note**: `customer_transaction_data` has been merged only during the cross validation. The reason is that this table contains the redeemed discount amount and other features which are directly related with the target variable redemption_status. If the merge is performed before cross validation, there would be *target leak* from this table.

Code: 

```python
def merge_data(data):
    
    data_unmerged = data.copy()
    
    #merge data to campaign Data many to 1 on campaign_id key (left join)
    campaign_data_merge = pd.merge(data,campaign_data,on='campaign_id',how='left')
    #coupon to item_data (many to 1) on item_id key (left join) - call coupon item 
    coupon_to_item = pd.merge(coupon_item_mapping,item_data,on='item_id',how='left')
    
    mode_fn = lambda x: pd.Series.mode(x)[0]

    aggs= ['nunique',mode_fn]

    coupon_to_item_agg = coupon_to_item.groupby(['coupon_id']).agg({'item_id':'count',
                                                               'brand':aggs,
                                                               'brand_type':aggs,
                                                               'category':aggs}).reset_index()
    
    coupon_to_item_agg.columns = ['coupon_id','coupon_size','brand_nunique','brand_mode',
                                 'brand_type_nunique','brand_type_mode',
                                 'category_nunique','category_mode']
    
    #data to coupon item on coupon_id key (left join)
    data = pd.merge(campaign_data_merge,coupon_to_item_agg,on='coupon_id',how='left')
    #data to customer demographics on customer_id key (left join)
    data = pd.merge(data,customer_demographics,on='customer_id',how='left')
        
    return data    
```

  ```python
train  =  merge_data(train)  
print('Train Merge complete')  
test  =  merge_data(test)  
print('Test Merge complete')
```

## Cross Validation
There are 2 possible approaches of cross validation for this dataset.
- To use `StratifiedKFold` cross validation as there is an high class imbalance in the target column "redemption_status". (i.e) only less than 1% of data has target value of 1.

> **Stratification**: It  is a technique where we rearrange the data in a way that each fold has a good representation of the whole dataset. It forces each fold to have at least m instances of each class. This approach ensures that one class of data is not overrepresented especially when the target variable is unbalanced.

- To perform `time series` based validation
> **Time series cross-validation**:  In this validation,  validation data is determined based on time period and not random. So specific recent periods would be used as validation set and we consider the previous period’s data as the training set.

Without time series based validation, if StratifiedKFold CV was used, then CV score used to shoot upto  0.98 or 0.99 AUC while test score in public leaderboard remains in the range 0.91 and this is because of target leak in CV since it uses the future transactions and hence due to this overfitting, there was not much of improvement in test score in public leaderboard.
So, it has been decided to use the **time series based cross validation** .

### Time Series Cross Validation
- `Walk Forward` (alias Forward chaining) time series CV technique is used to perform validation splits. 
- Each validation set comprises of 2 campaign periods and remaining previous periods would belong to train set.
- There are 5 splits utilized in the code for time series cross validation
- Appropriate campaign ids are selected for the validation and train data for each fold

Code to specify valid and train set

```python
import datetime
#time series model dates
valid_campaign_ids =[]
train_campaign_ids =[]

valid_campaign_ids +=       [[11,13]]
valid_campaign_ids +=       [[10,12]]
valid_campaign_ids +=   [[9,8]]
valid_campaign_ids += [[6,7]]
valid_campaign_ids += [[4,5]]

train_campaign_ids +=   [[26,27,28,29,30,1,2,3,4,5,6,7,8,9,10,12]]
train_campaign_ids +=   [[26,27,28,29,30,1,2,3,4,5,6,7,8,9]]
train_campaign_ids +=   [[26,27,28,29,30,1,2,3,4,5,6,7]]
train_campaign_ids +=   [[26,27,28,29,30,1,2,3,4,5]]
train_campaign_ids +=   [[26,27,28,29,30,1,2,3]]
```
### Merging during cross validation
As mentioned earlier, merging with `customer_transaction_data` table is performed *only during cross validation* since the transaction data contains the target related information such as coupon_discount etc. 

#### How transaction data is merged with train / test ?
- For train and validation set of each fold, only the `customer_transaction_data` records whose `date` is less than the minimum campaign  `start_date` in the validation set, are utilized for merging and aggregation.  (i.e) only transactions prior to the validation set is considered for model training.
- For test set, all records from `customer_transaction_data` is utilized for merging since there were no transaction records once test set campaign period begins

Code that sets the filter date
```python
val_min_start_date = val['start_date'].min()
```
Code that performs the mentioned filtering before merge
```python
 mask = customer_transaction_data['date'] < filter_date
 cust_trans_cur = customer_transaction_data[mask]
 ```

#### What data is merged ?

 - Grouped by **customer_id** in `customer_transaction_data`, aggregates such as min, max, median, mean, standard deviation are generated on columns *quantity, coupon_discount, other_discount and selling_price* and some of the date features and merged with `train` / `test` on customer_id key
 
 Code:
 ```python
 def merge_trans(data,filter_date):
    aggs=['mean','sum','min','max','median','std']
    mode_fn = lambda x: pd.Series.mode(x)[0]
    
    if filter_date is not None:
        mask = customer_transaction_data['date'] < filter_date
        cust_trans_cur = customer_transaction_data[mask]
    else:
        cust_trans_cur = customer_transaction_data
   

    cust_tran_to_item_agg = cust_trans_cur.groupby(['customer_id']).agg({'item_id':['count','nunique',mode_fn],
                                                           'date_isweekend':'mean',
                                                           'date_month':['mean',mode_fn],
                                                           'date_week':['mean',mode_fn],
                                                           'date_dayofweek':['mean',mode_fn],
                                                           'quantity':aggs,
                                                           'other_discount':aggs,
                                                           'coupon_discount':aggs,
                                                           'selling_price':aggs
                                                              }).reset_index()

    cust_tran_to_item_agg.columns = ['customer_id','trans_size','item_id_nunique','item_id_mode',
                             'date_isweekend_mean','date_month_mean','date_month_mode',
                              'date_week_mean','date_week_mode','date_dayofweek_mean','date_dayofweek_mode',
                              'quantity_mean','quantity_sum','quantity_min','quantity_max','quantity_median','quantity_std',
                              'other_discount_mean','other_discount_sum','other_discount_min','other_discount_max','other_discount_median','other_discount_std',
                              'coupon_discount_mean','coupon_discount_sum','coupon_discount_min','coupon_discount_max','coupon_discount_median','coupon_discount_std',
                              'selling_price_mean','selling_price_sum','selling_price_min','selling_price_max','selling_price_median','selling_price_std'
                             ]
    #data to coupon item on coupon_id key (left join)
    data = pd.merge(data,cust_tran_to_item_agg,on='customer_id',how='left')
    return data
 ```
 
 - Grouped by **customer_id and coupon_id combination**, data from `customer_transaction_data` is merged with `train` / `test` and aggregates such as min, max, median, mean, standard deviation are generated on columns *quantity, coupon_discount, other_discount and selling_price* and some of the date features. Also there is additional filtering performed for this merged data by filtering only the records whose transaction `date` is less than `start_date` of each record merged from `train`
	 -  First, `coupon_item_mapping` and `item_data` are merged on "item_id" and now the merged data contains both coupon_id and item_id
	 - Then `train` / test data is merged with the above merged data so that train / test data contains *one record per item_id and customer_id* combination. The data also contains corresponding coupon_id info.
	 - Then the above merged data is further merged with the `customer_transaction_data` on *customer_id and item_id*
	 - Then the merged data is grouped `by customer_id and coupon_id` and aggregates are generated on columns *quantity, coupon_discount, other_discount and selling_price*. 
	 - These aggregates are in-turn merged into `train` / test on customer_id and coupon_id

Code: 
```python
def merge_customer_coupon(coupon_to_item,data_prior_merge,
                         filter_date):
    data_merged = pd.merge(data_prior_merge,coupon_to_item,on='coupon_id',how='left')
    data_merged = pd.merge(data_merged,customer_transaction_data,on=['customer_id','item_id'],how='inner')
 
    aggs=['mean','sum','min','max','median','std']
    mode_fn = lambda x: pd.Series.mode(x)[0]
    
    groupbycols = ['customer_id','coupon_id']
    
    # to filter out records for current validation set
    if filter_date is not None:
        mask = data_merged['date'] < filter_date
        print('before valid date filter:',data_merged.shape)
        data_merged = data_merged[mask]
        print('after valid date filter:',data_merged.shape)
    else:
        data_merged = data_merged
    
    #to filter records that do not belong to current campaign
#     print(data_merged.shape)
    data_merged = data_merged[data_merged['date'] < data_merged['start_date']]
#     print(data_merged.shape)
    
#     cust_coupon_trans_agg = cust_coupon_trans_agg.filter(lambda x: x['date'] < x['start_date']) 
    cust_coupon_trans_agg = data_merged.groupby(groupbycols) .agg({'item_id':['count','nunique',mode_fn],
                                                               'date_isweekend':'mean',
                                                               'date_month':['mean',mode_fn],
                                                               'date_week':['mean',mode_fn],
                                                               'date_dayofweek':['mean',mode_fn],
                                                               'quantity':aggs,
                                                               'other_discount':aggs,
                                                               'coupon_discount':aggs,
                                                               'selling_price':aggs
                                                                  }) \
                                                    .reset_index()

    cols = ['trans_size','item_id_nunique','item_id_mode',
                             'date_isweekend_mean','date_month_mean','date_month_mode',
                              'date_week_mean','date_week_mode','date_dayofweek_mean','date_dayofweek_mode',
                              'quantity_mean','quantity_sum','quantity_min','quantity_max','quantity_median','quantity_std',
                              'other_discount_mean','other_discount_sum','other_discount_min',
                                'other_discount_max','other_discount_median','other_discount_std',
                              'coupon_discount_mean','coupon_discount_sum','coupon_discount_min',
                                'coupon_discount_max','coupon_discount_median','coupon_discount_std',
                              'selling_price_mean','selling_price_sum','selling_price_min',
                                'selling_price_max','selling_price_median','selling_price_std'
                             ]
    cols_renamed = ['cust_coupon_' + col for col in cols]
#     print(cols_renamed)
#     print(cust_coupon_trans_agg.columns)
    cust_coupon_trans_agg.columns = groupbycols + cols_renamed
    #data to coupon item on coupon_id key (left join)
    data_merged = pd.merge(data_prior_merge,cust_coupon_trans_agg,on=groupbycols,how='left')

    return data_merged
```
##### TFIDF Features
 - Similar to previous merge (i.e.) Grouped by **customer_id and coupon_id combination** and only the columns to generate aggregates are categorical columns: *item_id, brand_type, brand and category*.
 
	 - The mentioned categorical columns need to be utilized to generate beyond simple aggregations inorder to extract maximum information from merging. Hence these categorical columns are treated as **text information** where document represents the various entries of each column within each customer_id and coupon_id combination and all transactions represent the complete possible texts. For e.g. item_id, there can be maximum of 74000+ text possible and list of all item_ids in each customer_id and coupon_id combination row represents a document.
	 - Hence **tfidf** (term frequency inverse document frequency) vector features are generated on these categorical data.  The number of features generated depends upon the number of possible values that the corresponding categorical column can have.
	 - For fields *brand_type and category*, there are only less than 20 values and hence `full tfidf features` are generated on these columns.
	 - For fields *item_id and brand*, there are 74000+ and 5000+ corresponding possible values and generating such huge number of features would be impossible under limited memory constraints and computational time constraints of model training. Hence for these fields, only `top 10 high tfidf features` are determined for each row and hence a total of 20 features would be generated for these fields together.
	 
Code to generate full tfidf features

```python
def gen_tfidf_fullfeats(raw_cols_to_gen,data_texts):
    tf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))
    for col in raw_cols_to_gen:
        tfidf_feats = tf.fit_transform(data_texts[col+'_texts'])
        tfidf_feats = pd.DataFrame(tfidf_feats .todense(), columns = ['tfidf_dense_'+col+"_"+x for x in tf.get_feature_names()])
        data_texts = pd.concat([data_texts, X_tfidf], axis=1)
    return data_texts
```
Code to generate top tfidf features
```python
def get_top_tf_idf_words(response, feature_names,top_n=10):
    sorted_data = np.sort(response.data)[:-(top_n+1):-1]
    sorted_nzs = np.argsort(response.data)[:-(top_n+1):-1]
    return feature_names[response.indices[sorted_nzs]],sorted_data
def get_top_fulldata(responses,feature_names,top_n=10):
    topdata =[]
    topfeatsdata =[]
    topvaluesdata =[]
    for response in tqdm(responses):
        topfeats_topn_size =  np.full(top_n, np.nan) 
        topvalues_topn_size =  np.full(top_n, np.nan) 
        topfeats,topvalues = get_top_tf_idf_words(response,feature_names,top_n)
        topfeats_topn_size[:len(topfeats)] = topfeats[:]
        topvalues_topn_size[:len(topfeats)] = topvalues[:]
        topfeatsdata += [topfeats_topn_size]
        topvaluesdata += [topvalues_topn_size]
        
    topfeatsdata = np.array(topfeatsdata)
    topvaluesdata = np.array(topvaluesdata)
    return topfeatsdata,topvaluesdata
def gen_tfidf_topfeats(top_n,raw_cols_to_gen,data_texts):
    tf = TfidfVectorizer(tokenizer=lambda x: x.split(' '))

    for col in raw_cols_to_gen:
        tfidf_feats = tf.fit_transform(data_texts[col+'_texts'])
        feature_names = np.array(tf.get_feature_names())
        topfeatsdata,topvaluesdata = get_top_fulldata(tfidf_feats ,feature_names,top_n)

        #generate dataframe columns
        for i in range(topfeatsdata.shape[1]):
            namecol = 'tfidf_'+ col + '_name_top_' + str(i+1)
            valuecol = 'tfidf_'+ col + '_value_top_' + str(i+1)
            data_texts[namecol] = topfeatsdata[:,i]
            data_texts[valuecol] = topvaluesdata[:,i]
```

#### Model parameter tuning

 - 	Since the generated features are close to around 300, feature_fraction have been tuned to 0.4 (we are intimating the model that approximately 75 features have to be used in each tree ) and it also forces feature selection at tree level which would reduce overfitting.
 
Code that sets feature_fraction of the model:
```python
param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.4, 
         "bagging_freq": 1,
         "bagging_fraction": 0.75,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         'n_estimators' : 10000,
         "random_state": 4590}
```

#### Cross Validation Model Execution
Using LightGBM, model training is performed on 5 splits time series cross validation as mentioned earlier and uses the generated merge features (also called as encodings in the code) corresponding to each fold.
```python
import datetime

def runtimeseries(tr_encs,val_encs,test_encs,params,n_splits,fold_feats,
                  fit_function,predict_function):
    n_splits = 5

    model_iterations =[]
    fold_importance_df = []
    
    oof = np.zeros(train.shape[0])
    predictions = np.zeros(test.shape[0])
    start = time.time()
    valid_scores =[]
    models =[]

        
    for fold_ in range(n_splits):
        
        print('******************* ')
        print('fold: ',fold_)
        print('valid_campaign_ids: ',valid_campaign_ids[fold_])
        print('******************* ')
        
        cur_features  = fold_feats[fold_].copy()
        
        tr =  tr_encs[fold_]; val = val_encs[fold_]; test_cur = test_encs[fold_]
        y_tr  = tr[targetcol]; y_val = val[targetcol]
        tr = tr[cur_features]; val = val[cur_features]; test_cur = test_cur[cur_features]
        
        print(y_tr.shape)
        print(tr.shape)
        print(val.shape)
        print(y_val.shape)
        
        print(y_tr.unique())
        print(y_val.unique())
        
        clf = fit_function(tr,val,y_tr,y_val,param)
        print('Fit complete')
        models += [clf]
        
        val_preds = predict_function(clf,val)
        
        val_iterations = params['n_estimators']
        if hasattr(clf, 'best_iteration'):
               val_iterations = clf.best_iteration

        model_iterations+=[val_iterations]

        val_score = log_loss(y_val, val_preds)
        print('Cur Val Log loss Score:',val_score)
        val_score = roc_auc_score(y_val, val_preds)
        print('Cur Val AUC Score:',val_score)
        valid_scores+=[val_score]
        
        predictions += predict_function(clf,test_cur) / n_splits
        print('Test Pred complete')
        

        if hasattr(clf, 'feature_importance'):
            feature_importance_df = pd.DataFrame()
            feature_importance_df["Feature"] = cur_features
            feature_importance_df["importance"] = clf.feature_importance(importance_type='gain')    
            fold_importance_df += [feature_importance_df]

    print('valid scores:',valid_scores)
    print("CV AUC score: ",roc_auc_score(target, oof))
    
    return models,model_iterations,predictions,oof,fold_importance_df
```
#### Test Predictions
##### Why different approach for test predictions ?
- Test Predictions are not generated from cross validation. The reason is that the model was not utilizing the complete set of customer_transaction_data. The cross validation was trained using merged data filtered upto start of validation and also each coupon and customer combination features in train is filtered for corresponding prior transactions (ie. before start of campaign date of each combination). 
- So, it is decided to utilize the full training set without validation set for test predictions and also get the merged data for training including all transactions (i.e no filter). 
##### How cross validation results are going to be used in full train model ?
- But, it does not mean that the earlier cross validation is not useful. The cross validation is very useful in determining the number of estimators for executing full train model.
- There are below possible iteration numbers are used from the 5 splits of time series CV
	- Average number of iterations 
	- Maximum number of iterations
	- Minimum number of iterations
- 3 different models using full train set is generated using above iterations as estimators for light gbm model and then all these 3 models are blended (ie ensembled) to generate final test predictions. 
- With this test predictions, the test score in public LB have improved 2 points from 0.91 to 0.93
