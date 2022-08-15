library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)
library(xgboost)
library(caret)
data.table::setDTthreads(1)

setwd("~/Downloads/Kaggle/hm")
source("./src/8_arrow_h.R")

# Read data ---------------------------------------------------------------

articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

articles_data = articles %>% collect %>% as.data.table
setDT(articles_data)
articles_data[, grp := .GRP, by = .(detail_desc)]


# Dynamic attr
articles_dy = readRDS("data/t8/articles_dy.rds")
setDT(articles_dy)
customers_dy = readRDS("data/t8/customers_dy.rds")
setDT(customers_dy)

articles_similarity_o = readRDS("data/t8/articles_similarity.rds")
setDT(articles_similarity_o)

articles_d2v_o = readRDS("data/t8/articles_doc2vec.rds")
setDT(articles_d2v_o)

sar_all = readRDS("data/t8/sar.rds")
setDT(sar_all)

btgr = readRDS("data/t8/btgr.rds")
setDT(btgr)

fse = readRDS("data/t8/fse.rds")
setDT(fse)

prodemb = readRDS("data/t8/prodemb.rds")
setDT(prodemb)



TOPX = 20

# Train -------------------------------------------------------------------

curr_yr = 2020
curr_wk = 37

purchased_customers = transactions %>% 
  filter(yr == curr_yr, wk == curr_wk) %>% 
  select(aid, cid) %>% 
  distinct %>% 
  mutate(purchased = 1) %>% 
  collect() %>% 
  as.data.table
num_purchased_total = length(unique(purchased_customers$cid))

# Sample customers
set.seed(100)
cids_sample = c(sample(unique(purchased_customers$cid), size = floor(num_purchased_total*0.05), replace = F), 328)

negatives = executeStrategy(curr_yr = curr_yr, curr_wk = curr_wk, 
                            topx = TOPX, wks = (curr_wk - 2):(curr_wk - 1), cids = cids_sample)

# Check which candidates were not captured
if(0){
  candidates = merge(negatives %>% mutate(candidate = 1), purchased_customers, by = c("aid", "cid"), all.x = T, all.y = T) 
  View(candidates[is.na(candidate) & purchased == 1][, N := length(unique(aid)), by = .(cid)][order(-N, cid)][, grp := .GRP, by = .(cid)][order(grp)])
  
  wks = (curr_wk - 2):(curr_wk - 1)
  cids_obs = transactions %>% 
                filter(yr == curr_yr, wk %in% wks) %>% 
                pull(cid) %>% unique
  cids_obs = intersect(cids_obs, cids_sample)
  checkPR(negatives, curr_wk = curr_wk, cids = cids_sample)
  checkPR(negatives[s1_prior_purchases != 0], curr_wk = curr_wk, cids = cids_sample)
  checkPR(negatives[cid %in% cids_obs], curr_wk = curr_wk, cids = cids_obs)
  checkPR(negatives[!(cid %in% cids_obs)], curr_wk = curr_wk, cids = setdiff(cids_sample, cids_obs))
  
  candidates = constructCandidates(negatives = negatives, 
                                   purchased_customers = purchased_customers %>% filter(cid %in% cids_sample), 
                                   curr_yr = curr_yr, curr_wk = curr_wk, wks = (curr_wk - 2):(curr_wk - 1), keep_purchased = F)
  checkPR(candidates[s11_prodemb != 0, .(aid, cid)], curr_wk = curr_wk, cids = cids_sample)
  
  cid_c = 1175164
  customers_dy[cid == cid_c]
  customers %>% filter(cid == cid_c) %>% collect
  
  transactions %>% filter(cid == cid_c) %>% collect %>% 
    merge(articles_dy %>% mutate(yr = 2020), by.x = c("aid", "yr", "wk"), by.y = c("aid", "yr", "wk_predict"), all.x = T) %>% 
    merge(articles_data, by = "aid") %>% 
    merge(articles_dy %>% mutate(yr = 2020, wk_predict = wk_predict - 1), by.x = c("aid", "yr", "wk"), by.y = c("aid", "yr", "wk_predict"), all.x = T, suffixes = c("", ".new")) %>% 
    arrange(desc(t_dat)) %>% View
  
  transactions %>% 
    filter(yr == curr_yr, wk == curr_wk - 1) %>% 
    collect %>% filter(cid %in% (customers %>% collect %>% 
                                   filter(postal_code == "fd4427d79ca53dda0eef7b62352213f55367edaaa65ecef5c28a9792b39d0381") %>% 
                                   pull(cid))) %>% 
    as.data.table %>% 
    {.[, .(.N), by = (aid)][order(-N)]}
  
  transactions %>% 
    filter(yr == curr_yr, wk == curr_wk - 1, sales_channel_id == 1) %>% 
    collect %>% 
    as.data.table %>% 
    {.[, .(.N), by = (aid)][order(-N)][1:20]}
  
  transactions %>% 
    filter(yr == curr_yr, wk == curr_wk) %>% collect %>%
    filter (cid %in% (customers_dy[wk_predict == 37 & is.na(days_since_last_seen_c)] %>% pull(cid))) %>%
    merge(articles_dy %>% mutate(yr = 2020), by.x = c("aid", "yr", "wk"), by.y = c("aid", "yr", "wk_predict"), all.x = T) %>%
    sapply(function(x) round(median(x, na.rm = T), 2))
    
    
}

candidates = constructCandidates(negatives = negatives, 
                                 purchased_customers = purchased_customers %>% filter(cid %in% cids_sample), 
                                 curr_yr = curr_yr, curr_wk = curr_wk, wks = (curr_wk - 2):(curr_wk - 1), keep_purchased = F)
# 3991
rm(negatives)
gc()

# Fill missing strategy values with 0
to_fill = colnames(candidates)[grepl("s[0-9].*", colnames(candidates))]
for (col in to_fill){
  set(candidates, which(is.na(candidates[[col]])), col, 0)
}

if (0){
merge(purchased_customers  %>% filter(cid %in% cids_sample), 
negatives, by = c("cid", "aid"), all.x = T, all.y = F)[is.na(s1)][, .(.N), by = "aid"][order(-N)]
}

# Remove any cases where the customer didn't purchase anything
train = candidates[, if(sum(purchased, na.rm = T) > 0) .SD, by = .(cid)]

# Create folds
set.seed(100)
toSample = train[, .(cid = unique(cid))]
toSample[, folds := sample(1:5, .N, replace = T)]
train = merge(train, toSample, by = "cid", sort = F)

# Order the data
train = train[order(cid)]
query_groups_train = train[, .N, by = .(cid)]$N
folds = lapply(1:5, function(idx) which(train$folds == idx))

toIgnore = c("aid", "cid", "folds")
character_cols = c("Active", 
                   "index_group_no",
                   "product_group_name", 
                   "perceived_colour_value_name",
                   "garment_group_no", "index_code",
                   "modal_index_group_no",
                   "modal_index_code"
                   )
response = "purchased"
x = setdiff(colnames(train), c(toIgnore, response))

dtrain <- lgb.Dataset(train[, x, with = F] %>% as.matrix, label = train[[response]],
                      group =  query_groups_train, categorical_feature = character_cols)

params = list(
  objective = "lambdarank", 
  feature_fraction = 0.67,
  bagging_fraction = 0.67,
  eta = 0.03,
  max_depth = 20,
  metric = "map",
  eval_at = 12,
  num_leaves = 31
)

set.seed(123)
model = lgb.cv(params = params, data = dtrain, nrounds = 400, eval_freq = 20, nfold = 3, verbose = 1, early_stopping_rounds = 20)
print(model$best_iter)

model = lgb.train(params = params, data = dtrain, nrounds = model$best_iter, valids = list("valid" = dtrain), eval_freq = 20)


if (0){
  to_validate = copy(candidates)
  setDT(to_validate)
  to_validate[, grp := floor(.I/100000)*100000]
  
  grpn = uniqueN(to_validate$grp)
  pb <- txtProgressBar(min = 0, max = grpn, style = 3)
  to_validate[, pred := {setTxtProgressBar(pb, .GRP); model$predict(.SD[, x, with = F] %>% as.matrix);}, by = .(grp)]
  
  to_validate = to_validate[order(cid, -pred)]
}

if (0){
  dtrain <- xgb.DMatrix(train[, x, with = F] %>% sapply(as.numeric) %>% as.matrix, label = train[[response]])
  
  params_xgb = list(
    objective = "binary:logistic", 
    subsample = 0.67,
    colsample_bytree = 0.67,
    eta = 0.03,
    max_depth = 8,
    eval_metric = "map@12"
  )
  
  set.seed(123)
  model.xgb = xgb.cv(params = params_xgb, data = dtrain, nrounds = 400, eval_freq = 20, nfold = 3, verbose = 1, early_stopping_rounds = 20)
  print(model.xgb$best_iter)
  
  model.xgb = xgb.train(params = params_xgb, data = dtrain, nrounds = model.xgb$best_iter)
  
}



# map12 = validateModel(candidates, model, curr_yr = curr_yr, curr_wk = curr_wk, cids = cids_sample)
# 0.03320927 
# 0.0380 with 50 top art_d2v
# 0.03879099 with 12 top art_d2v and better scores for earlier scenarios
# 0.03330431  with 12 top pop and removal of some other scenarios
# 0.03034805 with 20 top pop and bought together (enabling previous scenarios)

# Validate ----------------------------------------------------------------

curr_yr = 2020
curr_wk = 38

purchased_customers = transactions %>% 
  filter(yr == curr_yr, wk == curr_wk) %>% 
  select(aid, cid) %>% 
  distinct %>% 
  mutate(purchased = 1) %>% 
  collect() %>% 
  as.data.table
num_purchased_total = length(unique(purchased_customers$cid))

# Sample customers
set.seed(100)
cids_sample = sample(unique(purchased_customers$cid), size = floor(num_purchased_total), replace = F)

#cids_sample = unique(purchased_customers$cid)

negatives = executeStrategy(curr_yr = curr_yr, curr_wk = curr_wk, 
                            topx = TOPX, wks = (curr_wk - 2):(curr_wk - 1), cids = cids_sample)
candidates = constructCandidates(negatives = negatives, 
                                 purchased_customers = purchased_customers %>% filter(cid %in% cids_sample), 
                                 curr_yr = curr_yr, curr_wk = curr_wk, wks = (curr_wk - 2):(curr_wk - 1), keep_purchased = F)
# 15025
# 17647 with art_d2v
# 21212 with top 50 art_d2v
# 17685 with top 12 art_d2v
# 17778 with price binned
# 16043 with removal of two scenarios and topx = TOPX
# 20025 with top 20 and purchased together
rm(negatives)
gc()

map12 = validateModel(candidates, model, curr_yr = curr_yr, curr_wk = curr_wk, cids = cids_sample, model.xgb = NULL)
# 0.02519999
# 0.0268 with MF
# 0.02632039 with art_d2v
# 0.02777998 with top 50 art_d2v
# 0.02760443 with top 12 art_d2v
# 0.0273 with pb
# 0.02767386 with removal of two scenarios and topx = 12
# 0.028146
# 0.02763845 with top 20 and purchased together
# 0.02811012 with samex, diff vars


if (0){
  actuals = transactions %>% filter(yr == curr_yr, wk == curr_wk, cid %in% cids_sample) %>% 
    select(aid, cid) %>%
    collect %>% setDT %>% 
    {.[, .(actual = list(aid)), by = .(cid)]}
  
  to_validate = copy(candidates)
  setDT(to_validate)
  to_validate[, grp := floor(.I/100000)*100000]
  
  grpn = uniqueN(to_validate$grp)
  pb <- txtProgressBar(min = 0, max = grpn, style = 3)
  to_validate[, pred := {setTxtProgressBar(pb, .GRP); model$predict(.SD[, x, with = F] %>% as.matrix);}, by = .(grp)]
  
  to_validate_2 = merge(to_validate, actuals[, .(aid = unlist(actual), actual = 1), by = .(cid)], by = c("cid", "aid"), all = T)
  to_validate_2 = to_validate_2[order(cid, -pred)]  
  
  to_validate_2[, any_p := any(purchased), by = .(cid)]
  to_validate_2[any_p == T]
  cid_c = 180
  View(to_validate_2[cid == cid_c])
  transactions %>% filter(yr == curr_yr, wk >=  curr_wk-2, cid  == cid_c) %>% collect %>% merge(articles_data, by = "aid") %>% arrange(desc(t_dat)) %>% View
}


# Re-train -------------------------------------------------------------------

curr_yr = 2020
curr_wk = 38

purchased_customers = transactions %>% 
  filter(yr == curr_yr, wk == curr_wk) %>% 
  select(aid, cid) %>% 
  distinct %>% 
  mutate(purchased = 1) %>% 
  collect() %>% 
  as.data.table
num_purchased_total = length(unique(purchased_customers$cid))

# Sample customers
set.seed(100)
cids_sample = sample(unique(purchased_customers$cid), size = floor(num_purchased_total), replace = F)

negatives = executeStrategy(curr_yr = curr_yr, curr_wk = curr_wk, 
                            topx = TOPX, wks = (curr_wk - 2):(curr_wk - 1), cids = cids_sample)
candidates = constructCandidates(negatives = negatives, 
                                 purchased_customers = purchased_customers %>% filter(cid %in% cids_sample), 
                                 curr_yr = curr_yr, curr_wk = curr_wk,  wks = (curr_wk - 2):(curr_wk - 1), keep_purchased = F)
# 21212
rm(negatives)
gc()

if (0){
merge(purchased_customers  %>% filter(cid %in% cids_sample), 
      negatives, by = c("cid", "aid"), all.x = T, all.y = F)[is.na(s1)][, .(.N), by = "aid"][order(-N)]
}

# Remove any cases where the customer didn't purchase anything
train = candidates[, if(sum(purchased, na.rm = T) > 0) .SD, by = .(cid)]

rm(candidates)
gc()

# Create folds
# set.seed(100)
# toSample = train[, .(cid = unique(cid))]
# toSample[, folds := sample(1:5, .N, replace = T)]
# train = merge(train, toSample, by = "cid")
# folds = lapply(1:5, function(idx) which(train$folds == idx))

# Order the data
train = train[order(cid)]

toIgnore = c("aid", "cid", "folds")
character_cols = c("Active", 
                   "index_group_no",
                   "product_group_name", 
                   "perceived_colour_value_name",
                   "garment_group_no", "index_code",
                   "modal_index_group_no", "preferred_channel", "last_channel",
                   "modal_index_code"
)
response = "purchased"
x = setdiff(colnames(train), c(toIgnore, response))

query_groups_train = train[, .N, by = .(cid)]$N
dtrain <- lgb.Dataset(train %>% select(all_of(x)) %>% as.matrix, label = train[[response]],
                      group =  query_groups_train, categorical_feature = character_cols)

rm(train)
rm(candidates)
gc()

params = list(
  objective = "rank_xendcg", 
  feature_fraction = 0.67,
  bagging_fraction = 0.67,
  eta = 0.03,
  max_depth = 20,
  metric = "map",
  eval_at = 12,
  num_leaves = 31
)

gc()
set.seed(123)
model = lgb.cv(params = params, data = dtrain, nrounds = 400, eval_freq = 20, nfold = 3, early_stopping_rounds = 20)
print(model$best_iter)

model = lgb.train(params = params, data = dtrain, nrounds = model$best_iter, 
                  valids = list("valid" = dtrain), eval_freq = 20)
if (0){
lgb.save(model, "lgbm_8_with_similarity_repurch")
}

if (0){
  dtrain <- xgb.DMatrix(train[, x, with = F] %>% sapply(as.numeric) %>% as.matrix, label = train[[response]])
  
  params_xgb = list(
    objective = "binary:logistic", 
    subsample = 0.67,
    colsample_bytree = 0.67,
    eta = 0.03,
    max_depth = 8,
    eval_metric = "map@12"
  )
  
  set.seed(123)
  model.xgb = xgb.cv(params = params_xgb, data = dtrain, nrounds = 400, eval_freq = 20, nfold = 3, verbose = 1, early_stopping_rounds = 20)
  print(model.xgb$best_iter)
  
  model.xgb = xgb.train(params = params_xgb, data = dtrain, nrounds = model.xgb$best_iter)
  
}

#map12 = validateModel(candidates, model, curr_yr = curr_yr, curr_wk = curr_wk, cids = cids_sample)
# 0.03118055

rm(dtrain)
gc()

# Submission --------------------------------------------------------------

curr_yr = 2020
curr_wk = 39

customer_list = customers %>% select(cid) %>% collect %>% as.data.table
customer_list = customer_list[order(cid)]
customer_list[, grp := floor(.I/50000)*50000]
grpn = uniqueN(customer_list$grp)

purchased_customers = data.table(aid = integer(0), cid = integer(0), purchased = numeric(0))

pb <- txtProgressBar(min = 0, max = grpn, style = 3)

customer_list[, pred := {
  setTxtProgressBar(pb, .GRP); 
  negatives = executeStrategy(curr_yr = curr_yr, curr_wk = curr_wk, topx = TOPX, 
                              wks = (curr_wk - 2):(curr_wk - 1), cids = cid)
  candidates = constructCandidates(negatives = negatives, 
                                   purchased_customers = purchased_customers, 
                                   curr_yr = curr_yr, curr_wk = curr_wk,
                                   wks = (curr_wk - 2):(curr_wk - 1), keep_purchased = F)
  preds = model$predict(candidates[, x, with = F] %>% as.matrix);
  candidates[, pred := preds]
  candidates = candidates[order(cid, -pred)]
  candidates = candidates[candidates[, .I[1:12], by = .(cid)]$V1]
  list(candidates[, .(list(aid)), by = .(cid)]$V1)
  }, by = .(grp)]

# Format
to_submit = customer_list[, .(aid = unlist(pred)), by = .(cid)]
to_submit[, idx := 1:.N, by = .(cid)]

to_submit = merge(to_submit, article_mapping, by = "aid", sort = F) %>%
  merge(customer_mapping, by = "cid", sort = F)
to_submit = to_submit[order(cid, idx)]

to_submit = to_submit[, .(customer_id, article_id)][, .(prediction = paste(head(article_id, 12), collapse = " ")), 
                                                      by = customer_id]
data.table::fwrite(to_submit, file = "submission_lgbm_8_with_similarity_repurch.csv.gz")
