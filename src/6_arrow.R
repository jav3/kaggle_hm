library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)
library(caret)
data.table::setDTthreads(4)

setwd("~/Downloads/Kaggle/hm")
source("./src/6_arrow_h.R")

# Read data ---------------------------------------------------------------

articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

curr_yr = 2020
wks = 34:39
wks_train = 34:37
wks_test = setdiff(wks, wks_train)

# Article attr - dynamic --------------------------------------------------

# Create base data frame of articles by week
article_list = articles %>%
  distinct(aid) %>%
  collect
article_list = article_list %>% 
  dplyr::slice(rep(1:n(), times = length(wks))) %>%
  mutate(wk = rep(wks, each = nrow(article_list)), num_purchases_aw = 0)

# Dynamic article attributes: popularity
articles_dy = transactions %>%
  filter(yr == curr_yr, wk >= min(wks)) %>%
  group_by(wk) %>%
  map_batches(function(batch){
    batch %>% 
      group_by(wk, aid) %>%
      summarise(num_purchases_aw = n_distinct(cid))
  }) %>%
  ungroup %>%
  bind_rows(article_list) %>%
  group_by(aid, wk) %>%
  summarise(num_purchases_aw = sum(num_purchases_aw)) %>%
  group_by(wk) %>%
  mutate(popularity = num_purchases_aw/sum(num_purchases_aw), 
         popularity_num = rank(-popularity)) %>%
  arrange(aid, wk) %>%
  group_by(aid) %>%
  mutate(last_w_popularity = lag(popularity),
         last2_w_popularity = lag(popularity, 2))
setDT(articles_dy)
articles_dy[, yr := curr_yr]
articles_dy[, wk_start := as.Date(paste(yr, wk, 1, sep = "-"), "%Y-%U-%u")]

# First seen date
first_seen = transactions %>%
  group_by(yr, wk) %>%
  select(yr, wk, aid, t_dat) %>%
  map_batches(function(batch){
    batch %>%
      group_by(yr, wk, aid) %>%
      summarise(first_seen = min(t_dat))
  }) %>%
  group_by(aid) %>%
  summarise(first_seen = min(first_seen), yr = year(first_seen), wk = week(first_seen))
setDT(first_seen)
first_seen[, wk_start := as.Date(paste(yr, pmin(wk, 52), 1, sep = "-"), "%Y-%U-%u")]

# Last seen date
last_seen = transactions %>%
  group_by(yr, wk) %>%
  select(yr, wk, aid, t_dat) %>%
  map_batches(function(batch){
    batch %>%
      group_by(yr, wk, aid) %>%
      summarise(last_seen = max(t_dat))
  })
setDT(last_seen)
last_seen = last_seen[, .(last_seen = max(last_seen)), by = .(aid, wk)]
last_seen[, yr := year(last_seen)]
last_seen[, wk_start := as.Date(paste(yr, pmin(wk, 52), 1, sep = "-"), "%Y-%U-%u")]

# Merge onto main dataset
articles_dy = first_seen[, .(aid, first_seen, wk_start)][articles_dy, on = .(aid, wk_start), roll = T]
articles_dy = last_seen[, .(aid, last_seen, wk_start)][articles_dy, on = .(aid, wk_start), roll = T]
articles_dy[, days_on_market := as.numeric(wk_start - first_seen)]
articles_dy[, days_since_last_seen := as.numeric(wk_start - last_seen)]

rm(first_seen)
rm(last_seen)
rm(article_list)
gc()


# Article pool ------------------------------------------------------------

# Only keep as candidates articles seen in the last 30 days from the end of training
article_pool = articles_dy %>% 
  filter(last_seen >= (transactions %>% pull(t_dat) %>% max) - days(30) & wk %in% wks_train) %>% 
  pull(aid) %>% 
  unique

# Check recall from this article pool
transactions %>% 
  filter(yr == curr_yr, wk %in% wks_test) %>%
  pull(aid) %>%
  unique %>% 
  {mean(. %in% article_pool)}
# 91%


# Customer attr - dynamic -------------------------------------------------

# Create base data frame of customers by week
customer_list = customers %>%
  distinct(cid) %>%
  collect
customer_list = customer_list %>% 
  dplyr::slice(rep(1:n(), times = length(wks))) %>%
  mutate(wk = rep(wks, each = nrow(customer_list)))

# Dynamic article attributes: popularity
customers_dy = transactions %>%
  filter(yr == curr_yr, wk >= min(wks)) %>%
  collect %>%
  merge(articles %>% select(aid, index_code, index_group_no) %>% collect, by = "aid") %>%
  bind_rows(customer_list)
setDT(customers_dy)
  
customers_dy = customers_dy[, .(num_purchases_cw_ch1 = sum(sales_channel_id == 1, na.rm = T),
                                num_purchases_cw_ch2 = sum(sales_channel_id == 2, na.rm = T),
                                modal_index_code = Mode(index_code),
                                modal_index_group_no = Mode(index_group_no),
                                preferred_channel = Mode(sales_channel_id)
                                ), by = .(wk, cid)]
customers_dy[, yr := curr_yr]
customers_dy[, wk_start := as.Date(paste(yr, wk, 1, sep = "-"), "%Y-%U-%u")]

# Last seen date
last_seen_c = transactions %>%
  group_by(yr, wk) %>%
  select(yr, wk, cid, t_dat) %>%
  map_batches(function(batch){
    batch %>%
      group_by(yr, wk, cid) %>%
      summarise(last_seen_c = max(t_dat))
  })
setDT(last_seen_c)
last_seen_c = last_seen_c[, .(last_seen_c = max(last_seen_c)), by = .(cid, wk)]
last_seen_c[, yr := year(last_seen_c)]
last_seen_c[, wk_start := as.Date(paste(yr, pmin(wk, 52), 1, sep = "-"), "%Y-%U-%u")]

# Merge onto main dataset
customers_dy = last_seen_c[, .(cid, last_seen_c, wk_start)][customers_dy, on = .(cid, wk_start), roll = T]
customers_dy[, days_since_last_seen_c := as.numeric(wk_start - last_seen_c)]
customers_dy[, modal_index_code := as.numeric(modal_index_code)]

# Fill missing
customers_dy = customers_dy[order(cid, wk)]
customers_dy[, modal_index_code := nafill(modal_index_code, type = "locf"), by = .(cid)]
customers_dy[, modal_index_group_no := nafill(modal_index_group_no, type = "locf"), by = .(cid)]
customers_dy[, preferred_channel := nafill(preferred_channel, type = "locf"), by = .(cid)]

rm(last_seen_c)
rm(customer_list)
gc()

# Train -------------------------------------------------------------------
candidates = constructCandidates(negatives = executeStrategy(curr_yr = 2020, curr_wk = 37, topx = 20, wks = wks_train), 
                                 curr_yr = 2020, curr_wk = 37)

# Remove any cases where the customer didn't purchase anything
train = candidates[, if(sum(purchased, na.rm = T) > 0) .SD, by = .(cid)]

# Create folds
set.seed(100)
toSample = train[, .(cid = unique(cid))]
toSample[, folds := sample(1:5, .N, replace = T)]
train = merge(train, toSample, by = "cid")
folds = lapply(1:5, function(idx) which(train$folds == idx))

# Order the data
train = train[order(cid)]

toIgnore = c("aid", "cid", "folds")
character_cols = c("Active", "index_group_no", "index_code", "modal_index_group_no", "modal_index_code")
response = "purchased"
x = setdiff(colnames(train), c(toIgnore, response))

query_groups_train = train[, .N, by = .(cid)]$N
dtrain <- lgb.Dataset(train[, x, with = F] %>% as.matrix, label = train[[response]],
                      group =  query_groups_train, categorical_feature = character_cols)

params = list(
  objective = "rank_xendcg", 
  feature_fraction = 0.67,
  bagging_fraction = 0.67,
  eta = 0.1,
  max_depth = 8,
  metric = "map",
  eval_at = 12,
  num_leaves = 8
)

set.seed(123)
model = lgb.cv(params = params, data = dtrain, nrounds = 500, eval_freq = 20, nfold = 5)

model = lgb.train(params = params, data = dtrain, nrounds = model$best_iter, valids = list("valid" = dtrain), eval_freq = 20)

map12 = validateModel(candidates, model, curr_yr = 2020, curr_wk = 37)
# 0.031


# Test --------------------------------------------------------------------
candidates = constructCandidates(negatives = executeStrategy(curr_yr = 2020, curr_wk = 38, topx = 20, wks = wks_train), 
                                 curr_yr = 2020, curr_wk = 38)
validateModel(candidates, model, curr_yr = 2020, curr_wk = 38)
# 0.0279


# Submission --------------------------------------------------------------

# Re-train on latest
candidates = constructCandidates(negatives = executeStrategy(curr_yr = 2020, curr_wk = 38, topx = 20, wks = wks_train), 
                                 curr_yr = 2020, curr_wk = 38)

# Remove any cases where the customer didn't purchase anything
train = candidates[, if(sum(purchased, na.rm = T) > 0) .SD, by = .(cid)]

# Create folds
set.seed(100)
toSample = train[, .(cid = unique(cid))]
toSample[, folds := sample(1:5, .N, replace = T)]
train = merge(train, toSample, by = "cid")
folds = lapply(1:5, function(idx) which(train$folds == idx))

# Order the data
train = train[order(cid)]

query_groups_train = train[, .N, by = .(cid)]$N
dtrain <- lgb.Dataset(train[, x, with = F] %>% as.matrix, label = train[[response]],
                      group =  query_groups_train, categorical_feature = character_cols)

set.seed(123)
model = lgb.cv(params = params, data = dtrain, nrounds = 500, eval_freq = 20, nfold = 5)

model = lgb.train(params = params, data = dtrain, nrounds = model$best_iter, valids = list("valid" = dtrain), eval_freq = 20)

map12 = validateModel(candidates, model, curr_yr = 2020, curr_wk = 38)
# 0.0266

# Submission
to_submit = constructCandidates(negatives = executeStrategy(curr_yr = 2020, curr_wk = 39, topx = 20, wks = 34:38), 
                                 curr_yr = 2020, curr_wk = 39)
to_submit[, grp := floor(.I/1000000)*1000000]

cat("Computing predictions\n")
grpn = uniqueN(to_submit$grp)
pb <- txtProgressBar(min = 0, max = grpn, style = 3)
to_submit[, pred := {setTxtProgressBar(pb, .GRP); model$predict(.SD[, x, with = F] %>% as.matrix);}, by = .(grp)]
cat("\nFinished computing predictions\n")

to_submit = to_submit[order(cid, -pred)]
to_submit = to_submit[to_submit[, .I[1:12], by = .(cid)]$V1] #, .SDcols = c("aid", "cid", "pred")]

# Format
to_submit2 = merge(to_submit, article_mapping, by = "aid", sort = F) %>%
  merge(customer_mapping, by = "cid", sort = F)
to_submit2 = to_submit2[order(cid, -pred)]

to_submit3 = to_submit2[, .(customer_id, article_id)][, .(prediction = paste0(head(article_id, 12), sep = " ", collapse = " ")), 
                                             by = customer_id]
data.table::fwrite(to_submit3, file = "submission_lgbm.csv.gz")

