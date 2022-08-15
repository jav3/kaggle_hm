library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)
library(caret)
library(h2o)
data.table::setDTthreads(1)

h2o.init()

setwd("~/Downloads/Kaggle/hm")
source("./src/8_arrow_h.R")

curr_yr = 2020
curr_wk = 37

# Read data ---------------------------------------------------------------

articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

articles_data = articles %>% collect %>% as.data.table

# Dynamic attr
articles_dy = readRDS("data/t8/articles_dy.rds")
setDT(articles_dy)
customers_dy = readRDS("data/t8/customers_dy.rds")
setDT(customers_dy)

curr_purchases = transactions %>% 
  filter(yr == curr_yr, wk == curr_wk) %>%
  select(cid, aid, t_dat) %>%
  collect()
setDT(curr_purchases)

# Strat 1 -----------------------------------------------------------------

# Prior purchases - s1
week_end_date = as.Date("2020-01-01") %m+% days((curr_wk-1)*7 - 1)
wks = (curr_wk - 2):(curr_wk - 1)

prior_purchases = transactions %>% 
  filter(yr == curr_yr, wk < curr_wk & wk %in% wks) %>%
  select(cid, aid, t_dat) %>%
  collect %>%
  as.data.table
prior_purchases = prior_purchases[order(cid, aid, desc(t_dat))] %>%
  unique(by = c("cid", "aid")) %>%
  # Label for strategy
  mutate(strat = "s1_prior_purchases", score = as.numeric(week_end_date - t_dat))
checkPR(prior_purchases, curr_wk = curr_wk)

dat = prior_purchases %>% 
  merge(articles_data[, .(aid, index_group_name)], by = "aid") %>%
  merge(curr_purchases[, .(cid, aid, purchased = 1)] %>% unique, by = c("cid", "aid"), all.x = T)
dat[is.na(purchased), purchased := 0]
dat[, purchased := as.factor(purchased)]

train = as.h2o(dat, "train.hex")
x = c("index_group_name", "score")
y = "purchased"

mod = h2o.randomForest(x, y, training_frame = train, nfolds = 3, stopping_metric = "lift_top_group", 
                       ntrees = 100, col_sample_rate_per_tree = 0.67, sample_rate = 0.67, seed = 123, model_id = "mod.hex")
h2o.varimp(mod)
dat$pred = h2o.predict(mod, train) %>% as.data.frame %>% pull(p1)
mod_2 = rpart::rpart(pred ~ index_group_name + score, data = dat)
rpart.plot::rpart.plot(mod_2)
dat[pred > 0.0043, summary(score)]
checkPR(prior_purchases, curr_wk = curr_wk)
# MAX MAP12: 0.0278
# Full Covered Customer: 0.0168
# Not Covered Customer: 0.9522
# Candidate multiplier: 2.1488
# Unique Article Id: 23890
# Precision:  108.9726 
# Recall:  0.01971831 
# Fscore:  0.01252451 
# [1] "_"

# Decision: past week as primary, with past two weeks as second choice

# Strat 2 -----------------------------------------------------------------

# Top popular - s2
topx = 5
top_pop_last_w = articles_dy[wk_predict == curr_wk & popularity_num <= topx, aid]
top_pop_last_w = curr_purchases %>% pull(cid) %>% unique %>%
  CJ(top_pop_last_w) %>%
  setNames(c("cid", "aid")) %>%
  # Label for strategy
  mutate(strat = "s2_top_popular")
top_pop_last_w = merge(top_pop_last_w, articles_dy[wk_predict == curr_wk, .(aid, score = price)], 
                       by = "aid", all.x = T)
top_pop_last_w[is.na(score), score := max(articles_dy[wk_predict == curr_wk]$price, na.rm = T)]
checkPR(top_pop_last_w, curr_wk = curr_wk)

dat = top_pop_last_w %>% 
  merge(articles_data[, .(aid, index_group_name, index_group_no)], by = "aid") %>%
  merge(customers_dy[wk_predict == curr_wk, .(cid, modal_index_group_no = modal_index_group_no)], by = "cid") %>%
  merge(curr_purchases[, .(cid, aid, purchased = 1)] %>% unique, by = c("cid", "aid"), all.x = T)
dat[is.na(purchased), purchased := 0]
dat[, purchased := as.factor(purchased)]
dat[, same_group := modal_index_group_no == index_group_no]
dat[, modal_index_group_no := as.factor(modal_index_group_no)]

train = as.h2o(dat, "train.hex")
x = c("modal_index_group_no", "same_group", "score")
y = "purchased"

mod = h2o.randomForest(x, y, training_frame = train, nfolds = 3, stopping_metric = "lift_top_group", stopping_rounds = 10,
                       ntrees = 10, col_sample_rate_per_tree = 0.67, sample_rate = 0.67, seed = 123, model_id = "mod.hex")
h2o.varimp(mod)
dat$pred = h2o.predict(mod, train) %>% as.data.frame %>% pull(p1)

mod_2 = rpart::rpart(pred ~ same_group + as.factor(modal_index_group_no) + score, data = dat)
rpart.plot::rpart.plot(mod_2)

# Decision: Recommend top pop from same modal group items 
checkPR(dat[same_group == 1 | is.na(modal_index_group_no)][, .(cid, aid)], curr_wk = curr_wk)
# MAX MAP12: 0.0094
# Full Covered Customer: 0.0037
# Not Covered Customer: 0.9747
# Candidate multiplier: 1.3726
# Unique Article Id: 5
# Precision:  167.1967 
# Recall:  0.008209381 
# Fscore:  0.00692021 
# [1] "_"

# Try recommending top pop per same modal group
topx = 5
top_pop_group = articles_dy[wk_predict == curr_wk, .(aid, price, popularity_num)] %>% 
  merge(articles_data[, .(aid, index_group_no)], by = "aid", all.x = F) %>%
  arrange(index_group_no, popularity_num) %>%
  {.[, head(.SD, topx), by = .(modal_index_group_no = index_group_no)]}

top_pop_last_w = curr_purchases %>% select(cid) %>%
  as.data.table %>%
  merge(customers_dy[wk_predict == curr_wk, .(cid, modal_index_group_no)], by = "cid") %>%
  {.[, modal_index_group_no := ifelse(is.na(modal_index_group_no), 1, modal_index_group_no)]} %>%
  merge(top_pop_group, by = c("modal_index_group_no"), allow.cartesian = T)
checkPR(top_pop_last_w, curr_wk = curr_wk)
# MAX MAP12: 0.0117
# Full Covered Customer: 0.0053
# Not Covered Customer: 0.9713
# Candidate multiplier: 1.58
# Unique Article Id: 25
# Precision:  168.5838 
# Recall:  0.009372121 
# Fscore:  0.007265244 
# [1] "_"

# Map top pop to prior purchase categories
topx = 5
wks = (curr_wk - 1)
top_pop_group = articles_dy[wk_predict == curr_wk, .(aid, price, popularity_num)] %>% 
  merge(articles_data[, .(aid, index_group_no)], by = "aid", all.x = F) %>%
  arrange(index_group_no, popularity_num) %>%
  {.[, head(.SD, topx), by = .(index_group_no)]}

top_pop_last_w = transactions %>% 
  filter(yr == curr_yr, wk < curr_wk & wk %in% wks, cid %in% unique(curr_purchases$cid)) %>%
  select(cid, aid, t_dat) %>%
  collect %>%
  as.data.table %>%
  merge(articles_data[, .(aid, index_group_no)], by = "aid", all.x = F) %>%
  select(cid, index_group_no) %>%
  unique %>%
  merge(top_pop_group, by = c("index_group_no"), allow.cartesian = T)
top_pop_last_w = top_pop_last_w[order(cid, popularity_num)][, head(.SD, topx), by = .(cid)]
checkPR(top_pop_last_w, curr_wk = curr_wk)
# MAX MAP12: 0.0029
# Full Covered Customer: 0.0013
# Not Covered Customer: 0.9935
# Candidate multiplier: 0.3062
# Unique Article Id: 25
# Precision:  142.7198 
# Recall:  0.002145584 
# Fscore:  0.003285186 
# [1] "_"

# Decision: use top pop by modal index group 
