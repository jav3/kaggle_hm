library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)


# Helpers -----------------------------------------------------------------
vc <- function(dt, id){
  View(dt[cid == id] %>% merge(articles, by = "aid") %>% {.[order(t_dat)]})
}

va <- function(dt, id){
  View(dt[aid == id])
}

# Read data ---------------------------------------------------------------

setwd("~/Downloads/Kaggle/hm")

articles = read_arrow("data/articles_m.arrow")
customers = read_arrow("data/customers_m.arrow")
transactions = read_arrow("data/transactions_m.arrow")


# Article popularity ------------------------------------------------------
setDT(transactions)
setDT(articles)
setDT(customers)

# Filter on date range ----------------------------------------------------
trans = transactions[t_dat > max(t_dat) - weeks(5)]

# Partition ---------------------------------------------------------------
trans[, wk := week(t_dat)]
trans[, train := (wk < max(wk))*1]
wks = unique(trans$wk)
wks_train = unique(trans[train == 1]$wk)
wks_test = unique(trans[train == 0]$wk)


# Purchase indicator ------------------------------------------------------
trans[, purchased := 1]

# Features ----------------------------------------------------------------

# Article popularity
d_num_purchases_aw = articles[, .(aid, wk = list(wks))]
d_num_purchases_aw = d_num_purchases_aw[, .(wk = unlist(wk)), by = .(aid)]
d_num_purchases_aw[, num_purchases_aw := 0]
d_num_purchases_aw = rbindlist(list(d_num_purchases_aw, 
                                    trans[, .(num_purchases_aw = length(unique(cid))), by = .(aid, wk)]), 
                               use.names = T, fill = T)
d_num_purchases_aw = d_num_purchases_aw[, .(num_purchases_aw = sum(num_purchases_aw)), by = .(aid, wk)]
d_num_purchases_aw[, popularity := num_purchases_aw/sum(num_purchases_aw), by = .(wk)]
d_num_purchases_aw[, popularity_num := frank(-num_purchases_aw), by = .(wk)]
d_num_purchases_aw = d_num_purchases_aw[order(aid, wk)]
d_num_purchases_aw[, last_w_popularity := lag(popularity), by = .(aid)]
d_num_purchases_aw[is.na(last_w_popularity) & wk > min(d_num_purchases_aw$wk), last_w_popularity := 0]
d_num_purchases_aw[, last2_w_popularity := lag(popularity, n = 2), by = .(aid)]
d_num_purchases_aw[is.na(last2_w_popularity) & wk > min(d_num_purchases_aw$wk) + 1, last2_w_popularity := 0]


# First seen date of article ----------------------------------------------
first_seen = merge(articles[, .(aid)], trans[, .(first_seen_a = min(t_dat)), by = .(aid)], by = "aid", all.x = T)

# Negative sampling -------------------------------------------------------

# Keep top X popular items
topx = 100
neg_samples = merge(d_num_purchases_aw[!is.na(popularity_num) & popularity_num < topx, 
                                       .(aid, wk, popularity_num)], 
                    trans[train == T, .(cid, aid, wk)], all.x = F, all.y = F, by = c("aid", "wk"))

# Add an "A" to identify articles
neg_samples = unique(neg_samples[, aid := paste0("A", aid)])

# Create graph
gph = graph_from_data_frame(neg_samples[, .(aid, cid)])

# Specify type = TRUE if user, else article (bipartite graph)
V(gph)$type = V(gph)$name %in% neg_samples$cid

# Sample some customers
set.seed(123)
num_customers = 50000
neg_samples_cust = unique(neg_samples[, .(cid)])[sample(.N, num_customers)]

# Sample negatives from the graph using neighbours
sampleArticles <- function(x, steps = 5){
  samples = random_walk(gph, as.character(x), steps = steps, mode = "all") 
  res = names(samples) %>% 
    {.[substr(., 1, 1) == "A"]} %>% # Keep only articles
    substring(2) %>% # Remove the "A"
    unique
  return (res)
}
set.seed(123)
neg_samples_cust2 = pblapply(neg_samples_cust$cid, sampleArticles)

# Re-format as data frame
neg_samples_cust3 = data.table(cid = neg_samples_cust$cid, negs = neg_samples_cust2) %>% 
  .[, .(aid = as.numeric(unlist(negs))), by = .(cid)] %>%
  unique

# Add topx popular articles by week for each customer in train
set.seed(100)
top_pop_a = unique(trans[train == 1, .(cid, wk)])
top_pop_a[, aid := sample(d_num_purchases_aw[popularity_num <= topx]$aid, .N, replace = T), by = .(wk)]

neg_samples_cust4 = rbindlist(list(neg_samples_cust3, top_pop_a), use.names = T, fill = T)
neg_samples_cust4 = unique(neg_samples_cust4)

# Add weeks for missing cases
set.seed(123)
neg_samples_cust4[is.na(wk), wk := sample(wks_train, .N, replace = T)]

# Remove any cases that have already been purchased
neg_samples_cust5 = merge(neg_samples_cust4, 
                              trans[, .(cid, aid, purchased)], by = c("cid", "aid"), all.x = T)
neg_samples_cust5 = neg_samples_cust5[is.na(purchased)]
neg_samples_cust5[, purchased := 0]
neg_samples_cust5[, train := 1]

# Prepare data ------------------------------------------------------------

all = trans[, .(cid, aid, wk, purchased, train, t_dat)] %>% 
  unique %>%
  {rbindlist(list(., neg_samples_cust5), use.names = T, fill = T)}

# Set t_dat for negative samples to first day of week
all[is.na(t_dat), t_dat := as.Date(paste(2022, wk, 1, sep="-"), "%Y-%U-%u")]

all = merge(all, 
            d_num_purchases_aw[, .(aid, wk, last_w_popularity, last2_w_popularity)], 
            by = c("aid", "wk"), all.x = T)

# First seen (remove any cases where transaction date <= first_seen)
all = merge(all, first_seen, by = "aid", all.x = T)
all = all[t_dat > first_seen_a]
all[, days_on_market := as.numeric(t_dat - first_seen_a)]

# Format data for modelling -----------------------------------------------
response = "purchased"
x = c("last_w_popularity", "last2_w_popularity")#, "days_on_market")

# Order the data
all = all[order(train, cid, t_dat)]

# Create query groups for ranking
query_groups = all[train == 1, .N, by = .(cid)]$N

# Convert character cols
character_cols = c()
all_m = all[, c(x, "train", response), with = F]

dtrain <- lgb.Dataset(all_m[train == 1, x, with = F] %>% as.matrix, 
                      label = all[train == 1][[response]], categorical_feature = character_cols, group =  query_groups)

# Train model -------------------------------------------------------------

params = list(
  objective = "rank_xendcg", 
  feature_fraction = 0.67,
  bagging_fraction = 0.67,
  eta = 0.1,
  max_depth = 8,
  metric = "map",
  eval_at = 1:12
)

set.seed(123)
mod.cv <- lgb.cv(params = params, data = dtrain, nfold = 5, early_stopping_rounds = 50)
mod.cv$best_iter
# 73

query_groups_train = all[train == 1 & wk == max(wks_train) - 1, .N, by = .(cid)]$N
dtrain <- lgb.Dataset(all[train == 1 & wk == max(wks_train) - 1, x, with = F] %>% as.matrix, 
                      label = all[train == 1  & wk == max(wks_train) - 1][[response]], categorical_feature = character_cols, 
                      group =  query_groups_train)

query_groups_valid = all[train == 1 & wk == max(wks_train), .N, by = .(cid)]$N
dvalid <- lgb.Dataset(all[train == 1 & wk == max(wks_train), x, with = F] %>% as.matrix, 
                      label = all[train == 1  & wk == max(wks_train)][[response]], categorical_feature = character_cols, 
                      group =  query_groups_valid)

set.seed(123)
model = lgb.train(params = params, data = dtrain, nrounds = 100, valids = list("valid" = dvalid), eval_freq = 10)

lgb.importance(model)


# Predictions -------------------------------------------------------------

# Predict on only customers in the testing dataset -- no impact on metrics
test = all[train == 0, .(cid = unique(cid))]

articles_to_consider = first_seen[first_seen_a < max(as.Date(paste(2022, wks_train, 7, sep="-"), "%Y-%U-%u")) & first_seen_a >= max(all[train == 0]$t_dat) - 30]

test = d_num_purchases_aw[(wk %in% wks_test) & aid %in% articles_to_consider$aid]
test = merge(test, first_seen, by = "aid", all.x = T)
test[, days_on_market := as.numeric(max(trans[train == 0]$t_dat) - first_seen_a)]

dtest <- test[, x, with = F] %>% as.matrix

preds = model$predict(dtest)

sum(trans[train == 0]$aid %in% test[which(order(-preds) %in% 1:50)]$aid)

