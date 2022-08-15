library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(pbapply)
library(caret)
data.table::setDTthreads(4)

setwd("~/Downloads/Kaggle/hm")
source("src/8_arrow_h.R")

# Read data ---------------------------------------------------------------

articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

customers_data = customers %>% collect
setDT(customers_data)

articles_data = articles %>% collect
setDT(articles_data)

articles_dy = readRDS("data/t8/articles_dy.rds")

curr_yr = 2020
wks = 36:37
curr_wk = 38

# Prior purchases ---------------------------------------------------------

prior_purchases = transactions %>% 
  filter(yr == curr_yr, wk < curr_wk & wk %in% wks) %>%
  select(cid, aid) %>%
  distinct(cid, aid) %>% 
  collect() %>%
  mutate(candidate = 1)
setDT(prior_purchases)

res = checkPR(prior_purchases, curr_wk = curr_wk)
# MAX MAP12: 0.0222
# Full Covered Customer: 0.0147
# Not Covered Customer: 0.9644
# Candidate multiplier: 1.0664
# Unique Article Id: 18611
# Precision:  69.95396 
# Recall:  0.01524367

# Most popular ------------------------------------------------------------

topx = 20

top_pop_last_w = articles_dy %>%
  filter(wk_predict == curr_wk, popularity_num <= topx) %>%
  pull(aid)
top_pop_last_w = customers %>%
  select(cid) %>%
  pull %>%
  CJ(top_pop_last_w) %>%
  setNames(c("cid", "aid"))
res = checkPR(top_pop_last_w, curr_wk = curr_wk)
# At 10:
# MAX MAP12: 0.0247
# Full Covered Customer: 0.0111
# Not Covered Customer: 0.9419
# Candidate multiplier: 64.1928
# Unique Article Id: 10
# Precision:  3227.429 
# Recall:  0.01988977
# At 20:
# MAX MAP12: 0.0397
# Full Covered Customer: 0.0163
# Not Covered Customer: 0.903
# Candidate multiplier: 128.3856
# Unique Article Id: 20
# Precision:  3664.966 
# Recall:  0.03503051

# CBF
cbf = readRDS("data/t8/cbf38.rds")
cbf = cbf[, .(aid = unlist(preds)[1:10], score = unlist(scores)[1:10]), by = .(cid)]

res = checkPR(cbf, curr_wk = curr_wk)

# At 50:
# MAX MAP12: 0.0249
# Full Covered Customer: 0.0156
# Not Covered Customer: 0.9577
# Candidate multiplier: 29.9099
# Unique Article Id: 21906
# Precision:  1802.253 
# Recall:  0.01659586
# At 10:
# MAX MAP12: 0.0124
# Full Covered Customer: 0.0081
# Not Covered Customer: 0.9793
# Candidate multiplier: 5.9926
# Unique Article Id: 17896
# Precision:  814.2352 
# Recall:  0.007359822

# Similarity to recently purchased & purchased before
prior_purchases = transactions %>% 
  filter(yr == curr_yr, wk < curr_wk & wk %in% wks) %>%
  select(cid, aid) %>%
  distinct(cid, aid) %>%
  collect() %>%
  mutate(candidate = 1)
setDT(prior_purchases)

articles_similarity = readRDS("data/t8/articles_similarity.rds")
setDT(articles_similarity)
articles_similarity = articles_similarity[linked_aid %in% unique(prior_purchases$aid)]
similar_purchase = merge(prior_purchases, articles_similarity[score >= 0.9], by = "aid", all = F, allow.cartesian = T)
similar_purchase[, aid := NULL]
setnames(similar_purchase, "linked_aid", "aid")
res = checkPR(similar_purchase, curr_wk = curr_wk)
# At 0.9:
# MAX MAP12: 0.0144
# Full Covered Customer: 0.0073
# Not Covered Customer: 0.972
# Candidate multiplier: 6.8244
# Unique Article Id: 15072
# Precision:  567.7559 
# Recall:  0.01201995

# Special offers seen in the past weeks
spec_offers = articles_dy %>%
  filter(wk_predict == curr_wk, popularity > 0, popularity > popularity_1d_prior) %>%
  filter(aid %in% articles_data[department_name == "Campaigns" & 
                                  garment_group_name == "Special Offers"]$aid) %>%
  pull(aid)
spec_offers = customers %>%
  select(cid) %>%
  pull %>%
  CJ(spec_offers) %>%
  setNames(c("cid", "aid"))
res = checkPR(spec_offers, curr_wk = curr_wk)
# MAX MAP12: 0.0116
# Full Covered Customer: 0.0068
# Not Covered Customer: 0.9781
# Candidate multiplier: 66.2182
# Unique Article Id: 11
# Precision:  8531.249 
# Recall:  0.007761836

# Combination of some of the above ----------------------------------------

all = rbindlist(list(top_pop_last_w, cbf, similar_purchase, prior_purchases), use.names = T, fill = T) %>% 
  unique(by = c("cid", "aid"))
res = checkPR(all, curr_wk = curr_wk)

# MAX MAP12: 0.0776
# Full Covered Customer: 0.0413
# Not Covered Customer: 0.8461
# Candidate multiplier: 141.8616
# Unique Article Id: 23518
# Precision:  2252.083 
# Recall:  0.06299128


trans_curr = transactions %>% filter(yr == curr_yr, wk == curr_wk) %>% select(cid, aid, t_dat) %>% collect %>% as.data.table

merge(trans_curr[, .(cid, aid, purchased = 1)], all[, .(cid, aid, candidate = 1)], 
      by = c("cid", "aid"), all.x = T, all.y = F) %>%
  {.[, .(num_purchased = .N, 
         sum_candidates = sum(candidate, na.rm = T),
         diff = .N - sum(candidate, na.rm =)), by = .(aid)][order(-diff)]}


merge(trans_curr[cid == 331, .(cid, aid, t_dat)], all[cid == 331, .(cid, aid, rec = 1)], by = c("cid", "aid"), all.x = T, all.y = F)




