library(SAR)
library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Matrix)
library(Metrics)
library(Hmisc)

setwd("~/Downloads/Kaggle/hm")

Rcpp::sourceCpp("src/8_cpp_h.cpp")
source("src/8_arrow_h.R")

# Read data ---------------------------------------------------------------
articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

# Download entire article and customer sets
articles = articles %>% collect
setDT(articles)

customers = customers %>% collect
setDT(customers)

curr_yr = 2020
# wks_train = 2 # Train for 2 weeks prior to wks_test
wks_test = c(35, 36, 37, 38, 39) # Test on these weeks - the last one is for Kaggle


usif = fread("data/t8/fsa_usif.csv")
usif = as.matrix(usif)

usif_norm = sqrt(rowSums(usif*usif))
usif = sweep(usif, 1, usif_norm, "/")

trans_sub = transactions %>% 
  filter(yr == curr_yr, wk %in% wks_test) %>%
  select(cid, aid, t_dat) %>%
  collect()
setDT(trans_sub)
trans_sub = unique(trans_sub)

preds = Rcpp_crossprod(usif[unique(trans_sub$aid),], t(usif[unique(trans_sub$aid),]), topx = 30)

cbf = data.table(aid = unique(trans_sub$aid),
                 preds = asplit(preds[,1:30], 1),
                 scores = asplit(preds[,31:60], 1)
)

cbf = cbf[, .(preds = unlist(preds), scores = unlist(scores)), by = .(aid)]
cbf[, linked_aid := unique(trans_sub$aid)[preds]]

cbf[, preds := NULL]
cbf = cbf[aid != linked_aid]

saveRDS(cbf, "data/t8/fse.rds")


if (0){
  
usif = t(usif)
  
# Results
res = vector("list", length(wks_test))

i = 1

week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
wks = (wks_test[i] - wks_train):(wks_test[i] - 1)

# Construct model -----------------------------------------------------
trans_sub = transactions %>% 
  filter(yr == curr_yr, wk %in% wks) %>%
  select(cid, aid, t_dat) %>%
  collect()
setDT(trans_sub)
trans_sub = unique(trans_sub)

mod = sar(user = trans_sub$cid, 
          item = factor(trans_sub$aid),
          time = trans_sub$t_dat, similarity = "count", support_threshold = 5)

affinity_mat = SAR:::make_affinity(user = trans_sub$cid, 
                                   item = factor(trans_sub$aid),
                                   time = trans_sub$t_dat, #t0 = as.Date("2020-09-22"), 
                                   wt = NULL, half_life = 30)

# Item-item sim-mat (keeping only items seen in the last two weeks)
sim_mat = tcrossprod(t(usif[,unique(trans_sub$aid)]))

sim_mat = sim_mat + 1

preds = Rcpp_crossprod(sim_mat, sim_mat)

# SAR
num_recs = 20
preds = Rcpp_crossprod(t(affinity_mat), sim_mat, topx = num_recs)



user_emb = usif %*% affinity_mat
user_emb = t(user_emb)
ue_norm = sqrt(rowSums(user_emb*user_emb))
user_emb = sweep(user_emb, 1, ue_norm, "/")

num_recs = 100
preds = Rcpp_crossprod(as.matrix(user_emb), usif)

cbf = data.table(cid = as.numeric(rownames(user_emb)), 
                 preds = asplit(preds[,1:num_recs], 1),
                 scores = asplit(preds[,(num_recs + 1):(2*num_recs)], 1)
)

cbf = cbf[, .(item = unlist(preds), score = unlist(scores)), by = .(cid)]

checkPR(cbf[order(cid, -score)][, head(.SD, 3), by = .(cid)][, .(cid, aid = item)], curr_yr = curr_yr, curr_wk = wks_test[i])

}