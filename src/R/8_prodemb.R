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

trans_sub = transactions %>% 
  filter(yr == curr_yr, wk %in% wks_test) %>%
  select(cid, aid, t_dat) %>%
  collect()
setDT(trans_sub)
trans_sub = unique(trans_sub)


usif = fread("data/t8/prodemb_img_128.csv")
usif[, article_id := gsub("\\.jpg", "", sub('.*\\/', '', image_id))]
usif = merge(usif, article_mapping, by = "article_id")
usif = usif[aid %in% unique(trans_sub$aid)]
usif_aid = usif$aid
usif = as.matrix(usif %>% select(-aid, -article_id, -image_id))

usif_norm = sqrt(rowSums(usif*usif))
usif = sweep(usif, 1, usif_norm, "/")

preds = Rcpp_crossprod(usif, t(usif), topx = 30)

cbf = data.table(aid = usif_aid,
                 preds = asplit(preds[,1:30], 1),
                 scores = asplit(preds[,31:60], 1)
)

cbf = cbf[, .(preds = unlist(preds), scores = unlist(scores)), by = .(aid)]
cbf[, linked_aid := usif_aid[preds]]

cbf[, preds := NULL]
cbf = cbf[aid != linked_aid]

saveRDS(cbf, "data/t8/prodemb.rds")
