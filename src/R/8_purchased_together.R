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

curr_yr = 2020
wks_train = 10 # Train for 2 weeks prior to wks_test
wks_test = c(35, 36, 37, 38, 39) # Test on these weeks - the last one is for Kaggle

res = vector("list", length(wks_test))

for (i in seq_len(length(wks_test))){

  week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
  wks = (wks_test[i] - wks_train):(wks_test[i] - 1)
  half_life = 5 # 5 days assumption
  
  purchases = transactions %>%
    filter(yr == curr_yr, wk %in% wks) %>% 
    collect %>%
    select(aid, cid, t_dat) %>%
    unique()
  setDT(purchases)
  
  bought_together = purchases[purchases
                              , on = .(cid = cid, aid < aid)
                              , .(aidX = x.aid, tdatX = x.t_dat, aidY = i.aid, tdatY = i.t_dat)
                              , nomatch = 0L
                              , allow.cartesian = TRUE
  ]
  bought_together[, wt := 1/2^(abs(as.numeric(tdatX - tdatY))/half_life)]
  bought_together_agg = bought_together[, .(wt = sum(wt)), by = .(aidX, aidY)]
  
  bought_together_agg = bought_together_agg[order(aidX, -wt)]
  bought_together_agg = rbindlist(list(bought_together_agg, 
                                       bought_together_agg[, .(aidX = aidY, aidY = aidX, wt)])
  )
  res[[i]] = bought_together_agg[order(aidX, -wt)][, head(.SD, 20), by = .(aidX)][, wk_predict := wks_test[i]]
  cat("Completed ", i, "\n")
}

btgr = rbindlist(res)

saveRDS(btgr, "data/t8/btgr.rds")


