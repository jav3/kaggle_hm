library(SAR)
library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Matrix)
library(Metrics)
library(Hmisc)

setwd("~/Downloads/Kaggle/hm")

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

# Save results ------------------------------------------------------------

curr_yr = 2020
wks_train = 2 # Train for 2 weeks prior to wks_test
wks_test = c(35, 36, 37, 38, 39) # Test on these weeks - the last one is for Kaggle

# Results
res = vector("list", length(wks_test))

for (i in seq_len(length(wks_test))){
  
  week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
  wks = (wks_test[i] - wks_train):(wks_test[i] - 1)
  
  # Construct SAR model -----------------------------------------------------
  trans_sub = transactions %>% 
    filter(yr == curr_yr, wk %in% wks) %>%
    select(cid, aid, t_dat) %>%
    collect()
  setDT(trans_sub)
  trans_sub = unique(trans_sub)
  
  
  # Construct SAR model -----------------------------------------------------
  
  mod = sar(user = trans_sub$cid, 
            item = factor(trans_sub$aid),
            time = trans_sub$t_dat, similarity = "count", support_threshold = 5)
  preds = user_predict(mod, userdata = unique(trans_sub$cid), 
                       k = 20, backfill = F, include_seed_items = F, reftime = week_end_date)
  preds = melt.data.table(as.data.table(preds), id.vars = "user")
  preds[, ranking := as.numeric(gsub("[^0-9.-]", "", variable))]
  preds[, variable := gsub("[0-9.-]", "", variable)]
  preds = dcast.data.table(preds, user + ranking ~ variable)
  setnames(preds, c("user", "rec"), c("cid", "aid"))
  preds[, cid := as.numeric(cid)]
  preds[, aid := as.numeric(aid)]
  preds[, ranking := NULL]
  preds[, score := as.numeric(score)]
  preds = preds[score > 0]
  preds[, wk_predict := wks_test[i]]
  res[[i]] = copy(preds)
  print(i)
}

sar_all = rbindlist(res)

saveRDS(sar_all, "data/t8/sar.rds")
