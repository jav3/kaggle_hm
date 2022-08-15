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

articles_dy = readRDS("data/t8/articles_dy.rds")
setDT(articles_dy)

articles_data = articles %>% collect
setDT(articles_data)
articles_data[, grp := .GRP, by = .(detail_desc)]

# d2v = readRDS("data/t8/articles_doc2vec.rds")
# setDT(d2v)

curr_yr = 2020
wks_train = 4 # Train for this many weeks prior to wks_test
wks_test = c(37, 38, 39) # Test on these weeks - the last one is for Kaggle

# Article attr - dynamic --------------------------------------------------

# Create base data frame of articles by week
customer_list = customers %>%
  distinct(cid) %>%
  collect %>% 
  as.data.table

res = vector("list", length(wks_test))

for (i in seq_len(length(wks_test))){
  
  week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
  wks = (wks_test[i] - wks_train):(wks_test[i] - 1)
  half_life = 5 # 5 days assumption
  
  # Behaviour
  behaviour = transactions %>%
    filter(yr == curr_yr, wk %in% wks) %>%
    collect %>%
    merge(articles %>% select(aid, index_code, index_group_no, garment_group_no) %>% collect, by = "aid") %>%
    merge(articles_dy[wk_predict == wks_test[i], .(aid, days_on_market, all_time_lowest, popularity, old_version)], by = "aid")
  setDT(behaviour)
  setorder(behaviour, cid, t_dat)
  
  behaviour[, price := price*590]
  behaviour[, days_to_end_of_train := as.numeric(week_end_date - t_dat)]
  behaviour[, popularity := 1/2^(days_to_end_of_train/half_life)]
  
  behaviour = behaviour[, .(mean_purchases_cw_ch1 = mean(sales_channel_id == 1, na.rm = T),
                            mean_purchases_cw_ch2 = mean(sales_channel_id == 2, na.rm = T),
                            modal_index_code = Mode(index_code),
                            modal_index_group_no = Mode(index_group_no),
                            modal_garment_group_no = Mode(garment_group_no),
                            preferred_channel = Mode(sales_channel_id),
                            average_channel = weighted.mean(sales_channel_id, popularity, na.rm = T),
                            median_price = median(price, na.rm = T),
                            average_price = weighted.mean(price, popularity, na.rm = T),
                            last_channel = last(sales_channel_id), 
                            median_days_on_market = median(days_on_market, na.rm = T),
                            average_days_on_market = weighted.mean(days_on_market, popularity, na.rm = T),
                            median_all_time_lowest = median(as.numeric(all_time_lowest), na.rm = T),
                            average_all_time_lowest = weighted.mean(as.numeric(all_time_lowest), popularity, na.rm = T),
                            median_popularity = median(popularity, na.rm = T),
                            average_popularity = mean(popularity, na.rm = T)
  ), by = .(cid)]

  # Chance of repurchase/return
  repurchases = transactions %>%
    filter(yr == curr_yr, wk <= max(wks) & wk >= max(wks) - 4) %>% 
    collect
  setDT(repurchases)
  repurchases = unique(repurchases, by = c("cid", "aid", "t_dat"))
  repurchases = repurchases[, num_purchases := .N, by = .(cid, aid)][order(cid, aid, t_dat)]
  avg_purchased = repurchases[, .(avg_purchased = mean(num_purchases)), by = .(cid)]
  avg_repurchase_time = repurchases[num_purchases > 1, .(ttnp = as.numeric(t_dat - shift(t_dat))), by = .(cid, aid)] %>%
    {.[, .SD[2:.N], by = .(cid, aid)]} %>%
    {rbindlist(list(., repurchases[num_purchases == 1, .(ttnp = 0), by = .(cid, aid)]), use.names = T, fill = T)} %>%
    {.[, .(ttnp = as.numeric(mean(ttnp))), by = .(cid)]}
  
  # Hist repurchase
  hist_repurchase = repurchases[, .(hist_repurchase = sum(num_purchases > 1)/.N), by = .(cid)]
  
  # Chance of repurchase/return - more mature data
  repurchases = transactions %>%
    filter(yr == curr_yr, wk < max(wks) & wk >= max(wks) - 4) %>% 
    collect
  setDT(repurchases)
  repurchases = unique(repurchases, by = c("cid", "aid", "t_dat"))
  repurchases = repurchases[, num_purchases := .N, by = .(cid, aid)][order(cid, aid, t_dat)]
  avg_purchased_m1w = repurchases[, .(avg_purchased_m1w = mean(num_purchases)), by = .(cid)]
  avg_repurchase_time_m1w = repurchases[num_purchases > 1, .(ttnp = as.numeric(t_dat - shift(t_dat))), by = .(cid, aid)] %>%
    {.[, .SD[2:.N], by = .(cid, aid)]} %>%
    {rbindlist(list(., repurchases[num_purchases == 1, .(ttnp = 0), by = .(cid, aid)]), use.names = T, fill = T)} %>%
    {.[, .(ttnp_m1w = as.numeric(mean(ttnp))), by = .(cid)]}
  
  # Chance of purchase similar version
  repurchases = transactions %>%
    filter(yr == curr_yr, wk <= max(wks) & wk >= max(wks) - 4) %>% 
    collect
  setDT(repurchases)
  repurchases = merge(repurchases, articles_data[, .(aid, grp)], by = "aid", all.x = T)
  repurchases = unique(repurchases, by = c("cid", "aid", "t_dat"))
  repurchases_2 = repurchases[repurchases, 
                              on = .(cid = cid, grp = grp, t_dat < t_dat), 
                              .(cid = cid, grp = grp, aidX = x.aid, tdatX = x.t_dat, aidY = i.aid, tdatY = i.t_dat),
                              nomatch = 0L,
                              allow.cartesian = T]
  repurchases_same_grp = repurchases_2[, .(num_purchases_same_grp_after = .N,
                                           days_to_next_purchase_after = as.numeric(min(tdatY - tdatX))), by = .(cid, aidX, grp)]
  repurchases_same_grp[, num_purchases_same_grp := max(num_purchases_same_grp_after), by = .(cid, grp)]
  repurchases_same_grp[, days_to_next_purchase := min(days_to_next_purchase_after), by = .(cid, grp)]
  repurchases_same_grp[, grp := NULL]
  setnames(repurchases_same_grp, "aidX", "aid")
  repurchases_same_grp_2 = repurchases_same_grp[, .(avg_num_purchases_same_grp_after = mean(num_purchases_same_grp_after),
                                                    avg_days_to_next_purchase_after = mean(days_to_next_purchase_after),
                                                    avg_num_purchases_same_grp = mean(num_purchases_same_grp),
                                                    avg_days_to_next_purchase = mean(days_to_next_purchase)
  ), by = .(cid)]
  
  hist_repurchase_same_grp = unique(repurchases, by = c("cid", "grp", "t_dat")) %>%
    {.[, num_purchases := .N, by = .(cid, grp)][order(cid, grp, t_dat)]} %>%
    {.[, .(hist_repurchase_same_grp = sum(num_purchases > 1)/.N), by = .(cid)]}
  
  # Last seen date
  last_seen_c = transactions %>%
    filter(t_dat <= week_end_date) %>%
    select(yr, wk, cid, t_dat) %>%
    collect
  setDT(last_seen_c)
  last_seen_c = last_seen_c[, .(last_seen_c = max(t_dat)), by = .(cid)]
  last_seen_c[, days_since_last_seen_c := as.numeric(week_end_date - last_seen_c)]
  
  all = customer_list %>%
    mutate(wk_predict := wks_test[i]) %>%
    merge(behaviour, by = "cid", all.x = T) %>%
    merge(last_seen_c, by = "cid", all.x = T) %>%
    merge(avg_purchased, by = "cid", all.x = T) %>%
    merge(avg_repurchase_time, by = "cid", all.x = T) %>%
    merge(avg_purchased_m1w, by = "cid", all.x = T) %>%
    merge(avg_repurchase_time_m1w, by = "cid", all.x = T) %>%
    merge(repurchases_same_grp_2, by = "cid", all.x = T) %>%
    merge(hist_repurchase, by = "cid", all.x = T) %>%
    merge(hist_repurchase_same_grp, by = "cid", all.x = T)
  
  res[[i]] = all
  print(i)
}

customers_dy = rbindlist(res)

saveRDS(customers_dy, "data/t8/customers_dy.rds")

