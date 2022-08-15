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

articles_data = articles %>% collect
setDT(articles_data)
articles_data[, grp := .GRP, by = .(detail_desc)]

curr_yr = 2020
wks_train = 2 # Train for 2 weeks prior to wks_test
wks_test = c(37, 38, 39) # Test on these weeks - the last one is for Kaggle

# Article attr - dynamic --------------------------------------------------

# Create base data frame of articles by week
article_list = articles %>%
  distinct(aid) %>%
  collect %>% 
  as.data.table

res = vector("list", length(wks_test))

for (i in seq_len(length(wks_test))){
  
  week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
  wks = (wks_test[i] - wks_train):(wks_test[i] - 1)
  half_life = 5 # 5 days assumption
  
  # Popularity
  purchases = transactions %>%
    filter(yr == curr_yr, wk %in% wks) %>% 
    collect
  setDT(purchases)
  
  purchases[, days_to_end_of_train := as.numeric(week_end_date - t_dat)]
  purchases[, popularity := 1/2^(days_to_end_of_train/half_life)]
  
  total_popularity = sum(purchases$popularity, na.rm = T)
  total_popularity_1d_prior = sum(purchases[t_dat != week_end_date]$popularity, na.rm = T)
  
  popular_purchases = rbindlist(list(article_list, purchases[, .(aid, popularity, t_dat)]), use.names = T, fill = T)
  popular_purchases =popular_purchases[, .(popularity = sum(popularity, na.rm = T)/total_popularity, 
                                           popularity_1d_prior = sum(popularity[t_dat != week_end_date], na.rm = T)/total_popularity_1d_prior), 
                                       by = .(aid)]
  popular_purchases[, wk_predict := wks_test[i]]
  popular_purchases[is.na(popularity), popularity := 0]
  popular_purchases[is.na(popularity_1d_prior), popularity_1d_prior := 0]
  popular_purchases[, popularity_num := frank(-popularity)]
  popular_purchases[, popularity_num_1d_prior := frank(-popularity_1d_prior)]
  
  # First seen date and days on market
  first_seen = transactions %>%
    select(yr, wk, aid, t_dat) %>%
    filter(t_dat <= week_end_date) %>%
    group_by(aid) %>%
    summarise(first_seen = min(t_dat)) %>%
    collect
  setDT(first_seen)
  first_seen[, days_on_market := as.numeric(week_end_date - first_seen)]
  
  # Define new arrivals
  # AW20 upcoming
  release_date = as.Date("2022-09-22")
  if (week_end_date >= release_date){
    aw20_aid = articles %>% select(aid, detail_desc) %>% collect %>% filter(grepl("AW20", detail_desc)) %>% pull(aid)
    first_seen_aw20 = data.table(aid = aw20_aid, first_seen = release_date, 
                                 days_on_market = as.numeric(week_end_date - release_date))
    first_seen = rbindlist(list(first_seen, first_seen_aw20), use.names = T, fill = T)
    first_seen = unique(first_seen, by = "aid")
  }
  first_seen[, new_arrival_1w := days_on_market <= 7]
  first_seen[, new_arrival_2w := days_on_market <= 14]
  first_seen[, new_arrival_1m := days_on_market <= 28]
  
  # Last seen date and days since last seen
  last_seen = transactions %>%
    select(yr, wk, aid, t_dat) %>%
    filter(t_dat <= week_end_date) %>%
    group_by(aid) %>%
    summarise(last_seen = max(t_dat)) %>%
    collect
  setDT(last_seen)
  last_seen = last_seen[, days_since_last_seen := as.numeric(week_end_date - last_seen)]
  
  # Marked down
  # Markdown
  markdown = transactions %>%
    filter(yr == curr_yr) %>%
    filter(t_dat <= week_end_date) %>%
    group_by(yr, wk) %>%
    select(yr, wk, aid, t_dat, price, sales_channel_id) %>%
    collect()
  setDT(markdown)
  markdown[, price := price*590]
  setorder(markdown, sales_channel_id, aid, t_dat)
  markdown = markdown[, .(price = median(price)), by = .(sales_channel_id, aid, wk)]
  setorder(markdown, sales_channel_id, aid, wk)
  markdown[, last_week_price := lag(price), by = .(sales_channel_id, aid)]
  markdown[, last_month_price := lag(price, 4), by = .(sales_channel_id, aid)]
  markdown[, last_2month_price := lag(price, 8), by = .(sales_channel_id, aid)]
  markdown[, md_lw := price < last_week_price]
  markdown[, md_lm := price < last_month_price]
  markdown[, md_l2m := price < last_2month_price]
  markdown[, cum_min_price := cummin(price), by = .(sales_channel_id, aid)]
  markdown[, all_time_lowest := abs(price - cum_min_price) < 1]
  markdown[, discount := round((1 - price/max(price))*100, 0), by = .(aid)]
  markdown = markdown[sales_channel_id == 2 & wk == max(wk), .(aid, price, md_lw, md_lm, md_l2m, all_time_lowest, discount)]
  
  # Old versions
  similar_articles = articles %>% collect %>% as.data.table
  similar_articles[, grp := .GRP, by = .(detail_desc)]
  similar_articles = merge(similar_articles, first_seen[, .(aid, first_seen)], by = "aid", all.x = T)
  setorder(similar_articles, grp, first_seen, na.last = T)
  similar_articles[, old_version := first_seen < max(first_seen, na.rm = T) - months(3), by = grp]
  similar_articles = similar_articles[, .(aid, old_version)]
  
  # Stock estimate based on sales
  stock = transactions %>%
    filter(yr == curr_yr, t_dat == week_end_date) %>%
    select(cid, aid) %>%
    collect %>%
    as.data.table
  stock = stock[, .(stock_count = length(unique(cid))), by = .(aid)]
  
  # Chance of repurchase/return
  repurchases = transactions %>%
    filter(yr == curr_yr, wk <= max(wks) & wk >= max(wks) - 4) %>% 
    collect
  setDT(repurchases)
  repurchases = unique(repurchases, by = c("cid", "aid", "t_dat"))
  repurchases = repurchases[, num_purchases := .N, by = .(cid, aid)][order(cid, aid, t_dat)]
  avg_purchased = repurchases[, .(avg_purchased = mean(num_purchases)), by = .(aid)]
  avg_repurchase_time = repurchases[num_purchases > 1, .(ttnp = as.numeric(t_dat - shift(t_dat))), by = .(cid, aid)] %>%
    {.[, .SD[2:.N], by = .(cid, aid)]} %>%
    {rbindlist(list(., repurchases[num_purchases == 1, .(ttnp = 0), by = .(cid, aid)]), use.names = T, fill = T)} %>%
    {.[, .(ttnp = as.numeric(mean(ttnp))), by = .(aid)]}
  
  # Chance of repurchase/return - more mature data
  repurchases = transactions %>%
    filter(yr == curr_yr, wk < max(wks) & wk >= max(wks) - 4) %>% 
    collect
  setDT(repurchases)
  repurchases = unique(repurchases, by = c("cid", "aid", "t_dat"))
  repurchases = repurchases[, num_purchases := .N, by = .(cid, aid)][order(cid, aid, t_dat)]
  avg_purchased_m1w = repurchases[, .(avg_purchased_m1w = mean(num_purchases)), by = .(aid)]
  avg_repurchase_time_m1w = repurchases[num_purchases > 1, .(ttnp = as.numeric(t_dat - shift(t_dat))), by = .(cid, aid)] %>%
    {.[, .SD[2:.N], by = .(cid, aid)]} %>%
    {rbindlist(list(., repurchases[num_purchases == 1, .(ttnp = 0), by = .(cid, aid)]), use.names = T, fill = T)} %>%
    {.[, .(ttnp_m1w = as.numeric(mean(ttnp))), by = .(aid)]}
  
  # Hist repurchase
  hist_repurchase = repurchases[, .(hist_repurchase = sum(num_purchases > 1)/.N), by = .(aid)]
  
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
  ), by = .(aid)]
  
  hist_repurchase_same_grp = unique(repurchases, by = c("cid", "grp", "t_dat")) %>%
    {.[, num_purchases := .N, by = .(cid, grp)][order(cid, grp, t_dat)]} %>%
    {.[, .(hist_repurchase_same_grp = sum(num_purchases > 1)/.N), by = .(aid)]}
  
  all = popular_purchases %>%
    merge(first_seen, by = "aid", all.x = T) %>%
    merge(last_seen, by = "aid", all.x = T) %>%
    merge(markdown, by = "aid", all.x = T) %>%
    merge(similar_articles, by = "aid", all.x = T) %>%
    merge(stock, by = "aid", all.x = T) %>%
    merge(avg_purchased, by = "aid", all.x = T) %>%
    merge(avg_repurchase_time, by = "aid", all.x = T) %>%
    merge(avg_purchased_m1w, by = "aid", all.x = T) %>%
    merge(avg_repurchase_time_m1w, by = "aid", all.x = T) %>%
    merge(repurchases_same_grp_2, by = "aid", all.x = T) %>%
    merge(hist_repurchase, by = "aid", all.x = T) %>%
    merge(hist_repurchase_same_grp, by = "aid", all.x = T)
  
  res[[i]] = all
  print(i)
}

articles_dy = rbindlist(res)

saveRDS(articles_dy, "data/t8/articles_dy.rds")

