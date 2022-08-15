library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(pbapply)
library(caret)
data.table::setDTthreads(4)
library(ggplot2)

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

customers_data = customers %>% collect
setDT(customers_data)

# Dynamic attr
customers_dy = readRDS("data/t7/customers_dy.rds")
setDT(customers_dy)
articles_dy = readRDS("data/t7/articles_dy.rds")
setDT(articles_dy)

# Similar articles
similar_articles_0 = readRDS("data/t7/similar_articles.rds")
similar_articles_0[ , linked_aid := lapply(1:.N, function(x) c(linked_aid[[x]]))]
similar_articles_0[ , score := lapply(1:.N, function(x) c( score[[x]]))]
similar_articles = similar_articles_0[, .(linked_aid = unlist(linked_aid), score = unlist(score)), by = .(aid, desc_idx)]
similar_articles = similar_articles[score > 0.5]
similar_articles = merge(similar_articles, articles_data[, .(aid, index_group_name_o = index_group_name)], by = "aid")
similar_articles = merge(similar_articles, articles_data[, .(linked_aid = aid, index_group_name_l = index_group_name)], by = "linked_aid")
similar_articles = similar_articles[index_group_name_o == index_group_name_l]
similar_articles = merge(similar_articles, articles_dy[wk_end - first_seen <= 14, .(sales = max(num_purchases_aw, na.rm = T)), by = .(aid)], 
                         by.x = "linked_aid", by.y = "aid")
similar_articles = similar_articles[, .(sales = max(sales, na.rm = T)), by = .(aid)]
similar_articles = merge(similar_articles, similar_articles_0[, .(aid, linked_aid, score)], by = "aid")
similar_articles = merge(similar_articles, articles_dy[!is.na(first_seen), .(aid, first_seen)] %>% unique, by = "aid", all.x = T)

similar_articles[first_seen == "2020-09-11"][order(-sales)]
articles_dy[wk == 38][order(-num_purchases_aw)]

articles_data[aid %in% articles_dy[week(first_seen) == 38 & year(first_seen) == 2020][order(-num_purchases_aw)]$aid[1:10]]
similar_articles_0[aid %in% articles_dy[week(first_seen) == 38 & year(first_seen) == 2020][order(-num_purchases_aw)]$aid[1:10]]

articles_data[aid %in% articles_dy[is.na(first_seen) & wk == 38]$aid]

articles_dy[aid %in% articles_data[grepl(pattern = "coat", detail_desc, ignore.case = T)]$aid][order(-num_purchases_aw)]

articles_data[aid %in% articles_dy[is.na(first_seen) & wk == 38]$aid & grepl(pattern = "AW20", detail_desc) ]
similar_articles[aid %in% articles_data[grepl(pattern = "AW20", detail_desc)]$aid ]

articles_dy[wk == 38][aid %in% articles_data[grepl(pattern = "AW20", detail_desc)]$aid][is.na(days_on_market) | days_on_market < 30]

curr_yr = 2020
wks = 34:39


# Article price change ----------------------------------------------------

purchases = transactions %>% 
#  filter(yr == curr_yr, wk %in% wks) %>%
  select(aid, t_dat, price, sales_channel_id, cid) %>%
  collect

setDT(purchases)
purchases[, price := round(price*590, 2)]
purchases[, sales_channel_id := as.character(sales_channel_id)]

purchase = unique(purchases, by = c("aid", "cid", "t_dat"))

purchases = purchases[order(aid, t_dat)]
purchases[, price_run := rleid(price), by = .(aid)]

purchases[t_dat >= max(t_dat)-5, .N, by = .(t_dat, aid, sales_channel_id)][order(-N)]



sel_aid = c(14012)

# merge(transactions %>% filter(yr == curr_yr & aid == sel_aid, sales_channel_id == 1) %>% collect %>% as.data.table, customers_data, by = "cid") %>% {.[, .(.N), by = .(postal_code)][order(-N)]}

articles_data[aid %in% sel_aid]
similar_articles[aid %in% sel_aid]
similar_articles_0[aid %in% sel_aid]
articles_dy[aid %in% sel_aid][num_purchases_aw > 0][order(aid, wk)]

ggplot(purchases[aid %in% sel_aid][, .(price = median(price), N = as.numeric(.N)), by = .(t_dat, sales_channel_id, aid = factor(aid))] %>%
         melt.data.table(id.vars = c("t_dat", "sales_channel_id", "aid")), 
       aes(x = t_dat, y = value, color = aid)) + 
  geom_point() + geom_smooth() + scale_x_date(date_breaks = "1 month") + 
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, size = 6)) + 
  facet_wrap(~ variable + sales_channel_id, scales = "free") 

# Submission --------------------------------------------------------------
# Not run
if (F){
  to_submit = customers %>% collect %>% select(cid)
  
  # New campaign
  aid_campaign = articles_data[aid %in% articles_dy[is.na(first_seen) & wk == 38]$aid & 
                                 grepl(pattern = "AW20", detail_desc) & product_group_name == "Garment Upper body" & 
                                 grepl("blouse|jersey|coat|cardigan", detail_desc, ignore.case = T)] %>% 
    select(aid) %>% 
    merge(article_mapping, by = "aid", sort = F)  %>%
    pull(article_id) %>%
    paste0(sep = "", collapse = " ")
  
  to_submit2 = customers %>% select(cid) %>% 
    merge(customer_mapping, by = "cid", sort = F)
  setDT(to_submit2)
  
  to_submit3 = to_submit2[, .(customer_id, prediction = aid_campaign)]
  data.table::fwrite(to_submit3, file = "submission_aw20.csv.gz")
}

