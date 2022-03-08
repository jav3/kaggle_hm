library(data.table)
library(dplyr)
library(lubridate)

# Read data ---------------------------------------------------------------

setwd("~/Downloads/Kaggle/hm")

articles = fread("data/articles.csv")
customers = fread("data/customers.csv")
trans = fread("data/transactions_train.csv")

# Latest date
max(trans$t_dat) # 2020-09-22

# Number of customers
length(unique(customers$customer_id)) # 1371980


# Build simple submission -------------------------------------------------
# Use last 12 purchases
# Fill with last week's hottest sales if less than 12 purchases

setkey(trans, customer_id)
trans = trans[order(customer_id, -t_dat)]
trans[, article_id := paste0("0", article_id)]

# Get hottest articles
hottest_articles = trans[t_dat >= max(trans$t_dat) %m-% weeks(1)]
hottest_articles = hottest_articles[, .(N = length(unique(customer_id))), by = .(article_id)][order(-N)][1:12]$article_id

# Get last 12 filled with hottest articles if less than 12 purchases
last12 = trans[, .(customer_id, article_id)][, .(prediction = list(head(unique(article_id), 12) %>% 
                                                                     {`if`(length(.) == 12, ., c(., hottest_articles[1:(12 - length(.))]))})), by = customer_id]

last12 = last12[, .(prediction = paste0(unlist(prediction), sep = " ", collapse = " ")), by = customer_id]
submission = merge(customers[, .(customer_id)], last12, by = "customer_id", all.x = T)

# Fill customers with no purchases
hottest_articles_string = paste0(hottest_articles, sep = " ", collapse = " ")
submission[is.na(prediction), prediction := hottest_articles_string]
data.table::fwrite(submission, file = "submission_last12filled.csv.gz")
# Scores 0.0201 on the leaderboard
