library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(pbapply)
library(caret)
library(glmnet)
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
wks_train = 2 # Train for 2 weeks prior to wks_test
wks_test = c(37, 38, 39) # Test on these weeks - the last one is for Kaggle


# Train -------------------------------------------------------------------

i = 1
week_end_date = as.Date("2020-01-01") %m+% days((wks_test[i]-1)*7 - 1)
wks = (wks_test[i] - wks_train):(wks_test[i] - 1)
half_life = 5 # 5 days assumption

# Prior purchases
prior_purchases_wtd = transactions %>% 
  filter(yr == curr_yr,  wk %in% wks) %>%
  select(cid, aid, t_dat) %>%
  collect()
setDT(prior_purchases_wtd)
prior_purchases_wtd = prior_purchases_wtd[, .(num_purchases_ac = .N), by = .(cid, aid)]
prior_purchases_wtd[, num_customers := length(unique(cid))]
prior_purchases_wtd[, num_customers_a := length(unique(cid)), by = .(aid)]
prior_purchases_wtd[, num_purchases_c := .N, by = .(cid)]
prior_purchases_wtd[, tf := num_purchases_ac/num_purchases_c]
prior_purchases_wtd[, idf := log(num_customers/num_customers_a)]
prior_purchases_wtd[, wt := tf*idf]

customers_data = customers %>% collect
setDT(customers_data)
customers_data[is.na(age), age := median(customers_data$age, na.rm = T)]
customers_data[, Active := ifelse(is.na(as.numeric(as.character(Active))), 0, 1)]
customers_data[, FN := ifelse(is.na(as.numeric(as.character(FN))), 0, 1)]
train = merge(prior_purchases_wtd, customers_data, by = "cid", all.x = T)

X = model.matrix(~ age + Active + FN - 1, data = train)

mod = cv.glmnet(x = X, y = train$wt, nfolds = 5, type.measure = "deviance")
plot(mod)
