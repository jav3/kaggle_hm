library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)
library(caret)
library(quanteda)
library(Matrix)
library(text2vec)
library(quanteda.textstats)
library(word2vec)
library(doc2vec)

setwd("~/Downloads/Kaggle/hm")


# Read data ---------------------------------------------------------------

articles = open_dataset("data/articles_m.arrow", format = "arrow")
customers = open_dataset("data/customers_m.arrow", format = "arrow")
transactions = open_dataset("data/transactions_p/", format = "arrow")

article_mapping = open_dataset("data/article_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(article_mapping, 1:2, c("article_id", "aid"))
customer_mapping = open_dataset("data/customer_mapping.arrow", format = "arrow") %>% collect %>% setDT
setnames(customer_mapping, 1:2, c("customer_id", "cid"))

articles_data = collect(articles)
setDT(articles_data)
articles_data[, desc_idx := .GRP, by = .(detail_desc)]

num_articles = articles %>% pull(aid) %>% n_distinct
num_customers = customers %>% pull(cid) %>% n_distinct
aid_list = articles %>% pull(aid)
cid_list = customers %>% pull(cid)


# Get all transactions ----------------------------------------------------

trans = transactions %>% collect
setDT(trans)
setorder(trans, cid, t_dat)

trans = trans[t_dat <= as.Date("2020-09-15")]
trans[, diff_days := as.numeric(t_dat - shift(t_dat, fill = 0)), by = .(cid)]
trans[, new_grp := (diff_days > 30)*1]
trans[, new_grp := cumsum(new_grp), by = .(cid)]
trans[, grp_size := .N, by = .(cid, new_grp)]

trans_2 = trans[grp_size > 1, .(art_str = paste(unique(aid), collapse = " "), t_dat = max(t_dat)), by = .(cid, new_grp)]


# Compute dfm -------------------------------------------------------------
art_tokens = tokens(trans_2$art_str)
art_fcm = fcm(art_tokens, context = "window", count = "weighted", weights = 1 / (1:5), tri = TRUE)

glove = GlobalVectors$new(rank = 50, x_max = 10)
wv_main = glove$fit_transform(art_fcm, n_iter = 100, convergence_tol = 0.01, n_threads = 8)
wv_context = glove$components
word_vectors = wv_main + t(wv_context)

a = word_vectors["95507", , drop = F]
cos_sim <- textstat_simil(x = as.dfm(word_vectors), y = as.dfm(a), method = "cosine")
head(sort(cos_sim[, 1], decreasing = TRUE), 5)


# Item similarity: I - similarity matrix of items - N x N
# Similarity a function of two items, e.g. purchased together, similarity of descriptions, etc. 
# e.g. f(s(i1, i2)) where s() is a measure of similarity of two items
#
# User affinity: U - which purchases user interacted with previously - 1 x N - each entry is user affinity to given item
# g(u, i) = w(user features, activity, inactive/new indicator, bias_i) + h(u, i)
# e.g. h(u, i) is a function of time purchased and number of purchases from user
# Introduce K features

