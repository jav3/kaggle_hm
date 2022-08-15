library(arrow)
library(data.table)
library(dplyr)
library(lubridate)
library(Metrics)
library(igraph)
library(pbapply)
library(lightgbm)
library(xgboost)
library(caret)
data.table::setDTthreads(1)

setwd("~/Downloads/Kaggle/hm")
source("./src/8_arrow_h.R")

w = c(0.0274, 0.0269, 0.0265, 0.0267)
w = w/sum(w)

dat1 = fread("submission_lgbm_8_with_similarity_with_mf.csv.gz")
dat2 = fread("submission_lgbm_8_with_similarity_repurch.csv.gz")
dat3 = fread("submission_lgbm_8_with_similarity_with_same_diff_cv.csv.gz")
dat4 = fread("submission_lgbm_8_with_similarity_with_d2v_top50.csv.gz")

dat1 = dat1[, aid := strsplit(prediction, " ")]
dat1 = dat1[, .(aid = unlist(aid)), by = .(customer_id)]
dat1[, rk := 1:.N, by = .(customer_id)]
dat1[, rkw := (1/(rk + 1))*w[1]]

dat2 = dat2[, aid := strsplit(prediction, " ")]
dat2 = dat2[, .(aid = unlist(aid)), by = .(customer_id)]
dat2[, rk := 1:.N, by = .(customer_id)]
dat2[, rkw := (1/(rk + 1))*w[2]]

dat3 = dat3[, aid := strsplit(prediction, " ")]
dat3 = dat3[, .(aid = unlist(aid)), by = .(customer_id)]
dat3[, rk := 1:.N, by = .(customer_id)]
dat3[, rkw := (1/(rk + 1))*w[3]]

dat4 = dat4[, aid := strsplit(prediction, " ")]
dat4 = dat4[, .(aid = unlist(aid)), by = .(customer_id)]
dat4[, rk := 1:.N, by = .(customer_id)]
dat4[, rkw := (1/(rk + 1))*w[4]]

dat_all = rbindlist(list(dat1, dat2, dat3, dat4))
dat_all = dat_all[, .(rkf = sum(rkw)), by = .(customer_id, aid)]
dat_all = dat_all[order(customer_id, -rkf)]

dat_all = dat_all[, head(.SD, 12), by = .(customer_id)]
to_submit = dat_all[, .(customer_id, article_id = aid)][, .(prediction = paste(head(article_id, 12), collapse = " ")), 
                                                    by = customer_id]
data.table::fwrite(to_submit, file = "submission_ens_best_3.csv.gz")
