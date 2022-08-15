toIgnore = c("aid", "cid", "folds")
character_cols = c("garment_group_no", "modal_index_group_no", "modal_index_code")
response = "purchased"

rolling_cv <- function(
  wk_cv = 35:37,
  SAMP_FRAC = 0.01,
  CV_FOLDS = 3,
  TOPX = 20,
  eta = 0.03,
  max_depth = 8,
  num_leaves = 31,
  min_data_in_leaf = 300,
  res_all = data.table()
){
  
  params = list(
    objective = "lambdarank", 
    feature_fraction = 0.67,
    bagging_fraction = 0.67,
    eta = eta,
    max_depth = max_depth,
    metric = "map",
    eval_at = 12,
    num_leaves = num_leaves,
    min_data_in_leaf = min_data_in_leaf
  )
  
  for (i in wk_cv){
    curr_yr = 2020
    curr_wk = i
    
    res = cbind(as.data.table(params), "train_week" = curr_wk, val = NA)
    
    wks = (curr_wk - 2):(curr_wk - 1)
    # Train -------------------------------------------------------------------
    
    purchased_customers = transactions %>% 
      filter(yr == curr_yr, wk == curr_wk) %>% 
      select(aid, cid) %>% 
      distinct %>% 
      mutate(purchased = 1) %>% 
      collect() %>% 
      as.data.table
    num_purchased_total = length(unique(purchased_customers$cid))
    
    # Sample customers
    set.seed(100)
    cids_sample = sample(unique(purchased_customers$cid), size = floor(num_purchased_total*SAMP_FRAC), replace = F)
    
    negatives = executeStrategy(curr_yr = curr_yr, curr_wk = curr_wk, 
                                topx = TOPX, wks = wks, cids = cids_sample)
    
    candidates = constructCandidates(negatives = negatives, 
                                     purchased_customers = purchased_customers %>% filter(cid %in% cids_sample), 
                                     curr_yr = curr_yr, curr_wk = curr_wk, wks = wks, keep_purchased = F)
    
    # Fill missing strategy values with 0
    to_fill = colnames(candidates)[grepl("s[0-9]_.*", colnames(candidates))]
    for (col in to_fill){
      set(candidates, which(is.na(candidates[[col]])), col, 0)
    }
    
    # Remove any cases where the customer didn't purchase anything
    train = candidates[, if(sum(purchased, na.rm = T) > 0) .SD, by = .(cid)]
    
    # Order the data
    train = train[order(cid)]
    query_groups_train = train[, .N, by = .(cid)]$N
    
    x = setdiff(colnames(train), c(toIgnore, response))
    
    dtrain <- lgb.Dataset(train[, x, with = F] %>% as.matrix, label = train[[response]],
                          group =  query_groups_train, categorical_feature = character_cols)
    
    set.seed(123)
    model = lgb.cv(params = params, data = dtrain, nrounds = 400, eval_freq = 20, nfold = CV_FOLDS, verbose = -1, early_stopping_rounds = 20)
    best_iter = model$best_iter
    
    set.seed(123)
    model = lgb.train(params = params, data = dtrain, nrounds = best_iter, valids = list("valid" = dtrain), eval_freq = 20)
    
    # Validate ----------------------------------------------------------------
    
    curr_yr = 2020
    curr_wk = curr_wk + 1
    wks = (curr_wk - 2):(curr_wk - 1)
    
    purchased_customers = transactions %>% 
      filter(yr == curr_yr, wk == curr_wk) %>% 
      select(aid, cid) %>% 
      distinct %>% 
      mutate(purchased = 1) %>% 
      collect() %>% 
      as.data.table
    num_purchased_total = length(unique(purchased_customers$cid))
    
    # Sample customers
    set.seed(100)
    cids_sample = sample(unique(purchased_customers$cid), size = floor(num_purchased_total*SAMP_FRAC), replace = F)
    
    negatives = executeStrategy(curr_yr = curr_yr, curr_wk = curr_wk, 
                                topx = TOPX, wks = wks, cids = cids_sample)
    candidates = constructCandidates(negatives = negatives, 
                                     purchased_customers = purchased_customers %>% filter(cid %in% cids_sample), 
                                     curr_yr = curr_yr, curr_wk = curr_wk, wks = wks, keep_purchased = F)
    
    rm(negatives)
    gc()
    
    # Fill missing strategy values with 0
    to_fill = colnames(candidates)[grepl("s[0-9]_.*", colnames(candidates))]
    for (col in to_fill){
      set(candidates, which(is.na(candidates[[col]])), col, 0)
    }
    
    map12 = validateModel(candidates, model, curr_yr = curr_yr, curr_wk = curr_wk, cids = cids_sample, cols = x)
    
    res$val = map12
    
    res_all = rbindlist(list(res_all, res), use.names = T, fill = T)
  }
  return (res_all)
}

cv_runs = data.table()

grd = expand.grid(max_depth = c(8, 12, 20), num_leaves = c(31, 51, 75, 100), min_data_in_leaf = c(5, 30, 100, 300))
for (i in 1:nrow(grd)){
  res = rolling_cv(max_depth = grd$max_depth[i], num_leaves = grd$num_leaves[i], min_data_in_leaf = grd$min_data_in_leaf[i])
  cv_runs = rbindlist(list(cv_runs, res), use.names = T, fill = T)
}

grd = expand.grid(max_depth = c(25, 30, 35), num_leaves = c(31), min_data_in_leaf = c(5, 30))
for (i in 1:nrow(grd)){
  res = rolling_cv(max_depth = grd$max_depth[i], num_leaves = grd$num_leaves[i], min_data_in_leaf = grd$min_data_in_leaf[i])
  cv_runs = rbindlist(list(cv_runs, res), use.names = T, fill = T)
}

grd = expand.grid(max_depth = c(20), num_leaves = c(16), min_data_in_leaf = c(30))
for (i in 1:nrow(grd)){
  res = rolling_cv(max_depth = grd$max_depth[i], num_leaves = grd$num_leaves[i], min_data_in_leaf = grd$min_data_in_leaf[i])
  cv_runs = rbindlist(list(cv_runs, res), use.names = T, fill = T)
}


cv_runs[, .(cv = mean(val), max_cv = max(val), min_cv = min(val)), by = c(setdiff(colnames(cv_runs), c("train_week", "val")))][order(-cv)]

cv_runs[num_leaves == 31 & min_data_in_leaf == 30 & max_depth == 20]

