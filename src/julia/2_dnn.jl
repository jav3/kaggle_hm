# Working dir
cd("/Users/javidali/Downloads/Kaggle/hm")

# Packages
using CSV
using DataFrames
using Dates
using Pipe
using Query
using Arrow
using Distributed
using ProgressMeter
using Statistics
using SparseArrays
using Flux
using LinearAlgebra
using StatsBase
using DataStructures
include("src/helpers.jl")

# Import data
transactions = DataFrame(Arrow.Table("data/transactions_m.arrow"))

customers = DataFrame(Arrow.Table("data/customers_m.arrow"))

articles = DataFrame(Arrow.Table("data/articles_m.arrow"))

# Filter on eight weeks of training period
transactions_f = filter(a -> a.t_dat > Date(2020,09,22) - Day(7*4), transactions, view = true)

num_customers = length(unique(transactions_f.cid))
num_articles = length(unique(transactions_f.aid))

customer_trans_count = @pipe combine(groupby(transactions_f, :cid), nrow) |> filter(:nrow => x -> x > 30, _)

trans = filter(:cid => in(customer_trans_count.cid), transactions_f)

# Matrix of customers x articles
num_customers = length(unique(trans.cid))
num_articles = length(unique(trans.aid))
customers_dict = Dict(val => idx for (idx, val) in enumerate(unique(trans.cid)))
articles_dict = Dict(val => idx for (idx, val) in enumerate(unique(trans.aid)))

leftjoin!(trans, customers, on = :cid)
leftjoin!(trans, articles[:, [:aid, :index_group_no, :index_code, :garment_group_no]], on = [:aid])


# Articles metadata

# Num purchases on single day by article
articles_meta = trans[:, [:aid, :t_dat]]
articles_meta = combine(groupby(articles_meta, [:aid, :t_dat]), nrow => :num_purchases)

# Num purchases in the 5 preceding days
articles_meta[!, :popularity] .= Float16(0.0)
sort!(articles_meta, :t_dat)
articles_meta[:, :num_purchases_past5] .= 0
for i in minimum(articles_meta.t_dat):Day(1):maximum(articles_meta.t_dat)
    # Get articles purchased over the the preceding 5 days
    sub_df = filter(:t_dat => x -> x >= i - Day(5) && x < i, articles_meta)
    num_purchases = combine(groupby(sub_df, [:aid]), :num_purchases => sum => :num_purchases)
    total_purchases = sum(num_purchases.num_purchases)
    num_purchases[:, :popularity] .= num_purchases.num_purchases ./(total_purchases * 1.0)
    num_purchases = Dict(Pair.(num_purchases.aid, num_purchases.popularity))

    # Get current day and merge on popularity metric
    curr_df = filter(:t_dat => x -> x == i, articles_meta, view = true)
    curr_df[:, :popularity] .= get.(Ref(num_purchases),  curr_df.aid, 0)
end

articles_meta = articles_meta[:, [:aid, :t_dat, :popularity]]

# Trans metadata

# Historical purchases 
trans_meta = trans[:, [:t_dat, :aid, :price, :cid, :index_group_no, :index_code, :garment_group_no]]
trans_meta[!, :last_index_group_no] .= Int64(0)
trans_meta[!, :last_index_code] .= String("")
trans_meta[!, :last_garment_group_no] .= Int64(0)
trans_meta[!, :mode_index_group_no] .= Int64(0)
trans_meta[!, :mode_index_code] .= String("")
trans_meta[!, :mode_garment_group_no] .= Int64(0)

sort!(trans_meta, :t_dat)
trans_group = groupby(trans_meta, :cid);

@showprogress for (k, sub_df) in pairs(trans_group)

    tmp = DataFrame("t_dat" => unique(sub_df.t_dat))
    tmp[!, :last_index_group_no] .=  Int64(0)
    tmp[!, :last_index_code] .=  String("")
    tmp[!, :last_garment_group_no] .=  Int64(0)
    tmp[!, :mode_index_group_no] .=  Int64(0)
    tmp[!, :mode_index_code] .=  String("")
    tmp[!, :mode_garment_group_no] .=  Int64(0)
    

    # Only get history if more than one date of purchase
    if (size(tmp.t_dat)[1] > 1)
        for i in 2:size(tmp.t_dat)[1]
            # Filter on dates less than dt
            sub_df2 = filter(:t_dat => x -> x < tmp.t_dat[i], sub_df, view = true)
    
            tmp[i, :last_index_group_no] = last(sub_df2.index_group_no)
            tmp[i, :last_index_code] = last(sub_df2.index_code)
            tmp[i, :last_garment_group_no] = last(sub_df2.garment_group_no)
            tmp[i, :mode_index_group_no] = mode(sub_df2.index_group_no)
            tmp[i, :mode_index_code] = mode(sub_df2.index_code)
            tmp[i, :mode_garment_group_no] = mode(sub_df2.garment_group_no)
        end
    end

    # Create dicts
    last_index_group_no = Dict(Pair.(tmp.t_dat, tmp.last_index_group_no))
    last_index_code = Dict(Pair.(tmp.t_dat, tmp.last_index_code))
    last_garment_group_no = Dict(Pair.(tmp.t_dat, tmp.last_garment_group_no))
    mode_index_group_no = Dict(Pair.(tmp.t_dat, tmp.mode_index_group_no))
    mode_index_code = Dict(Pair.(tmp.t_dat, tmp.mode_index_code))
    mode_garment_group_no = Dict(Pair.(tmp.t_dat, tmp.mode_garment_group_no))

    # Get historical purchase metrics and merge onto trans_meta
    curr_df = filter(:cid => in(k), trans_meta, view = true)
    curr_df[:, :last_index_group_no] .= get.(Ref(last_index_group_no),  curr_df.t_dat, 0)
    curr_df[:, :last_index_code] .= get.(Ref(last_index_code),  curr_df.t_dat, "")
    curr_df[:, :last_garment_group_no] .= get.(Ref(last_garment_group_no),  curr_df.t_dat, "")
    curr_df[:, :mode_index_group_no] .= get.(Ref(mode_index_group_no),  curr_df.t_dat, 0)
    curr_df[:, :mode_index_code] .= get.(Ref(mode_index_code),  curr_df.t_dat, "")
    curr_df[:, :mode_garment_group_no] .= get.(Ref(mode_garment_group_no),  curr_df.t_dat, "")
end

trans_meta = trans_meta[:, [:t_dat, :cid, :last_index_group_no, :last_index_code, :last_garment_group_no, :mode_index_group_no, :mode_index_code, :mode_garment_group_no]]
trans_meta = unique(trans_meta)

## Model
# Ref: https://armchairecology.blog/flux-seeds-dataset/

# Impute
trans[ismissing.(trans.age),:age] .= 30
trans[ismissing.(trans.fashion_news_frequency),:fashion_news_frequency] .= "NONE"
trans[ismissing.(trans.Active),:Active] .= "0.0"


# Add article metadata
leftjoin!(trans, articles_meta, on = [:aid, :t_dat])
leftjoin!(trans, trans_meta, on = [:cid, :t_dat])

# Data 
features = Matrix{Float32}(trans[:, [:age]])'
features = [features; 
unique(trans.index_group_no) .== permutedims(trans[!, :last_index_group_no]);
unique(trans.index_group_no) .== permutedims(trans[!, :mode_index_group_no]);
unique(trans.index_code) .== permutedims(trans[!, :last_index_code]);
unique(trans.index_code) .== permutedims(trans[!, :mode_index_code]);
unique(trans.garment_group_no) .== permutedims(trans[!, :last_garment_group_no]);
unique(trans.garment_group_no) .== permutedims(trans[!, :mode_garment_group_no]);
unique(trans.fashion_news_frequency) .== permutedims(trans[!, :fashion_news_frequency])]

# Normalize and one-hot
norm_transform = StatsBase.fit(ZScoreTransform, features, dims = 2)
features = StatsBase.transform(norm_transform, features)
labels = Flux.onehotbatch(trans.aid, sort(unique(trans.aid)))

# Train and test split
is_trn = trans.t_dat .< Date(2020,09,22) - Day(7)
trn_features = features[:, is_trn]
trn_labels = labels[:, is_trn]
tst_features = features[:, Not(is_trn)]
tst_labels = labels[:, Not(is_trn)]

data = Flux.DataLoader((trn_features, trn_labels), batchsize=128, shuffle = true) 

# Layers
W = Chain(Dense(75, 9271), softmax)

# Loss function
loss(x, y) = Flux.crossentropy(W(x), y)

# Simple measure
loss(tst_features, tst_labels)

# Optimizer
optimizer = Flux.Descent()

# Callback
evalcb() = @show(loss(tst_features, tst_labels))
throttled_cb = Flux.throttle(evalcb, 5)

Flux.@epochs 10 Flux.train!(loss, params(W), data, optimizer, cb = throttled_cb)

loss(trn_features, trn_labels)

@pipe StatsBase.countmap(Flux.onecold(W(trn_features))) |> DataStructures.sort(_, byvalue = true, rev = true)
@pipe StatsBase.countmap(Flux.onecold(trn_labels)) |> DataStructures.sort(_, byvalue = true, rev = true)

# Validation
trans[Not(is_trn),:]
Flux.onecold(W(trn_features))
W(tst_features)

mean(Flux.onecold(W(trn_features)) .== Flux.onecold(trn_labels))

# Predictions
pred_customer = deepcopy(DataFrame(customers))
leftjoin!(pred_customer, combine(last, groupby(trans_meta[trans_meta.t_dat .< Date(2020,09,22) - Day(7),:], [:cid])), on = :cid)

replace!(pred_customer.age, missing => 30)
replace!(pred_customer.fashion_news_frequency, missing => "NONE")
if (sum(ismissing(pred_customer.Active)) > 0) 
    replace!(pred_customer.Active, missing => "0.0")
end

sub_features = Matrix{Float32}(pred_customer[:, [:age]])'
sub_features = [sub_features; 
unique(trans.index_group_no) .== permutedims(pred_customer[!, :last_index_group_no]);
unique(trans.index_group_no) .== permutedims(pred_customer[!, :mode_index_group_no]);
unique(trans.index_code) .== permutedims(pred_customer[!, :last_index_code]);
unique(trans.index_code) .== permutedims(pred_customer[!, :mode_index_code]);
unique(trans.garment_group_no) .== permutedims(pred_customer[!, :last_garment_group_no]);
unique(trans.garment_group_no) .== permutedims(pred_customer[!, :mode_garment_group_no]);
unique(trans.fashion_news_frequency) .== permutedims(pred_customer[!, :fashion_news_frequency])]

replace!(sub_features, missing => 0)
sub_features = Matrix{Float32}(sub_features)

sub_features = StatsBase.transform(norm_transform, sub_features)
W(sub_features[:, 1:5000])

pred_loader = Flux.DataLoader(1:size(sub_features)[2], batchsize=5000);

sub_preds = zeros(Int32, 12, size(sub_features)[2])

@showprogress for dt in pred_loader
    res = W(sub_features[:, dt])
    sub_preds[:, dt] = mapslices(x -> partialsortperm(x, 1:12, rev = true), res; dims=1)
end

# Compute mapk
s1 = Dict{Int, Union{Missing, Float16}}()
for name in customers.cid
    s1[name] = missing
end

trans_group = groupby(filter(a -> a.t_dat > Date(2020,09,22) - Day(7), transactions, view = true), :cid);

@showprogress for (key, sub_df) in pairs(trans_group)
    preds = sort(unique(trans.aid))[sub_preds[:, key.cid]]
    curr_articles = sub_df.aid |> unique;
    res = helpers.apk(curr_articles, preds);
    setindex!(s1, res, key.cid)
end

# MAP@12 
@pipe values(s1) |> (sum(skipmissing(_))/(sum(skipmissing(_) .>=0)*1.0)) |> print("MAP :$_\n")
