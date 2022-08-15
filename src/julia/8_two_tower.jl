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
using TextAnalysis
using TextModels
using Embeddings
using StatsBase
include("src/julia/helpers.jl")
using Lathe
using Random

# Import data
transactions = DataFrame(Arrow.Table("data/transactions_m.arrow"))

customers = DataFrame(Arrow.Table("data/customers_m.arrow"))

# Normalize age
age_tf = StatsBase.fit(ZScoreTransform, collect(skipmissing(customers.age)))
customers.age = convert(Vector{Union{Missing, Float64}}, customers.age)
customers.age_tf = customers.age
replace!(customers.age_tf, missing => age_tf.mean[1])
customers.age_tf = StatsBase.transform(age_tf, collect(skipmissing(customers.age_tf)))

articles = DataFrame(Arrow.Table("data/articles_m.arrow"))

# Load Embeddings
using Embeddings
const embtable = load_embeddings(GloVe{:en}, 3)
const get_word_index = Dict(word=>ii for (ii,word) in enumerate(embtable.vocab))

function get_embedding(word)
    ind = get_word_index[word]
    emb = embtable.embeddings[:,ind]
    return emb
end

function avg_embedding(sentence)
    sentence = replace(sentence, "microfibre"=>"microfiber")
    sentence = replace(sentence, "underwired"=>"underwire")
    wrd_split = split(sentence, " ", keepempty=false) 
    if (size(wrd_split)[1] > 0)
        emb = hcat([get_embedding(w) for w in wrd_split if w in keys(get_word_index)]...)
        if (isempty(emb))
            res = zeros(size(embtable.embeddings)[1])
        else
            res = mean(emb, dims = 2)
        end
    else
        res = zeros(size(embtable.embeddings)[1])
    end    
    return res
end


# Convert article description to embedding
crps = Corpus(StringDocument.(replace(articles.detail_desc, missing => "")))
prepare!(crps, strip_punctuation | strip_case | strip_stopwords | strip_articles)
crps_emb = [avg_embedding(ss) for ss in text.(crps)]
sum(crps_emb .=== missing)
crps_emb_mt = reduce(hcat, crps_emb)
crps_tf = StatsBase.fit(ZScoreTransform, crps_emb_mt)
crps_emb_mt = StatsBase.transform(crps_tf, crps_emb_mt)


# Filter on two weeks for training and last week for testing
trans = filter(a -> a.t_dat > Date(2020,09,22) - Day(2*7), transactions)

num_customers = length(unique(trans.cid))
num_articles = length(unique(trans.aid))

# Matrix of customers x articles
num_customers = length(unique(trans.cid))
num_articles = length(unique(trans.aid))
customers_dict = Dict(val => idx for (idx, val) in enumerate(unique(trans.cid)))
articles_dict = Dict(val => idx for (idx, val) in enumerate(unique(trans.aid)))

aid_lookup = Dict(aidx => aid for (aid, aidx) in articles_dict)

# Index of articles in subset
trans[:, :cidx] = getindex.(Ref(customers_dict), trans[:, :cid])
trans[:, :aidx] = getindex.(Ref(articles_dict), trans[:, :aid]) 

train = filter(a -> a.t_dat <= Date(2020,09,22) - Day(1*7), trans)[:, [:aid,:cid, :aidx, :cidx]] 
train = unique(train)

resp = filter(a -> a.t_dat > Date(2020,09,22) - Day(1*7), trans)[:, [:aid,:cid, :aidx, :cidx]]
resp = filter(:cidx => in(unique(train.cidx)), resp)
resp = unique(resp)

num_purchases = spzeros(num_articles, num_customers)
num_purchases[CartesianIndex.(train.aidx, train.cidx)] .= 1

# Split cid's into train and test
function split_train_test(resp, split_size = 0.5, seed = 100)
    Random.seed!(seed)
    cids_to_sample = unique(resp.cidx)
    train_size = Int(floor(length(cids_to_sample)*split_size))
    cids_train = sample(cids_to_sample, train_size)
    cids_test = setdiff(cids_to_sample, cids_train)
    resp_train = filter(:cidx => in(cids_train), resp)
    resp_test = filter(:cidx => in(cids_test), resp)

    return (resp_train, resp_test)
end

resp_split = split_train_test(resp)

# Mini-batches with negative sampling
function make_batch(resp_train, batch_size = 100)

    # Sample of rows to form batch
    batch_ids = sample(1:size(resp_train)[1], batch_size)    

    # Get cid's
    batch_cidx = unique(resp_train.cidx[batch_ids])

    # Get aid's from batch
    batch_aidx = unique(resp_train.aidx[batch_ids])
    
    negatives = Dict(map(x -> x => setdiff(batch_aidx, resp_train[resp_train.cidx .== x, :aidx]), batch_cidx))

    batch_w_negatives = Dict(map(x -> resp_train[x, :cidx] => [resp_train[x, :aidx], negatives[resp_train[x, :cidx]]...], batch_ids))

    batch_w_negatives_length = Dict(i => length(batch_w_negatives[i]) for i in  keys(batch_w_negatives))

    user_query = hcat([num_purchases[:,repeat([key], val)] for (key, val) in batch_w_negatives_length]...)

    positives = zeros(size(user_query, 2))
    positives_idx = cumsum([1, collect(values(batch_w_negatives_length))...])
    deleteat!(positives_idx, length(positives_idx))
    positives[positives_idx] .= 1

    art_query =  hcat([crps_emb_mt[:, map(x -> aid_lookup[x], i)] for i in collect(values(batch_w_negatives))]...)

    return (positives, (user_query, art_query))
end

# Compute performance on testing data for single cidx
function perf(resp_test, cidx)
    user_query = num_purchases[:,cidx]
    pred = model((user_query, crps_emb_mt))
    pred_aid = articles[partialsortperm(-pred[1,:], 1:12),:].aid
    res = filter(:cidx => in(cidx), resp_test).aid
    recall = helpers.recall_at_k(res, pred_aid, 12)
    return recall
end

function perf_all(resp_test)
    res = [perf(x, key.cidx) for (key, x) in pairs(groupby(resp_test, :cidx)[1:100])]
    return (mean(collect(skipmissing(res))))
end


# Custom join layer
struct Join{T, F}
    combine::F
    paths::T
  end
  
  # Allow Join(op, m1, m2, ...) as a constructor
  Join(combine, paths...) = Join(combine, paths)

  Flux.@functor Join

  (m::Join)(xs::Tuple) = m.combine(map((f, x) -> f(x), m.paths, xs)...)
  (m::Join)(xs...) = m(xs)

# Model
model =  Chain(
    Join(
        .*,
        Dense(num_articles, 16),
        Dense(200, 16)
    ),
    x -> mean(x, dims = 1)
)

# Loss
function loss(y, x)
    delta = 0.1
    res = model(x)
    loss = mean(max.(delta .+ transpose(res) .* (1 .- 2 .* y), 0))
    return loss
end

opt = ADAM(0.01)
parameters = params(model)

# Validation data
Random.seed!(123)
valid_batch = make_batch(resp_split[2], 1000)


for epoch = 1:20
    println("Epoch $epoch... creating batch")
    curr_batch = make_batch(resp_split[1], 200)
    println("Epoch $epoch... calculating gradients")
    grads = Flux.gradient(parameters) do
        loss(curr_batch...)
    end
    println("Epoch $epoch... optimizing parameters")
    Flux.Optimise.update!(opt, parameters, grads)

    if epoch % 5 == 0
        println("Validating at epoch $epoch...")
        valid_loss = 0; #loss(valid_batch...)
        valid_map = perf_all(resp_split[2])
        println("Validation loss: $valid_loss. Validation recall@12: $valid_map")
    end
end    

