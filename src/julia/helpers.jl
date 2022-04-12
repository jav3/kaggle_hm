module helpers

function update_vec!(a::Vector, x)

    # If already in data, do nothing
    if x in skipmissing(a)
        return
    end

    # If not yet populated, set the first value to `x`
    if a[1] === missing
        a[1] = x
    else
        # Shift the array and change the first value to `x`
        a = circshift(a, 1)
        a[1] = x
    end
end

function precision_at_k(y_true, y_pred, k = 12)::Float16
    common = intersect(y_true, y_pred[1:k])
    return (length(collect(skipmissing(common)))/k)
end

function rel_at_k(y_true, y_pred, k = 12)::Float16
    if y_pred[k] === missing 
        return 0
    elseif y_pred[k] in skipmissing(y_true)
        return 1
    else
        return 0
    end
end

function apk(y_true, y_pred, k = 12)
    # If all actuals are missing, return missing
    if all(y_true.===missing)
        return missing
    end

    apk = 0.0
    for i in 1:k
        apk = apk + precision_at_k(y_true, y_pred, i)*rel_at_k(y_true, y_pred, i)
    end

    return (apk/min(k, length(y_true)))
end

function mapk(dict_true, dict_pred, k = 12)
    mapk = 0.0
    n = 0
    for (key, value) in dict_true
        if (all(value.data.===missing))
            # Do nothing
        else
            mapk += apk(value.data, dict_pred[key].data)
        end
    end
end

end