import json
import random

train_path = ["data_50000.json", "data_100000.json", "data_150000.json", "data_200000.json", "data_250000.json",
              "data_300000.json", "data_350000.json", "data_400000.json", "data_450000.json", "data_500000.json",
              "data_550000.json", "data_600000.json", "data_650000.json", "data_700000.json", "data_750000.json"]
val_path = ["data_800000.json"]
filter_size = 10000
hash_functions = 20
save_bot = True


def load_data(paths):
    data = []
    for p in paths:
        with open(p, 'r') as f:
            data += json.load(f)
    return [{'bot': d['bot'], 'user': d['user']} for d in data]


def rand_hash(modulus, a=None, b=None):
    if a is None or b is None:
        a, b = random.randint(1, modulus - 1), random.randint(1, modulus - 1)
    # print(a, b)
    func = lambda x: hash(x) % (a + b) % modulus
    return {'func': func, 'a': a, 'b': b}


def random_hashes(modulus, amount_hash_functions, seed=42, params=None):
    # e.x params = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    random.seed(seed)
    if params is None:
        fns = [rand_hash(modulus) for _ in range(amount_hash_functions)]
    else:
        fns = [rand_hash(modulus, **param) for param in params]
    return fns


def is_str_in_filter(bloom_filter, hashes, data):
    for h in hashes:
        if bloom_filter[h['func'](data["user"])] == 0:
            return False
    return True


def eval_filter(bloom_filter, hashes, data):
    gt = [d['bot'] for d in data]
    pred = [is_str_in_filter(bloom_filter, hashes, d) for d in data]
    return gt, pred


def fit_filter(bloom_filter, hashes, data, save_bot=save_bot):
    for d in data:
        if d['bot'] != 1 and save_bot:
            continue
        for h in hashes:
            bloom_filter[h['func'](d['user'])] = 1

    return bloom_filter


def accuracy(gt, pred):
    return sum([1 for i in range(len(gt)) if gt[i] == pred[i]]) / len(gt)


def get_filter():
    train_data = load_data(train_path)
    val_data = load_data(val_path)

    total_train = len(train_data)
    amount_bots_train = len([d for d in train_data if d['bot'] == 1])

    total_val = len(val_data)
    amount_bots_val = len([d for d in val_data if d['bot'] == 1])

    print("Total train amount of users: ", total_train)
    print("Total val amount of users: ", total_val)

    print("Train amount of bots: ", amount_bots_train)
    print("Val amount of bots: ", amount_bots_val)

    # Calculate amount of the same bots from val in train
    names = set([d['user'] for d in train_data])
    same_users = 0
    same_bots = 0
    for d in val_data:
        if d["user"] in names:
            if d['bot'] == 0:
                same_users += 1
            else:
                same_bots += 1
    print("Amount of the same users from val in train: ", same_users)
    print("Amount of the same bots from val in train: ", same_bots)

    # with open('bloom_filter.json', 'r') as f:
    #     data = json.load(f)
    # bloom_filter_saved = data['filter']
    # hashes_a_b = data['hashes']
    bloom_filter = [0] * filter_size
    hashes = random_hashes(filter_size, hash_functions)#, params=hashes_a_b)

    bloom_filter_fitted = fit_filter(bloom_filter, hashes, train_data)

    gt, pred = eval_filter(bloom_filter_fitted, hashes, train_data)
    print("Train accuracy: ", round(accuracy(gt, pred), 3))

    gt, pred = eval_filter(bloom_filter_fitted, hashes, val_data)
    print("Val accuracy: ", round(accuracy(gt, pred), 3))

    # Save filter
    # with open('bloom_filter.json', 'w') as f:
    #     json.dump({'filter': bloom_filter_fitted, 'hashes': [{'a': h['a'], 'b': h['b']} for h in hashes]}, f)

    return bloom_filter_fitted, hashes


if __name__ == '__main__':
    get_filter()