from sseclient import SSEClient as EventSource

from bloom_filter_fit import *

filter_type = 'user'  # 'random' or 'type'
filter_prob = 0.2
target_amount = 5000
log_freq = 100
URL = 'https://stream.wikimedia.org/v2/stream/recentchange'


def left_user(user): # TODO
    if filter_type == 'user':
        return hash(user['user']) % 100 > filter_prob * 100
    return True


def predict_bot(data, bloom_filter, hashes): # TODO add filter_type choice and ML classificator
    return is_str_in_filter(bloom_filter, hashes, data)


if __name__ == '__main__':
    bloom_filter, hashes = get_filter()
    # Start relay
    count = 0
    non_filtered_gt = []
    gt_bot = []
    pred_bot = []
    for event in EventSource(URL):
        if event.event == 'message':
            try:
                change = json.loads(event.data)
            except ValueError:
                continue

            non_filtered_gt.append(change["bot"])
            if not left_user(change):
                continue

            count += 1

            gt_bot.append(change['bot'])
            pred_bot.append(predict_bot(change, bloom_filter, hashes))

            if count % log_freq == 0:
                print('Processed {} events'.format(count))

            if count >= target_amount:
                break

    print('Accuracy: {}'.format(accuracy(gt_bot, pred_bot)))
    print('Non-filtered amount of bots/users: {}/{}'.format(sum(non_filtered_gt), len(non_filtered_gt) - sum(non_filtered_gt)))
    print('Filtered amount of bots/users: {}/{}'.format(sum(gt_bot), len(gt_bot) - sum(gt_bot)))
    print('Predicted amount of bots/users: {}/{}'.format(sum(pred_bot), len(pred_bot) - sum(pred_bot)))
