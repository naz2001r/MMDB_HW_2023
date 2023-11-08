import json
from sseclient import SSEClient as EventSource

EVENTS = 10e6  # 1 million
SAVE_FREQ = 5000
URL = 'https://stream.wikimedia.org/v2/stream/recentchange'

if __name__ == '__main__':
    # Collect data from the stream as csv file
    count = 0
    data = []

    for event in EventSource(URL):
        if event.event == 'message':
            try:
                change = json.loads(event.data)
            except ValueError:
                continue

            data.append(change)

            count += 1

            if count >= EVENTS:
                break

            if count % SAVE_FREQ == 0:
                with open('data.json', 'w') as outfile:
                    json.dump(data, outfile)
                print('Saved {} events'.format(count))

            # if count % 50 == 0:
            #     print('Processed {} events'.format(count))
