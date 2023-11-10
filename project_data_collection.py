import json
import logging
import os

from sseclient import SSEClient as EventSource
import datetime
# remove buffered output
os.environ["PYTHONUNBUFFERED"] = "1"
EVENTS = 10e6  # 1 million
SAVE_FREQ = 50000
URL = 'https://stream.wikimedia.org/v2/stream/recentchange'

logger = logging.getLogger("")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
if __name__ == '__main__':
    # print('Starting data collection')
    logger.addHandler(ch)
    logger.info('Starting data collection')
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
                with open(f'data_{count}.json', 'w') as outfile:
                    json.dump(data, outfile)
                    data = []
                logger.info('Time: {}, saved {} events'.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count))

            # if count % 50 == 0:
            #     print('Processed {} events'.format(count))
