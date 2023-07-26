import logging
import os
import time


def basic_logging(log_path=None) -> str:
    begin_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
    if log_path is not None:
        log_path = os.path.join(log_path, begin_time)
    else:
        log_path = os.path.join('./log', begin_time)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=os.path.join(log_path, 'log.log'),
                        filemode='w',
                        format='%(asctime)s %(levelname)s %(message)s',
                        level=logging.INFO)

    return log_path
