import logging


def logger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)s - %(levelname)s:%(lineno)s - %(message)s')
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log