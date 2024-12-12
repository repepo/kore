import logging


def configure():
    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.INFO,
    )
