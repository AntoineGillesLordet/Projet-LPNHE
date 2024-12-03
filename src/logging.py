import logging

logger = logging.getLogger("ZTFxUchuu")

def setup_logger(filename='logs.txt'):

    logger = logging.getLogger("ZTFxUchuu")
    
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M")
    )

    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(filename, mode="a")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s: %(message)s", datefmt="%Y/%m/%d %H:%M"
        )
    )

    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
