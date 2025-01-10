import logging

class RequestIdAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        rid = self.extra.get("rid", "")
        return f"rId:{rid} | {msg}", kwargs

def get_logger():
    logger = logging.Logger(__name__)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = get_logger()