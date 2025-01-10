import logging
from flask import g

class AppLogger(logging.Logger):
    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=None, stacklevel=1):
        rid = g.get("rid", "")
        msg = f"rId:{rid} | {msg}"
        super(AppLogger, self)._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel
        )

def get_logger():
    logger = AppLogger(__name__)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger

logger = get_logger()