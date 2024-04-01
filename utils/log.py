import os
from datetime import datetime
import logging
import sys

class Logger(object):
    def __init__(self, log_save_dir) -> None:
        # ensure the log directory existed
        os.makedirs(log_save_dir, exist_ok=True)
        # make log file
        self.log_file_name = os.path.join(log_save_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")

    def set_logger(self,level=logging.DEBUG): #logging.DEBUG级别以上的会打印出来
        """
        Method to return a custom logger with the given name and level
        """
        logger = logging.getLogger(self.log_file_name)
        logger.setLevel(level)
        # format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
        #                 "%(lineno)d — %(message)s")
        format_string = "%(message)s"
        log_format = logging.Formatter(format_string)
        # Creating and adding the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
        # Creating and adding the file handler
        file_handler = logging.FileHandler(self.log_file_name, mode='a')
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
        return logger
    
    def display(self, logger):
        logger.debug("=" * 45)
        logger.debug("")
        logger.debug(f"Start the logger! The log file's name is: {self.log_file_name}")
        