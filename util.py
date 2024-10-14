from datetime import datetime
import logging

now = datetime.now()
now = now.strftime("%Y-%m-%d")

logging.basicConfig(
    filename=f"logs/{now}.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True,
    level=logging.INFO
)

def generate_run_id():

    logging.info('Generating run id')
    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    return now

def format_float(x):
    return "{:.2f}".format(x)