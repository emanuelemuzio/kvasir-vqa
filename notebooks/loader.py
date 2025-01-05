import sys
sys.path.append('../src')
sys.path.append('../notebooks')

from datetime import datetime
from feature_extractor.data import prepare_data as prepare_data_func
from common.util import format_float as format_float_func
    
now = datetime.now()
now = now.strftime("%Y-%m-%d")
    
f = open(f"notebooks/logs/{now}", "w")
f.close()
    
prepare_data = prepare_data_func
format_float = format_float_func