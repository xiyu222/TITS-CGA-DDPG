#%%

# Runs the algorthim with default parameters
# from NashRL import run_Nash_Agent

from NashRL import *
import time
start = time.time()
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'

# num_sim: number of simulations to run for
# AN_file_name: file name to output action network parameters to (defult is "Action_Net")
# VN_file_name: file name to output value network parameters to (default is "Value_Net")

# 开始训练
run_Nash_Agent(MAX_EPISOIDE=1500 ,MAX_TIME_STEP=15, AN_file_name = "Action_Net")
end = time.time()
print(end-start)