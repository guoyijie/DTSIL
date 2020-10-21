import os
import subprocess as sp

seeds = [0,1,2,3,4]

jobs = []
for seed in seeds:
    log = '%s_s%s' % ('map_random_initial', seed)
    jobs.append({'log': log, 'seed':seed})


for job in jobs:
    print(job)

log_dir = 'result/ppo_diverse'

sp.call(['mkdir', '-p', log_dir]) 
for job in jobs:
    path = os.path.join(log_dir, job['log'])
    if not os.path.exists(path):
        sp.call(['mkdir', path]) 
        print("Starting: ", job)
        sp.call(['python3', 'run_ppo_diverse.py',
            '--seed', str(job['seed']),
            '--log', str(log_dir+'/'+job['log'])])

