import subprocess as sp
import time
import os
import signal

proc = sp.Popen(['python3','main_id.py'], 
                    stdout=sp.PIPE,
                    stderr=sp.PIPE)
start_time = time.time()
while True:
    status = proc.poll()
    if status is not None:
        proc = sp.Popen(['python3','main_id.py'], 
                    stdout=sp.PIPE,
                    stderr=sp.PIPE)
    if (time.time() - start_time ) > 60:
        print("Killing process")
        os.kill(proc.pid, signal.SIGTERM)
        proc = sp.Popen(['python3','main_id.py'], 
                    stdout=sp.PIPE,
                    stderr=sp.PIPE)
        start_time = time.time()

    
