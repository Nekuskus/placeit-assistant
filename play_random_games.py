import os
import sys
import subprocess
from random import randint
import pdb


returncode = subprocess.call(['cargo', 'build', '--release'])

if returncode != 0:
    print(f"Build call exited with code {returncode}, aborting.")
    exit(returncode)

if os.name != 'nt':
    print(f"Only implemented for Windows, current os is {os.name}.")

p = subprocess.Popen(["target\\release\\placeit-assistant.exe"], encoding='utf-8', stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.STDOUT)

n = 100
if len(sys.argv) > 0 and sys.argv[1].isdigit():
    n = int(sys.argv[1])

def flush(p):
    p.stdin.flush()
    p.stdout.flush()

for i in range(n):
    if p.poll() is None:
        already_used = [-1]
        game_ended = False
        for _ in range(20):
            if game_ended:
                break
            num = -1
            while num in already_used:
                num = randint(1, 999)
            already_used.append(num)
            #print(f"Guessing {num}")
            flush(p)
            p.stdin.write(f'F {num}\n')
            ended_iter = False
            while not ended_iter:
                flush(p)
                line = p.stdout.readline()
                #print(line)
                if "win" in line or "game over" in line:
                    while not ended_iter:
                        line = p.stdout.read(1)
                        if "$" in line:
                            flush(p)
                            p.stdin.write(f'n\n')
                            game_ended = True
                            ended_iter = True
                            break
                    ended_iter = True
                elif "Placed" in line:
                    break
        if not game_ended:
            p.stdin.write(f'n\n') # just for sanity
        # print(f"Finished game {i}")

exit(p.communicate('q\n'))