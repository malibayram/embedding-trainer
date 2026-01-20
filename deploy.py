
import pexpect
import sys
import time

HOST = "macmini"
USER = "alibayram" # Guessing user based on local user, or it uses config
PASS = "3434"
REMOTE_DIR = "~/embedding-trainer"

FILES_TO_SEND = [
    "evaluate_tabibench.py",
    "summarize_tabibench.py",
    "mteb-tr",
    "tabibench_results"
]

def run_ssh_cmd(cmd, description):
    print(f"--- {description} ---")
    print(f"CMD: {cmd}")
    child = pexpect.spawn(cmd, timeout=300)
    
    # Handle possible password prompt (case insensitive)
    # Also handle keyboard-interactive which might show "Password:"
    i = child.expect([
        '(?i)password:', 
        'Are you sure you want to continue connecting', 
        pexpect.EOF, 
        pexpect.TIMEOUT
    ])
    
    if i == 1: # Key fingerprint
        child.sendline('yes')
        i = child.expect(['(?i)password:', pexpect.EOF, pexpect.TIMEOUT])
        
    if i == 0: # Password
        child.sendline(PASS)
        # Wait for finish - expect EOF or maybe the prompt return?
        # If running a command like mkdir or rsync, it exits after done.
        # But if running a persistent command in background, we need to handle it.
        # For rsync/mkdir, they exit.
        child.expect(pexpect.EOF)
    elif i == 2: # EOF 
        pass 
    elif i == 3: # Timeout
        print("Timeout waiting for response. Buffer:")
        print(child.before.decode(errors='replace'))
        child.close()
        sys.exit(1)
        
    print(child.before.decode(errors='replace'))
    child.close()
    if child.exitstatus != 0:
        print(f"Command failed with exit status {child.exitstatus}")
        # sys.exit(child.exitstatus) # Don't exit strictly, might be rsync partials

print("Starting Deployment...")

# 1. Create directory
run_ssh_cmd(f"ssh {HOST} 'mkdir -p {REMOTE_DIR}'", "Creating Remote Directory")

# 2. Rsync files
files_str = " ".join(FILES_TO_SEND)
run_ssh_cmd(f"rsync -avz {files_str} {HOST}:{REMOTE_DIR}/", "Syncing Files")

# 3. Run Command
# Use nohup or screen? python script will wait? 
# User said "start this there". I'll run it interactively and let the user see output via this script's proxying, 
# OR just start it. If I use pexpect, I'll see the output.
# But it might be long running. 
# "start this there" -> implies I should kick it off.
# I'll run it directly. If it takes too long, pexpect timeout will hit.
# The user wants to see it run or just start it? "start this there"
# I will try to run it and maybe stream output for a bit or just run it.
# Let's run it simply.
remote_cmd = "python3 evaluate_tabibench.py boun-tabilab/TabiBERT --batch-size 16"
run_ssh_cmd(f"ssh {HOST} 'cd {REMOTE_DIR} && {remote_cmd}'", "Running Evaluation")
