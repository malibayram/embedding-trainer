
import pexpect
import sys
import os

HOST = "macmini"
USER = "alibayram"
PASS = "3434"
REMOTE_DIR = "~/embedding-trainer/tabibench_results/"
LOCAL_DIR = "tabibench_results/"

def run_rsync_fetch():
    # Sync from REMOTE to LOCAL
    cmd = f"rsync -avz {HOST}:{REMOTE_DIR} {LOCAL_DIR}"
    print(f"CMD: {cmd}")
    
    child = pexpect.spawn(cmd, timeout=300)
    
    # Handle possible password prompt (case insensitive)
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

if __name__ == "__main__":
    print("Fetching results from remote...")
    run_rsync_fetch()
