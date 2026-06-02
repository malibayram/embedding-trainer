
import pexpect
import sys

HOST = "macmini"
USER = "alibayram"
PASS = "3434"

def run_ssh_cmd(cmd):
    child = pexpect.spawn(cmd, timeout=30)
    i = child.expect(['(?i)password:', 'Are you sure you want to continue connecting', pexpect.EOF, pexpect.TIMEOUT])
    
    if i == 1:
        child.sendline('yes')
        i = child.expect(['(?i)password:', pexpect.EOF, pexpect.TIMEOUT])
        
    if i == 0:
        child.sendline(PASS)
        child.expect(pexpect.EOF)
    elif i == 3:
        print("Timeout connecting")
        child.close()
        return

    print(child.before.decode(errors='replace'))
    child.close()

print("Checking remote processes...")
run_ssh_cmd(f"ssh {HOST} 'ps aux | grep evaluate_tabibench | grep -v grep'")
