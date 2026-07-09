import paramiko
import warnings
import os
import tarfile
import io
import time

warnings.filterwarnings('ignore')

HOST = '39.106.45.168'
USER = 'root'
PASS = '3Pxs@g4VvZwdF3G'
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_cmd(ssh, cmd, timeout=600):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err and 'debconf' not in err.lower() and 'warning' not in err.lower():
        print(f'  > {err[:400]}')
    return out

def step(msg):
    print(f'\n=== {msg} ===')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=30)

# Re-transfer updated files (Dockerfile, etc.)
step('Transferring updated files')
exclude_dirs = {'node_modules', '.next', '.git', 'data', '__pycache__', '.opencode'}
exclude_exts = {'.db', '.pyc'}

tar_buffer = io.BytesIO()
with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if any(file.endswith(ext) for ext in exclude_exts):
                continue
            if file in ('dev-output.txt', 'deploy.py', 'deploy_fix.py'):
                continue
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, PROJECT_DIR)
            tar.add(filepath, arcname=arcname)

tar_buffer.seek(0)
sftp = ssh.open_sftp()
sftp.putfo(tar_buffer, '/opt/portfolio/project.tar.gz')
sftp.close()
print(f'Archive: {len(tar_buffer.getvalue())/1024:.0f} KB')

run_cmd(ssh, 'cd /opt/portfolio && tar xzf project.tar.gz && rm project.tar.gz')

# Stop existing containers
step('Stopping existing containers')
run_cmd(ssh, 'cd /opt/portfolio && docker compose down 2>&1 || true')

# Build
step('Building Docker image')
result = run_cmd(ssh, 'cd /opt/portfolio && docker compose build --no-cache 2>&1 | tail -30', timeout=600)
if 'failed' in result.lower() or 'error' in result.lower():
    print('BUILD FAILED - showing full logs:')
    run_cmd(ssh, 'cd /opt/portfolio && docker compose build 2>&1 | tail -50', timeout=600)

# Start
step('Starting containers')
run_cmd(ssh, 'cd /opt/portfolio && docker compose up -d 2>&1')
time.sleep(8)
run_cmd(ssh, 'cd /opt/portfolio && docker compose ps')

# Verify
step('Verifying')
run_cmd(ssh, 'curl -s -o /dev/null -w "Homepage EN: HTTP %{http_code}" http://localhost:3000/en 2>&1')
run_cmd(ssh, 'curl -s -o /dev/null -w "Homepage ZH: HTTP %{http_code}" http://localhost:3000/zh 2>&1')
run_cmd(ssh, 'curl -s -o /dev/null -w "Admin:       HTTP %{http_code}" http://localhost:3000/admin 2>&1')
run_cmd(ssh, 'curl -s -o /dev/null -w "Search API:  HTTP %{http_code}" "http://localhost:3000/api/search?q=test" 2>&1')
run_cmd(ssh, 'curl -s -o /dev/null -w "Via Nginx:   HTTP %{http_code}" http://localhost/en 2>&1')

# Logs
step('Container logs')
run_cmd(ssh, 'cd /opt/portfolio && docker compose logs --tail=10 web 2>&1')

ssh.close()
print('\n=== DONE ===')
print(f'Website: http://{HOST}')
