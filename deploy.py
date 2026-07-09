import paramiko
import warnings
import os
import tarfile
import io

warnings.filterwarnings('ignore')

HOST = '39.106.45.168'
USER = 'root'
PASS = '3Pxs@g4VvZwdF3G'
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_cmd(ssh, cmd, timeout=120):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    if out:
        print(out)
    if err and 'debconf' not in err.lower() and 'warning' not in err.lower():
        print(f'STDERR: {err[:300]}')
    return out

def step(msg):
    print(f'\n=== {msg} ===')

# Connect
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, username=USER, password=PASS, timeout=30)

# Step 1: Install Docker
step('Installing Docker')
run_cmd(ssh, 'apt-get update -qq')
run_cmd(ssh, 'apt-get install -y -qq ca-certificates curl gnupg > /dev/null 2>&1')
run_cmd(ssh, 'install -m 0755 -d /etc/apt/keyrings')
run_cmd(ssh, 'curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc')
run_cmd(ssh, 'chmod a+r /etc/apt/keyrings/docker.asc')

repo_line = 'deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable'
run_cmd(ssh, f'echo "{repo_line}" > /etc/apt/sources.list.d/docker.list')
run_cmd(ssh, 'apt-get update -qq')
run_cmd(ssh, 'apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin > /dev/null 2>&1', timeout=180)
run_cmd(ssh, 'systemctl start docker')
run_cmd(ssh, 'systemctl enable docker')
run_cmd(ssh, 'docker --version')
run_cmd(ssh, 'docker compose version')

# Step 2: Create project directory
step('Setting up project directory')
run_cmd(ssh, 'mkdir -p /opt/portfolio')

# Step 3: Create tar archive and transfer
step('Creating archive')
exclude_dirs = {'node_modules', '.next', '.git', 'data', '__pycache__'}
exclude_files = {'dev-output.txt', 'portfolio.db'}

def should_include(tarinfo):
    parts = tarinfo.name.split('/')
    for part in parts:
        if part in exclude_dirs:
            return None
    if tarinfo.name in exclude_files or tarinfo.name.endswith('.db'):
        return None
    return tarinfo

tar_buffer = io.BytesIO()
with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
    for root, dirs, files in os.walk(PROJECT_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            if file in exclude_files or file.endswith('.db'):
                continue
            filepath = os.path.join(root, file)
            arcname = os.path.relpath(filepath, PROJECT_DIR)
            tar.add(filepath, arcname=arcname)

tar_buffer.seek(0)
print(f'Archive size: {len(tar_buffer.getvalue()) / 1024 / 1024:.1f} MB')

# Step 4: Transfer via SFTP
step('Transferring files')
sftp = ssh.open_sftp()
sftp.putfo(tar_buffer, '/opt/portfolio/project.tar.gz')
print('Transfer complete')
sftp.close()

# Step 5: Extract and build
step('Extracting and building')
run_cmd(ssh, 'cd /opt/portfolio && tar xzf project.tar.gz && rm project.tar.gz')
run_cmd(ssh, 'cd /opt/portfolio && cp .env.example .env')

# Set secure passwords
run_cmd(ssh, "cd /opt/portfolio && sed -i 's/ChangeMe123!/Dck2026!Secure/' .env")
run_cmd(ssh, "cd /opt/portfolio && sed -i 's/your-random-secret-here/$(openssl rand -hex 32)/' .env")
run_cmd(ssh, "cd /opt/portfolio && sed -i 's/your-api-key-here/$(openssl rand -hex 16)/' .env")

step('Building Docker images (this takes a few minutes)')
run_cmd(ssh, 'cd /opt/portfolio && docker compose build --no-cache', timeout=600)

step('Starting containers')
run_cmd(ssh, 'cd /opt/portfolio && docker compose up -d')
run_cmd(ssh, 'cd /opt/portfolio && docker compose ps')

step('Cleaning up')
run_cmd(ssh, 'docker image prune -f > /dev/null 2>&1')

ssh.close()
print('\n=== DEPLOYMENT COMPLETE ===')
print(f'Website: http://{HOST}')
print('Admin: http://{HOST}/admin')
