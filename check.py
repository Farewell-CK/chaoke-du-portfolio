import paramiko, warnings
warnings.filterwarnings('ignore')

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('39.106.45.168', username='root', password='3Pxs@g4VvZwdF3G', timeout=30)

cmds = [
    'cd /opt/portfolio && docker compose ps 2>&1',
    'cd /opt/portfolio && docker compose logs --tail=5 web 2>&1',
    'curl -s -o /dev/null -w "EN:  %{http_code}" http://localhost:3000/en 2>&1',
    'curl -s -o /dev/null -w "ZH:  %{http_code}" http://localhost:3000/zh 2>&1',
    'curl -s -o /dev/null -w "ADM: %{http_code}" http://localhost:3000/admin 2>&1',
    'curl -s -o /dev/null -w "N80: %{http_code}" http://localhost/en 2>&1',
    'docker images 2>&1 | head -5',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    out = stdout.read().decode().strip()
    err = stderr.read().decode().strip()
    print(f'> {cmd[:60]}')
    if out:
        print(out)
    if err:
        print(f'  ERR: {err[:200]}')
    print()

ssh.close()
