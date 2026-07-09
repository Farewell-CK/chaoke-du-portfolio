import paramiko, warnings
warnings.filterwarnings('ignore')
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('39.106.45.168', username='root', password='3Pxs@g4VvZwdF3G', timeout=30)

cmds = [
    'cd /opt/portfolio && docker compose ps 2>&1',
    'echo "---LOGS---" && cd /opt/portfolio && docker compose logs --tail=3 web 2>&1',
    'echo "---HTTP---" && curl -s -o /dev/null -w "EN: %{http_code}\n" http://localhost:3000/en',
    'curl -s -o /dev/null -w "ZH: %{http_code}\n" http://localhost:3000/zh',
    'curl -s -o /dev/null -w "ADM: %{http_code}\n" http://localhost:3000/admin',
    'curl -s -o /dev/null -w "N80: %{http_code}\n" http://localhost/en',
    'echo "---IMAGES---" && docker images 2>&1',
]

for cmd in cmds:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=30)
    out = stdout.read().decode().strip()
    if out:
        print(out)

ssh.close()
