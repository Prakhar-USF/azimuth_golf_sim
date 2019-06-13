import time
import paramiko
from os.path import expanduser
from user_definition import *


def create_or_update_environment(ssh):
    """create or update the environment"""
    stdin, stdout, stderr = ssh.exec_command(
        "conda env create -f " +
        git_repo_name +
        "/environment.yml")

    if (b'already exists' in stderr.read()):
        print('Environment already exists.')
        ssh.exec_command("conda env update -f " + git_repo_name +
                         "/environment.yml -y")
        print("Update environment successfully.")
        ssh.exec_command('source activate MSDS603')


def git_clone_or_pull(ssh):
    """git clone repository to ec2"""
    # Confirm git works
    stdin, stdout, stderr = ssh.exec_command("git --version")

    # Install Git if needed
    if (b"command not found" in stderr.read()):
        print("Installing Git")
        ssh.exec_command("sudo yum -y install git")

        while True:
            print("...")
            time.sleep(10)
            stdin, stdout, stderr = ssh.exec_command("git --version")

            if (b"command not found" in stderr.read()):
                print("...")
                time.sleep(10)
            else:
                print("Finished installing Git")
                break
    
        # Store credential
        ssh.exec_command('git config --global user.name ' + git_user_id)
        ssh.exec_command('git config --global user.email ' + git_user_email)
        ssh.exec_command("git config --global credential.helper store")


    if (b"" is stderr.read()):
        stdin, stdout, stderr = \
            ssh.exec_command("cd " + git_repo_name + ";git pull")
        if stderr.read() == b'':
            print("Pull from Git successfully.")
        else:
            print("Git pull stderr:" + str(stderr.read()))
    else:
        git_clone_command = "git clone https://github.com/" + \
                             git_repo_owner + "/" + git_repo_name + ".git"
        print(git_clone_command)
        stdin, stdout, stderr = ssh.exec_command(git_clone_command)
        if stderr.read() == b'':
            print("Clone from Git successfully.")
        else:
            print("Git clone stderr:" + str(stderr.read()))


def launch_page(ssh):
    """Launch the webpage."""
    # ssh.exec_command(f'cd {git_repo_name}/website; pwd')

    ssh.exec_command('pwd;.conda/envs/MSDS603/bin/gunicorn '
                     '-D -R --threads 4 -b 0.0.0.0:8080 '
                     '--access-logfile server.log '
                     '--chdir /home/ec2-user/product-analytics-group-project-azimuth/website '
                     'app:application')
    stdin, stdout, stderr = ssh.exec_command('ps aux | grep gunicorn')
    print(stdout.read())
    print(f'The webiste is currently running at Port {port}.')


def kill_process(ssh):
    ssh.exec_command("pkill -9 gunicorn")


def main():
    """main function"""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ec2_address,
                username=user,
                key_filename=expanduser("~") + key_file)
    
    git_clone_or_pull(ssh)
    create_or_update_environment(ssh)
    kill_process(ssh)  # clean gunicorn process
    launch_page(ssh)   # launch the website

    # comment out the next line if you want to keep the website running
    # kill_process(ssh)

    ssh.close()


if __name__ == '__main__':
    main()
