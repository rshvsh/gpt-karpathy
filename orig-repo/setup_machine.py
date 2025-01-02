#!/usr/bin/env python3
import os
import sys
import subprocess

def run_commands_stream(commands, return_on_err=True):
    for c in commands:
        print(f"Running command: {c}")
        
        # Run the command and stream output
        process = subprocess.Popen(
            c,
            shell=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        try:
            # Stream stdout and stderr in real-time
            while True:
                output = process.stdout.readline()
                error = process.stderr.readline()

                if output:
                    print(output, end="")  # Print stdout line by line
                if error:
                    print(error, end="", file=sys.stderr)  # Print stderr line by line

                # Break loop if process is done and no more output
                if process.poll() is not None and not output and not error:
                    break
            
            # Check return code
            if process.returncode != 0:
                print(f"Command failed with return code {process.returncode}")
                if return_on_err:
                    return False

        except Exception as e:
            print(f"Error while running command: {e}", file=sys.stderr)
            return False

    return True

def install_packages():
    # update apt
    command = ["sudo", "apt-get", "update"]
    result = subprocess.run(command, capture_output=True, text=True)
    print("stdout:", result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    # List of packages to install
    packages = [
        "make", "build-essential", "libssl-dev", "zlib1g-dev",
        "libbz2-dev", "libreadline-dev", "libsqlite3-dev", "wget",
        "curl", "llvm", "libncurses5-dev", "libncursesw5-dev", "xz-utils",
        "tk-dev", "libffi-dev", "liblzma-dev", "python3-openssl", "git", "rclone"
    ]

    # Form the command
    command = ["sudo", "apt-get", "install", "-y"] + packages
    
    try:
        print("Installing dependencies...")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Print output for debugging or logging
        print("Installation Output:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("An error occurred while installing packages:")
        print(e.stderr)
        return False

def setup_pyenv():
    # check if pyenv is already installed
    pyenv_path = os.path.expanduser("~/.pyenv")
    if os.path.exists(pyenv_path):
        print("Looks like pyenv is already installed, continuing with configuration")
        # we can continue with the script
    else:
        # install pyenv
        command = "curl https://pyenv.run | bash"
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(result.stdout)
        if result.stderr:
            print(f"Error installing pyenv:\n {result.stderr}")
            return False

    # check for proper config already being there
    command = "command -v pyenv"
    result = subprocess.run(command, capture_output=True, text=True, shell=True)

    if "PYENV_ROOT" in os.environ and os.environ["PYENV_ROOT"] + "/bin" in os.environ["PATH"] and result.stdout:
        print("pyenv already configued, no need to re-configure")
        return True

    # we need to configure pyenv
    config = """
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
"""
    bashrc_path = os.path.expanduser("~/.bashrc")

    # check if the content already in bashrc, just not sourced
    with open(bashrc_path, "r") as bashrc:
        if config in bashrc.read():
            print("pyenv config is already in ~/.bashrc, please source ~/.bashrc in the shell")
            return False

    with open(bashrc_path, "a") as bashrc:
        bashrc.write("\n##### Added by setup_machine.py script\n")
        bashrc.write(config + "\n")
        print("pyenv config added to ~/.bashrc, please source ~/.bashrc in the shell")
        return False

def setup_venv():
    # check if the virtual env is already activated
    if os.getenv("VIRTUAL_ENV"):
        print("Already inside the virtual environment!\n")
        return True

    venv_name = "myenv"

    commands = [
        "pyenv install 3.11.11", # TODO:~ this is an interactive command - handle later, for now just press return
        "pyenv global 3.11.11",
        f"pyenv virtualenv 3.11.11 {venv_name}",
    ]
    if not run_commands_stream(commands, False): return False
    # now we need to activate virtualenv in the shell
    print(f"Please activate the virtual environment by running: pyenv activate {venv_name}")
    return False

def install_reqmt():
    reqmts_path = os.path.expanduser("../requirements.txt")

    commands = [
        "pip install --upgrade pip",
        f"pip install -r {reqmts_path}"
    ]
    return run_commands_stream(commands)

def check_rclone():
    vars = [
        "RCLONE_S3_SECRET_ACCESS_KEY",
        "RCLONE_S3_ACCESS_KEY_ID",
        "RCLONE_S3_BUCKET_ROOT",
        "RCLONE_S3_REGION"
    ]
    # Check if all variables are in os.environ
    if all(var in os.environ for var in vars):
        print("All required environment variables are set for rclone.")
    else:
        missing = [var for var in vars if var not in os.environ]
        print(f"Missing environment variables: {', '.join(missing)}")
        return False

    config = f"""
[s3-name]
type = s3
provider = AWS
region = {vars[3]}
location_constraint = {vars[3]}
acl = private
bucket_acl = private
"""

    config_file = os.path.expanduser("~/.config/rclone/rclone.conf")
    if os.path.exists(config_file):
        content = ""
        with open(config_file, "r") as f:
            content = f.read()
        if config in content:
            print(f"Config already exists in {config_file}")
        else:
            with open(config_file, "a") as f:
                print(f"Appending to config file {config_file}")
                f.write(config)
    else:
        config_dir = os.path.dirname(config_file)
        os.makedirs(config_dir, exist_ok=True)

        with open(config_file, "w") as f:
            print(f"Writing to config file {config_file}")
            f.write("\n##### Added by setup_machine.py script\n")
            f.write(config)
            f.write("\n")
    return True

def copy_datafiles(): # TODO:~ rclone does not return an error when the path doesn't exist - fix later
    bucket_root = "$RCLONE_S3_BUCKET_ROOT"
    folders = [
        "data_ag_news",
        "edu_fineweb10B"
    ]
    commands = []
    for folder in folders:
        if not os.path.exists(folder):
            commands.append(f"rclone copy s3-name:{bucket_root}/data/{folder} {folder}")
    return run_commands_stream(commands)

def main():
    # TODO:~ for now assume we are running in the orig-repo directory, fix later
    assert(os.path.basename(os.getcwd()) == "orig-repo")
    install_packages() or sys.exit("Error installing packages")
    setup_pyenv() or sys.exit("Error setting up pyenv")
    setup_venv() or sys.exit("Error setting up venv")
    install_reqmt() or sys.exit("Error installing requirements")
    check_rclone() or sys.exit("Error with rclone config")
    copy_datafiles() or sys.exit("Error copying data files")

if __name__ == "__main__":
    main()
