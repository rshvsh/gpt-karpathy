#!/usr/bin/env python3
import os
import sys
import subprocess
import venv

def install_packages():
    # update apt
    command = "sudo apt update"
    result = subprocess.run(command, capture_output=True, text=True)
    print("stdout:", result.stdout)
    if result.stderr:
        print("stderr:", result.stderr)

    # List of packages to install
    packages = [
        "make", "build-essential", "libssl-dev", "zlib1g-dev",
        "libbz2-dev", "libreadline-dev", "libsqlite3-dev", "wget",
        "curl", "llvm", "libncurses5-dev", "libncursesw5-dev", "xz-utils",
        "tk-dev", "libffi-dev", "liblzma-dev", "python3-openssl", "git"
    ]

    # Form the command
    command = ["sudo", "apt", "install", "-y"] + packages
    
    try:
        print("Installing dependencies...")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        # Print output for debugging or logging
        print("Installation Output:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print("An error occurred while installing packages:")
        print(e.stderr)

def main():
    # Define the virtual environment directory
    venv_dir = ".venv"
    
    # Check if we are already running inside the virtual environment
    if os.getenv("VIRTUAL_ENV"):
        print("Already inside the virtual environment!")
    else:
        print("Please rerun this script inside the virtual environment!")


    # install the necessary packages


if __name__ == "__main__":
    main()
