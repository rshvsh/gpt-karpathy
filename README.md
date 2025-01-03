# Running on Lambda

# `ssh` to the remote machine

You should create the rsa key `id_rsa_lambda` per the lambda instructions. Start the remote machine and get it's `<ip addr>`

On your local machine edit your `~/.ssh/config` file and edit the section for the new host with the new IP adress:

```bash
Host <ip addr>
  HostName <ip addr>
  IdentityFile ~/.ssh/id_rsa_lambda
  User ubuntu
```

Now you can ssh to the remote machine

```bash
ssh -i ~/.ssh/id_rsa_lambda ubuntu@<ip addr>
```

You should cleanup when you finish and shutdown the machine:
- Edit your `~/.ssh/known_hosts` file and delete the entries with the `<ip addr>`
- There should be three entries, one each with `ssh-ed25519`, `ssh-rsa` and `ecdsa-sha2-nistp256`

# Export the GitHub token and clone the repo

```bash
### Added these variables to your ~/.bashrc file
export GITHUB_USERNAME="<your github username>"
export GITHUB_TOKEN="<your github token>"
export GITHUB_AUTHOR_EMAIL="<your github email>"
GITHUB_COMMITTER_EMAIL="<your github email>"

# now source your ~/.bashrc file
source ~/.bashrc
git config --global user.email $GITHUB_AUTHOR_EMAIL
git config --global user.name $GITHUB_USERNAME

git clone https://github.com/rshvsh/gpt-karpathy.git
```

# Run the setup script

The following sections are now included in a setup script
- Install `pyenv`
- Configure the environment for `pyenv`
  - Setting up the virtual environment
  - Installing pip and the requirements
- Configuring S3 access with `rclone`
- Downloading the datafiles

```bash
cd gpt-karpathy/orig-repo
./setup_machine.py
```

NOTE: 

- For pyenv you have to run the setup script twice more and before the last time run `source ~/.bashrc`. I haven't diagnosed why this is required twice more vs once more, but for now this is idempotent and works :-)
- After this you also are prompted to initialize the virtual enviornment in the shell by running `pyenv activate myenv` in the shell. This is required because you have to do this in the calling shell. Again, you can just run the setup script again - this is by design.
- You will also be prompted to create the necessary RCLONE environment variables if you don't already have them. Do so and rerun the startup script - this is also by design.

## Install `pyenv`

```bash
sudo apt update

sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev \
libffi-dev liblzma-dev python3-openssl git

curl https://pyenv.run | bash
```

## Configure the environment for `pyenv`

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

source ~/.bashrc

pyenv install 3.11.11

pyenv global 3.11.11

pyenv virtualenv 3.11.11 myenv

pyenv activate myenv

pip install --upgrade pip

pip install -r requirements.txt
```

## Configuring S3 access with `rclone`

- Install rclone `sudo apt install rclone`
- Set `RCLONE_S3_ACCESS_KEY_ID` and `RCLONE_S3_SECRET_ACCESS_KEY` to point to your s3 credentials
- Set `RCLONE_S3_REGION=us-west-2` to the s3 region for the bucket
- Set `RCLONE_S3_BUCKET_ROOT` to the s3 bucket root for data files
- Create or edit your `~/.config/rclone/rclone.conf` to access your s3 account by adding the following section

```bash
[s3-name]
type = s3
provider = AWS
region = us-west-2
location_constraint = us-west-2
acl = private
bucket_acl = private
```

Common rclone commands:

```bash
rclone ls s3-name:$RCLONE_S3_BUCKET_ROOT
rclone mkdir s3-name:$RCLONE_S3_BUCKET_ROOT
rclone copy /path/to/local/file.txt s3-name:$RCLONE_S3_BUCKET_ROOT/path/name
rclone sync /local/path s3-name:$RCLONE_S3_BUCKET_ROOT/path/name --create-empty-src-dirs --progress
```

Note: be sure to fully qualify paths when using rclone. For example:

```bash
# this will create localdir s3, otherwise files will be copied to bucket-name directly
rclone copy /path/to/localdir s3-name:bucket-name/localdir

# this will create localdir on the local machine, otherwise files in localdir will be copied to /path/to
rclone copy s3-name:bucket-name/localdir /path/to/localdir 
```

# Configure VS Code

- Use thge Remote-SSH extension to connect to the remote host at `<ip address>`
- Install the python extention from Microsoft (seems like it needs to install something on the server)
- Once the virtual environment is created and activated, Cmd+Shift+P and search for Python: Select Interpreter and choose the venv

# Running `tmux` for background processing

Useful commands:

```bash
tmux new -s session_name
tmux attach -t session_name
tmux list-sessions
tmux kill-session -t session_name

# for detaching
ctrl+b then d while in the tmux session
# OR
tmux detach
```

# Sample commands

Some torchrun samples:

```bash
# ag_news dataset, run on two GPUs and debug the dataloader
torchrun --standalone --nproc_per_node=2 train_gpt2.py --micro-batch-size=16 --val-loss-freq=25 --hellaswag-freq=25 --dataset=data_ag_news --max-steps=1000 --warmup-steps=10 --generate-freq=25 --checkpoint-freq=50e --gen_text_prompt="In breaking news,"
```

Monitor the nvidia gpus:

```bash
watch -n 1 nvidia-smi
```

# Running on Hyperbolic

Before initiating an instance, be sure to setup your account with a public key. This is only needed one time for your account.

## Start and access the instance

Start the instance and get `$HOSTNAME` and `$PORTNUM`

```bash
export $HOSTNAME="hostname.xyz"
export $PORTNUM="portnum"

# visually check the key
ssh-keyscan -p 31183 $HOSTNAME

# add it to known hosts by concatenating
ssh-keyscan -p 31183 HOSTNAME >> ~/.ssh/known_hosts

# verify that the key is in there
ssh-keygen -lf <(ssh-keyscan -p 31183 HOSTNAME)

# login
ssh ubuntu@$HOSTNAME -p $PORTNUM
```

## Download the code on the remote machine

```bash
export GITHUB_TOKEN="github token"

git clone "repo url"

# update apt and the python version
sudo apt update
sudo apt install -y python3
sudo apt install -y python3-pip

# install from requirement.txt file
pip install -r requirements.txt
```

## Run you stuff on the remote machine

```bash
# use big to train the big model, otherwise just leave out big
python3 ./gpt.py big 
```

## Download files onto your local machine

```bash
# for a particular timestamp
export $TIMESTAMP="timestamp"

# list files to make sure
ssh -p $PORTNUM ubuntu@$HOSTNAME "ls /home/ubuntu/gpt-karpathy/output/*-$TIMESTAMP.*"

# scp stuff off the instance
scp -P $PORTNUM ubuntu@$HOSTNAME:/home/ubuntu/gpt-karpathy/output/* ./output
```
