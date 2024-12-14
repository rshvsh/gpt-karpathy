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
