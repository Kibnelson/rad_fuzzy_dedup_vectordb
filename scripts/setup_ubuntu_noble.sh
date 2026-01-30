
#!/bin/bash

sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev build-essential libffi-dev libssl-dev

sudo apt update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 10
sudo update-alternatives --set python /usr/bin/python3.10




sudo apt-get update
sudo apt-get install -y cmake build-essential
sudo apt-get install -y  libblas-dev liblapack-dev
sudo apt-get install -y build-essential cmake ninja-build libopenblas-dev libomp-dev
sudo apt-get install -y libatlas-base-dev libatlas3-base
sudo apt-get install -y clang-8
sudo apt-get install -y swig
sudo apt-get install -y gflags
sudo apt-get install -y libblas-dev liblapack-dev
sudo apt-get install -y build-essential cmake ninja-build libopenblas-dev libomp-dev
sudo apt-get install -y libgflags-dev
sudo apt-get install -y zip unzip
sudo apt install -y screen

curl -L "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
