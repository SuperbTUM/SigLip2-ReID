apt-get update -y
apt install python3.12-venv -y
python3.12 -m venv venv
source venv/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install peft transformers gdown num2words
gdown 0B8-rUzbwVRk0c054eEozWG9COHM
unzip -q Market-1501-v15.09.15.zip
gdown 0B0o1ZxGs_oVZWmtFdXpqTGl3WUU
unzip -q VeRi.zip
wget https://raw.githubusercontent.com/Syliz517/CLIP-ReID/refs/heads/master/datasets/keypoint_train.txt
wget https://raw.githubusercontent.com/Syliz517/CLIP-ReID/refs/heads/master/datasets/keypoint_test.txt
python3 ./LoRA_vision_tuning.py