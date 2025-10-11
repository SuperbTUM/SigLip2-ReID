python3 -v venv venv
source venv/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip3 install peft transformers gdown
gdown 0B8-rUzbwVRk0c054eEozWG9COHM
unzip -q Market-1501-v15.09.15.zip
python3 ./locked_image_tuning.py