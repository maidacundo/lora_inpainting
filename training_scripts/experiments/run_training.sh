

pip install -r requirements.txt

python3 train.py --lora-injection self-attention --dataset kvist_windows 

python3 train.py --lora-injection cross-attention --dataset kvist_windows 

python3 train.py --lora-injection geglu --dataset kvist_windows 

python3 train.py --lora-injection resnet-block --dataset kvist_windows 

python3 train.py --lora-injection text-encoder --dataset kvist_windows --pretrained-model kvist_windows_cross-attention