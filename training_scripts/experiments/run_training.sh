pip install -r requirements.txt
roboflow login
wandb login

python3 train_lora.py --lora-injection self-attention --dataset sommerhus 

python3 train_lora.py --lora-injection cross-attention --dataset sommerhus 

python3 train_lora.py --lora-injection geglu --dataset sommerhus 

python3 train_lora.py --lora-injection resnet-block --dataset sommerhus 

python3 train_lora.py --lora-injection text-encoder --dataset sommerhus

python3 train_textual_inversion.py --dataset sommerhus
python3 train_textual_inversion.py --dataset kvist_windows
