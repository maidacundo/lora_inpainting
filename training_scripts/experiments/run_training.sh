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


python3 train_lora.py --lora-injection attention-all --dataset sommerhus --criterion mse+ssim



# BASE MODEL TESTING

python3 test_different_base_model.py --dataset kvist_windows --base-model base
python3 test_different_base_model.py --dataset sommerhus --base-model base
python3 test_different_base_model.py --dataset 7er_stol --base-model base
python3 test_different_base_model.py --dataset dinesen --base-model base

python3 test_different_base_model.py --dataset kvist_windows --base-model realistic
python3 test_different_base_model.py --dataset sommerhus --base-model realistic
python3 test_different_base_model.py --dataset 7er_stol --base-model realistic
python3 test_different_base_model.py --dataset dinesen --base-model realistic