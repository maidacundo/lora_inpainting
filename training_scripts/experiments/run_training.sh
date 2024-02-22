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
python3 test_different_base_model.py --dataset kvist_windows --base-model base # fatto
python3 test_different_base_model.py --dataset sommerhus --base-model base # fatto
python3 test_different_base_model.py --dataset 7er_stol --base-model base # fatto

python3 test_different_base_model.py --dataset kvist_windows --base-model realistic # fatto
python3 test_different_base_model.py --dataset sommerhus --base-model realistic # fatto
python3 test_different_base_model.py --dataset 7er_stol --base-model realistic # fatto




# TIMESTEPS TESTING
python3 test_timesteps.py --dataset kvist_windows --upper-bound 1000 --lower-bound 800 # fatto
python3 test_timesteps.py --dataset kvist_windows --upper-bound 800 --lower-bound 600 # fatto
python3 test_timesteps.py --dataset kvist_windows --upper-bound 600 --lower-bound 400 # fatto
python3 test_timesteps.py --dataset kvist_windows --upper-bound 400 --lower-bound 200 # fatto
python3 test_timesteps.py --dataset kvist_windows --upper-bound 200 --lower-bound 1 # fatto

python3 test_timesteps.py --dataset kvist_windows --upper-bound 1000 --lower-bound 400 
python3 test_timesteps.py --dataset kvist_windows --upper-bound 1000 --lower-bound 1 # fatto

# INJECTION TESTING

python3 test_injection.py --dataset kvist_windows --injection self-attention 
python3 test_injection.py --dataset kvist_windows --injection cross-attention
python3 test_injection.py --dataset kvist_windows --injection geglu
python3 test_injection.py --dataset kvist_windows --injection resnet
python3 test_injection.py --dataset kvist_windows --injection text-encoder

python3 test_injection.py --dataset sommerhus --injection self-attention
python3 test_injection.py --dataset sommerhus --injection cross-attention
python3 test_injection.py --dataset sommerhus --injection geglu
python3 test_injection.py --dataset sommerhus --injection resnet
python3 test_injection.py --dataset sommerhus --injection text-encoder

python3 test_injection.py --dataset 7er_stol --injection self-attention
python3 test_injection.py --dataset 7er_stol --injection cross-attention
python3 test_injection.py --dataset 7er_stol --injection geglu
python3 test_injection.py --dataset 7er_stol --injection resnet
python3 test_injection.py --dataset 7er_stol --injection text-encoder

# EMBEDDING TESTING
- kvist # fatto
- sommerhus # fatto
- 7er_stol # da fare (?)


