

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

python3 test_injection.py --dataset kvist_windows --injection self-attn
python3 test_injection.py --dataset kvist_windows --injection cross-attn
python3 test_injection.py --dataset kvist_windows --injection geglu
python3 test_injection.py --dataset kvist_windows --injection resnet
python3 test_injection.py --dataset kvist_windows --injection text-encoder

python3 test_injection.py --dataset sommerhus --injection self-attn
python3 test_injection.py --dataset sommerhus --injection cross-attn
python3 test_injection.py --dataset sommerhus --injection geglu
python3 test_injection.py --dataset sommerhus --injection resnet
python3 test_injection.py --dataset sommerhus --injection text-encoder

python3 test_injection.py --dataset 7er_stol --injection self-attn
python3 test_injection.py --dataset 7er_stol --injection cross-attn
python3 test_injection.py --dataset 7er_stol --injection geglu
python3 test_injection.py --dataset 7er_stol --injection resnet
python3 test_injection.py --dataset 7er_stol --injection text-encoder

# EMBEDDING TESTING
- kvist # fatto
- sommerhus # fatto
- 7er_stol # da fare (?)

# TESTING LOSSES
python3 test_losses.py --dataset kvist_windows --loss mse
python3 test_losses.py --dataset kvist_windows --loss mse+ssim

python3 test_losses.py --dataset sommerhus --loss mse
python3 test_losses.py --dataset sommerhus --loss mse+ssim

python3 test_losses.py --dataset 7er_stol --loss mse
python3 test_losses.py --dataset 7er_stol --loss mse+ssim

# FINAL TESTING
python3 test_final_configuration.py --dataset kvist_windows --injection unet-all
python3 test_final_configuration.py --dataset sommerhus --injection unet-all
python3 test_final_configuration.py --dataset 7er_stol --injection unet-all
python3 test_final_configuration.py --dataset dinesen --injection unet-all

python3 test_final_configuration.py --dataset kvist_windows --injection small-all
python3 test_final_configuration.py --dataset sommerhus --injection small-all
python3 test_final_configuration.py --dataset 7er_stol --injection small-all
python3 test_final_configuration.py --dataset dinesen --injection small-all

python3 test_final_configuration.py --dataset kvist_windows --injection all
python3 test_final_configuration.py --dataset sommerhus --injection all
python3 test_final_configuration.py --dataset 7er_stol --injection all
python3 test_final_configuration.py --dataset dinesen --injection all
