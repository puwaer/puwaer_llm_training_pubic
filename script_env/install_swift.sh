conda create -n swift_env python==3.10
conda activate swift_env

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade pip
pip install vllm==0.8.5.post1 triton==3.2.0 bitsandbytes==0.45.5 wandb math_verify
pip install deepspeed==0.16.9
pip install flash-attn==2.7.4.post1 -v

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
git checkout v3.6.0
pip install -e . -v