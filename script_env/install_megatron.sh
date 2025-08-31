pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable -v
#pip install --no-build-isolation transformer_engine[pytorch]==2.2 -v

export MAX_JOBS=16
git clone https://github.com/NVIDIA/apex
cd apex
git checkout e13873debc4699d39c6861074b9a3b2a02327f92
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.12.0

export MAX_JOBS=4
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 27f501d
cd hopper
python setup.py install -v
python_path=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p $python_path/flash_attn_3
wget -P $python_path/flash_attn_3 https://raw.githubusercontent.com/Dao-AILab/flash-attention/27f501dbe011f4371bff938fe7e09311ab3002fa/hopper/flash_attn_interface.py
python -c "import flash_attn_3; print(flash_attn_3.__version__)"