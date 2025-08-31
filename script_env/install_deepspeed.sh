export CMAKE_ARGS="-DCMAKE_SYSTEM_PROCESSOR=x86_64"
git clone https://github.com/deepspeedai/DeepSpeed-Kernels.git
cd DeepSpeed-Kernels
CUDA_ARCH_LIST="80;86;89;90" python -m build --wheel

#ホイールを探してインストールpython
find -name "deepspeed_kernels*.whl"
pip install #deepspeed_kernelsのパス

cd ..

DS_BUILD_OPS=1 DS_BUILD_AIO=0 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_FP_QUANTIZER=0 DS_BUILD_SPARSE_ATTN=0 pip install -v deepspeed==0.16.9

ds_report