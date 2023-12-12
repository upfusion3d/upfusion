# Create a new conda environment.
conda create -n upfusion -c conda-forge python=3.10 -y
conda activate upfusion

# Install some pre-requisite packages that may be necessary for installing some downstream packages.
conda install -c conda-forge ninja cxx-compiler=1.3.0 -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y

# Install pytorch.
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install several packages through requirements.txt
pip install -r requirements.txt

# Build some final packages.
pip install ./external/gridencoder
pip install ./external/raymarching
