# MEC_RL_UAV
1. 创建环境：
conda create —prefix=RL python=3.7 
或者
conda create --prefix RL python=3.7
conda create --name=RL  python=3.7

2. 进入环境：
conda activate RL

3. 下载依赖：
conda install click cudnn=7.6.5 cudatoolkit=10.2.89 hypothesis=5.36.0 keras=2.3.1 matplotlib numpy=1.16.0 pandas=1.1.1 pillow=7.2.0

conda intall tensorflow-gpu==2.4.1 或 conda install tensorflow==2.4.1

conda install gym==0.21.0
conda install imageio==2.22.4
conda install ipython==7.34.0
conda install tqdm==4.64.0