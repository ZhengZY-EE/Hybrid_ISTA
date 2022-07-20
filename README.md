# Hybrid ISTA: Unfolding ISTA With Convergence Guarantees Using Free-Form Deep Neural Networks
This repository is for Hybrid ISTA proposed in the following paper:

Ziyang Zheng, Wenrui Dai, Duoduo Xue, Chenglin Li, Junni Zou, Hongkai Xiong, “Hybrid ISTA: Unfolding ISTA With Convergence Guarantees Using Free-Form Deep Neural Networks”, accepted by TPAMI, 2022. (DOI: https://doi.org/10.1109/TPAMI.2022.3172214)

The code is based on the ALISTA repository (https://github.com/VITA-Group/ALISTA) and the Gated LISTA repository (https://github.com/wukailun/GLISTA).

<!-- vim-markdown-toc GFM -->

* [Introduction](#introduction)
* [Running codes](#running-codes)
    * [Sparse Recovery](#sparse-recovery)
        * [Generate problem files](#generate-problem-files)
        * [Train or test models for sparse recovery](#train-or-test-models-for-sparse-recovery)
    * [Compressive Sensing](#compressive-sensing)
        * [Generate training and validation datasets](#generate-training-and-validation-datasets)
        * [Train or test models for compressive sensing](#train-or-test-models-for-compressive-sensing)
* [Cite this work](#cite-this-work)

<!-- vim-markdown-toc -->

## Introduction
It is promising to solve linear inverse problems by unfolding iterative algorithms (e.g., iterative shrinkage thresholding
algorithm (ISTA)) as deep neural networks (DNNs) with learnable parameters. However, existing ISTA-based unfolded algorithms restrict the network architectures for iterative updates with the partial weight coupling structure to guarantee convergence. In this paper, we propose hybrid ISTA to unfold ISTA with both pre-computed and learned parameters by incorporating free-form DNNs (i.e., DNNs with arbitrary feasible and reasonable network architectures), while ensuring theoretical convergence. We first develop HCISTA to improve the efficiency and flexibility of classical ISTA (with pre-computed parameters) without compromising the convergence rate in theory. Furthermore, the DNN-based hybrid algorithm is generalized to popular variants of learned ISTA, dubbed HLISTA, to enable a free architecture of learned parameters with a guarantee of linear convergence. To our best knowledge, this paper is the first to provide a
convergence-provable framework that enables free-form DNNs in ISTA-based unfolded algorithms. This framework is general to endow arbitrary DNNs for solving linear inverse problems with convergence guarantees. Extensive experiments demonstrate that hybrid ISTA can reduce the reconstruction error with an improved convergence rate in the tasks of sparse recovery and compressive sensing.

## Running codes
### Sparse Recovery
#### Generate problem files
To run most of experiments for sparse recovery, one need to first generate an instance of `Problem` class, which you can find in
`utils/prob.py` file.

One can refer to the ALISTA repository (https://github.com/VITA-Group/ALISTA) for more details.

#### Train or test models for sparse recovery

One can train or test the models in Fig. 3, 8 and 9 in our paper (https://doi.org/10.1109/TPAMI.2022.3172214) using the following commands.

To train or test a **HCISTA-0.1** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net Hybrid_ISTA_ada_fixT -T 16 -llam 0.1 \
    [--untied] -wm S -cvn 3 -ks 9 -fm 16 \
    --scope Hybrid_ISTA_ada_fixT --exp_id 0 --w_mode S
```

Explanation for the options (most options are same as ALISTA, and all options are parsed in `config.py`):
* `--task_type`: the task on which you will train/test your model. Possible values are:
  * `sc` standing for normal simulated sparse coding algorithm;
  * `cs` for natural image compressive sensing.
* `-g/--gpu`: the id of GPU used. GPU 0 will be used by default.
* `-t/--test` option indicates training or testing mode. Use this option for testing.
* `-M/--M`: the dimension of measurements.
* `-N/--N`: the dimension of sparse signals.
* `-P/--pnz`: the approximate of non-zero elements in sparse signals.
* `-S/--SNR`: the signal-to-noise ratio in dB unit in the measurements. inf means noiseless setting.
* `-C/--con_num`: the condition number. 0.0 (default) means the condition number will not be changed.
* `-n/--net`: specifies the network to use.
* `-T`: the number of layers.
* `-llam/--lasso_lam`: the weight of l1 norm term \labmda in LASSO.
* `-u/--untied`: whether the inserted DNNs are shared within layers. Use this option for constructing untied model. We adopt tied model (default) for hybrid ISTA models and untied baseline models in the paper.
* `-wm/--w_mode`: works when adopting untied models. Only mode 'S' is executable in HCISTA. See the counterpart in HLISTA models for more details.
* `-cvn/--conv_num`: the number of convolution layers of the inserted DNNs in each iteration. See Equation (58) for more details.
* `-ks/--kernel_size`: kernel size of convolution layers. See Equation (58) for more details.
* `-fm/--feature_map`: number of intermediate feature maps. See Equation (58) for more details.
* `--scope`: the name of variable scope of model variables in TensorFlow.
* `--exp_id`: experiment id, used to differentiate experiments with the same setting.


To train or test a **HCISTA-F** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net Hybrid_ISTA_ada_freeT -T 16 -llam 0.05 \
    [--untied] -wm S -cvn 3 -ks 9 -fm 16 \
    --scope Hybrid_ISTA_ada_freeT --exp_id 0 --w_mode S
```

To train or test a **LISTA-CP-U** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net LISTA_cp -T 16 -l 0.4 \
    --untied --scope LISTA_cp --exp_id 0 
```
Explanation for the new option:
* `-l/--lam`: initial \lambda in LISTA solvers.

To train or test a **HLISTA-CP** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net Hybrid_LISTA_cp -T 16 -l 0.4 \
    [--untied] [-wm D] -cvn 3 -ks 9 -fm 16 \
    --scope Hybrid_LISTA_cp --exp_id 0 
```
Explanation for the new options:
* `-u/--untied`: whether the weights are shared within layers, including \W^hat, \W^bar and DNNs. Use this option for constructing untied model (default 'tied').
* `-wm/--w_mode`: choices: ['D', 'S'] (works when adopting untied models). 'D' means that \W^hat and \W^bar are different in each iteration, while 'S' means the same. DNNs, \theta1, \theta2 and \alpha are always not shared.

To train or test a **LISTA-CPSS-U** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net LISTA_cpss -T 16 -l 0.4 -p 0.7 -maxp 13.0 \
    --untied --scope LISTA_cpss --exp_id 0 
```
Explanation for the new options:
* `-p/--percent`: the percentage of entries to be added to the support in each
  layer.
* `-maxp/--max_percent`: maximum percentage of entries to be selected.

To train or test a **HLISTA-CPSS** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net Hybrid_LISTA_cpss -T 16 -l 0.4 -p 0.7 -maxp 13.0 \
    [--untied] [-wm D] -cvn 3 -ks 9 -fm 16 \
    --scope Hybrid_LISTA_cpss --exp_id 0 
```

To train or test a **ALISTA** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net ALISTA -T 16 -l 0.4 -p 0.0 -maxp 0.0 \
    --scope ALISTA --exp_id 0 -W ./data/W.npy
```
Explanation for the new options:
* `-W`: pretrained weight for ALISTA models. Note that the weight changes when `--SNR` or `--con_num` changes. One can find some pretrained weights in `./data`.

We adopts the standard thresholding operator in ALISTA by default. One can modify `models/ALISTA.py` to utilize the thresholding operator with support selection to make options `-p` and `-maxp` valid.

To train or test a **HALISTA** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net Hybrid_ALISTA -T 16 -l 0.4 -p 0.0 -maxp 0.0 \
    [--untied] -cvn 3 -ks 9 -fm 16 \
    --scope Hybrid_ALISTA --exp_id 0 -W ./data/W.npy
```

To train or test a **Gated LISTA** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net GLISTA -T 16 -l 0.4 -overshoot -both_gate \
    --untied --scope GLISTA --exp_id 0 --better_wait 5000
```
Explanation for the new options:
* `-overshoot`: Use this option to adopt overshoot gate.
* `-both_gate`: Use this option to adopt both gain and overshoot gates in each iteration. 
* `--better_wait`: maximum waiting time for a better validation accuracy before going to the next training stage (default setting is 4000).

One can see more details in `models/GLISTA.py`.

To train or test a **HGLISTA** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net Hybrid_GLISTA -T 16 -l 0.4 -overshoot -both_gate \
    [--untied] [-wm D] -cvn 3 -ks 9 -fm 16 \
    --scope Hybrid_GLISTA --exp_id 0 --better_wait 5000
```

To train or test a **ELISTA** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net ELISTA -T 16 -l 0.4 -mt_flag \
    --scope ELISTA --exp_id 0 --better_wait 5000
```
Explanation for the new options:
* `-mt_flag`: Use this option to use multistage-thresholding operator.

Note that untied ELISTA model is not consistent with the theory in the paper (https://ojs.aaai.org/index.php/AAAI/article/view/17032/16839). One can construct untied ELISTA model in this code, but we do not recommend it.

To train or test a **HELISTA** model, use the following command:
```
python main.py --task_type sc -g 0 [-t] \
    --M 250 --N 500 --pnz 0.1 --SNR inf --con_num 0 \
    --net Hybrid_ELISTA -T 16 -l 0.4 -mt_flag \
    [--untied] [-wm S] -cvn 3 -ks 9 -fm 16 \
    --scope Hybrid_ELISTA --exp_id 0 --better_wait 5000
```

Some sparse recovery models with postfix `ComplexNet` in `models` such as `models\Hybrid_ISTA_ada_fixT_ComplexNet.py` and `models\Hybrid_LISTA_cp_ComplexNet.py` adopt complicated networks including DenseNet, U-Net, Vision Transformer and fully-connected networks. One can refer to Appendix D.1.2 for more details.

Some sparse recovery models with postfix `FixNetFunc` in `models` is used to evaluate the performance when DNN function is fixed, e.g., the output of DNNs is fixed to zero. One can refer to Appendix C.2 and Appendix D.1.1 for more details.

Sparse recovery model `models\Hybrid_ISTA_ada_fixT_assump1.py` is utilized to further clarify Assumption 1. One can refer to Appendix D.1.3 for more details.


### Compressive Sensing
#### Generate training and validation datasets

To run experiments for compressive sensing, one need to 
1. Download BSD500 dataset. Split into train, validation and test sets as you wish.
2. Prepare sensing matrices with various sampling ratios.
3. Generate training and validation datasets using `utils/data.py` file.

#### Train or test models for compressive sensing

Use the models LISTA-CP-U, HLISTA-CP and HALISTA to basicly explain how to train and test models for compressive sensing with sampling ratio 0.5. 

To train or test a **LISTA-CP-U** model, use the following command:
```
python main.py --task_type cs -g 0 [-t] \
    --F 256 --N 512 --net LISTA_cp_cs -T 16 \
    --untied --scope LISTA_cp --exp_id 0 \
    --M 128 -se ./data/0_50_mm.npy \
    -tf ./data/50_train_set.tfrecords \
    -vf ./data/50_val_set.tfrecords
```
Explanation for the new options:
* `-F/--F`: number of features of extracted patches. For example, 'F=256' means that the size of extracted image patches is 16*16. Image patches in training and validation datasets are generated using `utils/data.py` file.
* `-M/--M`: dimension of measurements. The value is equal to 'F' * sampling_ratio. For example, 'M=128' and 'F=256' mean that the sampling ratio is 0.5.
* `-se/--sensing`: sensing matrix file. 
* `-tf/--train_file`: file name of tfrecords file of training data for compressive sensing experiments.
* `-vf/--val_file`: file name of tfrecords file of validation data for compressive sensing experiments.

To train or test a **HLISTA-CP** model, use the following command:
```
python main.py --task_type cs -g 0 [-t] \
    --F 256 --N 512 --net Hybrid_LISTA_cp_cs -T 16 \
    --scope Hybrid_LISTA_cp_cs --exp_id 0 \
    --M 128 -se ./data/0_50_mm.npy \
    [--untied] [-wm D] -cvn 3 -ks 9 -fm 16 \
    -tf ./data/50_train_set.tfrecords \
    -vf ./data/50_val_set.tfrecords
```

To train or test a **HALISTA** model, use the following command:
```
python main.py --task_type cs -g 0 [-t] \
    --F 256 --N 512 --net Hybrid_ALISTA_cs -T 16 \
    --scope Hybrid_ALISTA_cs --exp_id 0 \
    --M 128 -se ./data/0_50_mm.npy \
    [--untied] [-wm D] -cvn 3 -ks 9 -fm 16 \
    -tf ./data/50_train_set.tfrecords \
    -vf ./data/50_val_set.tfrecords -W ./data/W_cs50.npy
```

Some compressive sensing models with postfix `ComplexNet` in `models` such as `models\Hybrid_ISTA_ada_fixT_cs_ComplexNet.py` and `models\Hybrid_LISTA_cp_ComplexNet_cs.py` adopt complicated networks. One can refer to Appendix D.2.2 for more details.

Compressive sensing models `models\Hybrid_ISTA_ada_fixT_cs_ISTANet_Plus.py` and `models\Hybrid_LISTA_cp_cs_ISTANet_Plus.py` are utilized to compare with `models\ISTA_Net_Plus.py` using the same training strategy and datasets. One can refer to Section 6.2.2 for more details.

## Cite this work
If you find our code helpful in your research or work, please cite our paper.
```
@article{zheng2022hybrid,
title={Hybrid ISTA: Unfolding ISTA With Convergence Guarantees Using Free-Form Deep Neural Networks},
author={Zheng, Ziyang and Dai, Wenrui and Xue, Duoduo and Li, Chenglin and Zou, Junni and Xiong, Hongkai},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
year={2022},
publisher={IEEE}
}
```
