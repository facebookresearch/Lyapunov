PyTorch original implementation of "Global Lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers" (NeurIPS 2024).

**Requirements**
 - Requirements are contained in Lyapunov.yml, you can setup a conda env with
```
conda env create -f Lyapunov_python_3_10.yml
```
or, if preferred or if you have conflict with the Lyapunov_python_3_10 you can use,
```
conda env create -f Lyapunov_python_3_9.yml
```
and then run
```
conda activate Lyapunov
```
Typically, we use Lyapunov_python_3_9 locally (with conda 4.12.0 and pip 24.3.1) and Lyapunov_python_3_10 on our clusters for compatibility issues. The only differences are in the version of python, dreal and torch. These environments were designed for Linux and Mac. Windows users may encounter some difficulties as dReal as a package might not be supported.

**Benchmarks**
- We provide 4 files in the `/benchmarks` folder. Those benchmarks represent the cited BPoly, FBarr, FLyap and FSOSTOOL.

**Generation**
- We provide three different json files and the `train.py` file to generate backward polynomial, backward non-polynomial and forward polynomial datasets. Please amend the `dump_path` field. 
- When generating the forward polynomial dataset, please allocate a good amount of CPUs to speed up the generation. Also, if you want to solve the barrier problem, change the flag `lyap_proper_fwd` to `False`.
- On data.prefix you will find the generated dataset. The storage location will also appear in the .stderr file after "The experiment will be stored in".

**Dataset creation**
- First generate examples with by using `train.py` with `--export_data true`, for instance (you can specify more flags to tailor the generation)
```
python train.py --dump_path /Your/Path/to/storage/ --export_data true --cpu true --reload_data '' --env_base_seed -1  --num_workers 20
```

- After generating enough samples, you should run the `create_dataset.py` python script to clean up and join multiple datasets or clean up a single dataset. This is a required step before the training stage. Amend the path to your location in the file `create_dataset.py` and then run
```
python create_dataset.py
```
- The script will create 3 important files ending with .train, .cleaned.valid and .cleaned.test.

**Train**
- We provide a json file and the `train.py` file to launch training. On the `reload_data` string, please provide task name (always ode_lyapunov), training dataset, evaluation dataset and different benchmarks (for example `"ode_lyapunov,/path/to/your/dataset.train,/path/to/your/dataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL"`).

- You can use the flags to tailor the generation / training. For instance
```
python train.py

--dump_path "/your/dump/path" #Export path for dataset and models
--n_enc_layers 6 # Number of layers encoders
--n_dec_layers 6 # Number of layers decoders
--emb_dim 640 # Embedding dimension
--n_heads 10 # Number of attention head
--batch_size 4 # Batch size
--batch_size_eval 16 # Batch size at evaluation

--max_src_len 0
--max_len 1024 # Max lengths of the sequences
--max_output_len 512 # Max lengths of the output sequences
--optimizer "adam_inverse_sqrt,warmup_updates=10000,lr=0.0001"
--epoch_size 300000
--max_epoch 100000
--num_workers 1
--export_data false # Set to true for generation, false for training

--eval_size 200
--eval_verbose 0
--beam_eval true

--lyap_polynomial_H true #
--lyap_basic_functions_num true
--lyap_pure_polynomial true # Only polynomial systems
--lyap_SOS_checker true # Use a sum-of-squares checker for the evaluation
--lyap_SOS_fwd_gen false # Set to true when generating examples with a forward distribution using a sum-of-squares checker

--stopping_criterion "valid_ode_lyapunov_beam_acc,100"
--validation_metrics "valid_ode_lyapunov_beam_acc"

--reload_size -1
--reload_data "ode_lyapunov,/path/to/your/dataset.train,/path/to/youdataset.valid.final,benchmarks/BPoly,benchmarks/FBarr,benchmarks/FLyap,benchmarks/FSOSTOOL" #Task followed by train dataset, valid dataset and test datasets
```
- The full list of flags for the environement and more details about what they represent can be found in the `register_args`method of `ode.py`. The full list of flag for the model architecture and training parameters can be found in the method `get_parser` of `train.py`

**Reference**  
This code is released under a Creative Commons License, see LICENCE file for more details. If you use this code, consider citing

```    
@article{alfarano2024global,  
  title={Global Lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers},  
  author={Alfarano, Alberto and Charton, Fran{\c{c}}ois and Hayat, Amaury},  
  journal={arXiv preprint arXiv:2410.08304},  
  year={2024}  
}
```
