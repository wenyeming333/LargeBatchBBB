# LargeBatchBBB
The flip_bbb.py file has a tabular_logger dependency which uses python3. To use python2, just comment out everything that depends on it.
To run the large batch BBB, 

```
python flip_bbb.py --data_dir [DATA_DIR] --batch_size [BATCHSIZE] --learning_rate [LR]
--num_iterations [training iterations] --isFlip [True if using Flipout] --LRT [True if using Local reparameterization trick]
```
