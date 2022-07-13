

python train.py --gpu 0 --num_dataloader_workers 12 --num_epochs 200 --batch_size 32 --training_set train-augment --model unet++ --negative_sample_probability 0.05 --lr 0.01 --save_most_recent --rotation_augmentation --output_dir output/train-single_unet++_0.05_0.01_rotation
python train.py --gpu 1 --num_dataloader_workers 12 --num_epochs 200 --batch_size 32 --training_set train-augment --model deeplabv3+ --negative_sample_probability 0.05 --lr 0.01 --save_most_recent --rotation_augmentation --output_dir output/train-single_deeplabv3+_0.05_0.01_rotation
python train.py --gpu 2 --num_dataloader_workers 12 --num_epochs 200 --batch_size 32 --training_set train-augment --model manet --negative_sample_probability 0.05 --lr 0.01 --save_most_recent --rotation_augmentation --output_dir output/train-single_manet_0.05_0.01_rotation

python inference_and_evaluate.py --gpu 3 --input_fn data/splits/test-single.csv --model_fn output/train-single_unet++_0.05_0.01_rotation/best_checkpoint.pt --output_fn results/train-augment_unet++_0.05_0.01_rotation.csv --model unet++
python inference_and_evaluate.py --gpu 3 --input_fn data/splits/test-single.csv --model_fn output/train-single_deeplabv3+_0.5_0.01_rotation_run2/best_checkpoint.pt --output_fn results/train-augment_deeplabv3+_0.05_0.01_rotation.csv --model deeplabv3+
python inference_and_evaluate.py --gpu 3 --input_fn data/splits/test-single.csv --model_fn output/train-single_manet_0.05_0.01_rotation/best_checkpoint.pt --output_fn results/train-augment_manet_0.05_0.01_rotation.csv --model manet

python inference.py --input_fn data/splits/test-single.csv --model_fn output/train-single_deeplabv3+_0.5_0.01_rotation_run2/best_checkpoint.pt --output_dir results/train-augment_deeplabv3+_0.05_0.01_rotation/ --gpu 0 --model deeplabv3+
python inference.py --input_fn data/splits/test-single.csv --model_fn output/train-single_manet_0.5_0.01_rotation/best_checkpoint.pt --output_dir results/train-augment_manet_0.05_0.01_rotation/ --gpu 1 --model manet

python postprocess.py --input_fn data/splits/test-single.csv --output_fn results/train-augment_deeplabv3+_0.05_0.01_rotation.geojson --input_dir results/train-augment_deeplabv3+_0.05_0.01_rotation/
python postprocess.py --input_fn data/splits/test-single.csv --output_fn results/train-augment_manet_0.05_0.01_rotation.geojson --input_dir results/train-augment_manet_0.05_0.01_rotation/
