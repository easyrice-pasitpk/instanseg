source .venv/bin/activate
python instanseg/scripts/train.py \
    --data_path /home/easyricedev0/NAS/M0/Segment_M0_ParboiledRice_WhiteRice_BlueBG/ \
    --dataset ez20251002 \
    --model_path /home/easyricedev0/NAS/M0/Segment_M0_ParboiledRice_WhiteRice_BlueBG/models/instanseg \
    --output_path /home/easyricedev0/NAS/M0/Segment_M0_ParboiledRice_WhiteRice_BlueBG/models/instanseg \
    --experiment_str ez20251002_instseg \
    --device cuda:0 \
    --num_workers 4 \
    --fp16 'true' \
    --batch_size 8 \
    --accumulation_steps 8 \
    --num_epochs 1000 \
    --lr 0.0001 \
    --optimizer adamw \
    --tile_size 512 \
    --window_size 256 \
    --augmentation_type minimal \
    --freeze_main_model 'true' \
    --rng_seed 5555
