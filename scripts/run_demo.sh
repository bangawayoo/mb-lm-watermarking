LP=F

python demo_watermark.py --run_gradio F \
                          --demo_public F \
                          --model_name_or_path 'gpt2-xl' \
                          --prompt_max_length 100 \
                          --max_new_tokens 300 \
                          --gamma 0.5 --detection_z_threshold 1.96 \
                          --load_fp16 $LP