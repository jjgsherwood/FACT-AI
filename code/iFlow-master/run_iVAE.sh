# Remove -c and -p arguments when training on CPU

# Testing default iVAE performance with variable dataseed.
for seed in $(seq 1 20)
do
    python3 main.py \
        -x 1000_40_5_5_3_$seed'_'gauss_xtanh_u_f \
        -i iVAE \
        -c \
        -p \
        -sr data
done

## Testing default iVAE performance with variable net seed.

# for seed in $(seq 1 20)
# do
#     python3 main.py \
#         -x 1000_40_5_5_3_1_gauss_xtanh_u_f \
#         -i iVAE \
#         -s $seed \
#         -c \
#         -p \
#         -sr model
# done