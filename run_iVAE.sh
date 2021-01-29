# Remove -c and -p arguments when training on CPU

# This sh file provides scripts to train the iVAE model on every configuration we tested 
# for the first 20 random seeds while saving the results in the appropriate JSON file.

# Testing default iVAE performance with variable dataseed and fixed network training seed.
for seed in $(seq 1 20)
do
    python main.py \
        -x 1000_40_5_5_3_$seed'_'gauss_xtanh_u_f \
        -i iVAE \
        -c \
        -p \
        -sr data
done

## Testing default iVAE performance with variable net seed and fixed data generation seed.

# for seed in $(seq 1 20)
# do
#     python main.py \
#         -x 1000_40_5_5_3_1_gauss_xtanh_u_f \
#         -i iVAE \
#         -s $seed \
#         -c \
#         -p \
#         -sr model
# done