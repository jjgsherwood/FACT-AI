# Remove -c and -p arguments when training on CPU

# Testing the paper's iFLow implementation
for seed in $(seq 1 1)
do
    python main.py \
        -x 1000_40_5_5_3_$seed"_"gauss_xtanh_u_f \
        -c \
        -p \
        -sr data
done

# Testing the paper's iFLow implementation, with fixed dataseed and variable model seed
# for seed in $(seq 1 20)
# do
#     python main.py \
#         -x 1000_40_5_5_3_1_gauss_xtanh_u_f \
#         -s $seed \
#         -c \
#         -p \
#         -sr model
# done