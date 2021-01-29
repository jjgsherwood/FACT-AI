# Remove -c and -p arguments when training on CPU

# This sh file provides scripts to train the iFlow model on every configuration we tested 
# for the first 20 random seeds while saving the results in the appropriate JSON file.

# Testing the paper's iFLow implementation for the first 20 seeds while saving the results in JSON file.
for seed in $(seq 1 20)
do
    python main.py \
        -x 1000_40_5_5_3_$seed"_"gauss_xtanh_u_f \
        -c \
        -p \
        -sr data
done

# # Testing the paper's iFLow implementation, with fixed dataseed and variable model seed
# for seed in $(seq 1 20)
# do
#     python main.py \
#         -x 1000_40_5_5_3_1_gauss_xtanh_u_f \
#         -s $seed \
#         -c \
#         -p \
#         -sr model
# done


# # Testing the "improved" iFlow model, as suggested by the code comment from the original authors.
# for seed in $(seq 1 20)
# do
#     python main.py \
#         -x 1000_40_5_5_3_$seed"_"gauss_xtanh_u_f \
#         -nph fixed \
#         -c \
#         -p \
#         -sr data
# done


# # Testing regular Flow network (so with the lambda(u) network removed)
# for seed in $(seq 1 20)
# do
#     python main.py \
#         -x 1000_40_5_5_3_$seed"_"gauss_xtanh_u_f \
#         -nph removed \
#         -c \
#         -p \
#         -sr data
# done

# # Testing iFlow with PlanarFlow instead of the original Cubic Spline flow
# for seed in $(seq 1 20)
# do
#     python main.py \
#         -x 1000_40_5_5_3_$seed"_"gauss_xtanh_u_f \
#         -ft PlanarFlow \
#         -c \
#         -p \
#         -sr data
# done

# # Testing regular Flow network (so with the lambda(u) network removed) 
# # with PlanarFlow instead of the original Cubic Spline flow as flow network
# for seed in $(seq 1 20)
# do
#     python main.py \
#         -x 1000_40_5_5_3_$seed"_"gauss_xtanh_u_f \
#         -ft PlanarFlow \
#         -nph removed \
#         -c \
#         -p \
#         -sr data
# done


