
# Set -i to iVAE for iVAE and iFlow for iFlow

# for seed in $(seq 1 100)  (use this for all seeds)
# for seed in $(seq 1 1)
# do
#     python3 main.py \
#         -x 1000_40_5_5_3_$seed'_'gauss_xtanh_u_f \
#         -i iFlow \
#         -ft RQNSF_AG \
#         -npa Softplus \
#         -fl 10 \
#         -lr_df 0.25 \
#         -lr_pn 10 \
#         -b 64 \
#         -e 20 \
#         -l 1e-3 \
#         -s 1 \
#         -u 0 \
#         -c \
#         -nph orig
# done

for seed in $(seq 1 1)
do
    python main.py \
        -x 1000_40_5_5_3_$seed"_"gauss_xtanh_u_f \
        -i iFlow \
        -ft Real_NVP \
        -npa Softplus \
        -fl 10 \
        -lr_df 0.25 \
        -lr_pn 10 \
        -b 64 \
        -e 20 \
        -l 1e-3 \
        -s 1 \
        -u 0 \
        -c \
        -nph orig \
        -p 
done

#python main.py \
#    -x 1000_40_5_5_3_1_gauss_xtanh_u_f \
#    -i iFlow \
#    -fl 10 \
#    -lr_df 0.5 \
#    -lr_pn 10 \
#    -b 64 \
#    -e 20 \
#    -l 1e-3 \
#    -s 1 \
#    -u 6 \
#    -c

#python main.py \
#    -x 100000_40_5_5_3_1_gauss_xtanh_u_f \
#    -i iFlow \
#    -fl 10 \
#    -lr_df 0.5 \
#    -lr_pn 10 \
#    -b 10000 \
#    -e 20 \
#    -l 1e-3 \
#    -s 1 \
#    -u 0 \
#    -c \
#    -p
