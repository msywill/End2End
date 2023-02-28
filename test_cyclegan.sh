#python test_cyclegan.py  --norm_discriminator spectral --epoch "35" --norm_discriminator spectral --dataroot /home/viktor/data/HighToLow_s7 --name cyclegan_100ep_spectral --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --norm_discriminator spectral --epoch "35" --norm_discriminator spectral --dataroot /home/viktor/data/HighToLow_s8 --name cyclegan_100ep_spectral --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#--load_detection_model
#python test_cyclegan.py --epoch "85" --dataroot /home/viktor/data/d2_1200 --name cyclemedgan_cath_A_100ep --model cycle_medgan --load_size 512 --crop_size 512  --sequence 0 --num_test 1199
#python test_cyclegan.py  --norm_discriminator spectral --epoch "35" --norm_discriminator spectral --dataroot /home/viktor/data/HighToLowFullTest --name cyclegan_100ep_spectral --model cycle_gan --load_size 512 --crop_size 512  --sequence 0 --num_test 1361 --direction BtoA
#python test_cyclegan.py --epoch "latest" --dataroot /home/viktor/data/HighToLowFullTest --name cyclegan_50ep --model cycle_gan --load_size 512 --crop_size 512  --sequence 0 --num_test 1361 --direction BtoA

python test_cyclegan.py  --epoch "10" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted_static_thrsh" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
python test_cyclegan.py  --epoch "10" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted_static_thrsh" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
mv /home/viktor/data/generated_sequences_cyclegan_weighted_static_thrsh /home/viktor/data/generated_sequences_cyclegan_weighted_static_thrsh_10

python test_cyclegan.py  --epoch "16" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted_static_thrsh" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
python test_cyclegan.py  --epoch "16" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted_static_thrsh" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
mv /home/viktor/data/generated_sequences_cyclegan_weighted_static_thrsh /home/viktor/data/generated_sequences_cyclegan_weighted_static_thrsh_16

#python test_cyclegan.py  --epoch "20" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted_static_thrsh" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "20" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted_static_thrsh" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted_static_thrsh /home/viktor/data/generated_sequences_cyclegan_weighted_static_thrsh_20
#
#python test_cyclegan.py  --epoch "26" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted_static_thrsh" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "26" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted_static_thrsh" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted_static_thrsh /home/viktor/data/generated_sequences_cyclegan_weighted_static_thrsh_26

#python test_cyclegan.py  --epoch "40" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted_gaussian_20" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "40" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted_gaussian_20" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted_gaussian_20 /home/viktor/data/generated_sequences_cyclegan_weighted_gaussian_20_40
#
#python test_cyclegan.py  --epoch "66" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted_gaussian_20" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "66" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted_gaussian_20" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted_gaussian_20 /home/viktor/data/generated_sequences_cyclegan_weighted_gaussian_20_66

#python test_cyclegan.py  --epoch "45" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "45" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted /home/viktor/data/generated_sequences_cyclegan_weighted_45
#
#python test_cyclegan.py  --epoch "80" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "80" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted /home/viktor/data/generated_sequences_cyclegan_weighted_80
#
#python test_cyclegan.py  --epoch "100" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "100" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted /home/viktor/data/generated_sequences_cyclegan_weighted_100
#
#python test_cyclegan.py  --epoch "110" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "110" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted /home/viktor/data/generated_sequences_cyclegan_weighted_110
#
#python test_cyclegan.py  --epoch "125" --dataroot /home/viktor/data/HighToLow_s7 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "125" --dataroot /home/viktor/data/HighToLow_s8 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted /home/viktor/data/generated_sequences_cyclegan_weighted_125
#
#python test_cyclegan.py  --epoch "10" --dataroot /home/viktor/data/HighToLow_s7 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "10" --dataroot /home/viktor/data/HighToLow_s8 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cycle_med_gan_weighted /home/viktor/data/generated_sequences_cycle_med_gan_weighted_10
#
#python test_cyclegan.py  --epoch "20" --dataroot /home/viktor/data/HighToLow_s7 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "20" --dataroot /home/viktor/data/HighToLow_s8 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cycle_med_gan_weighted /home/viktor/data/generated_sequences_cycle_med_gan_weighted_20
#
#python test_cyclegan.py  --epoch "30" --dataroot /home/viktor/data/HighToLow_s7 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "30" --dataroot /home/viktor/data/HighToLow_s8 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cycle_med_gan_weighted /home/viktor/data/generated_sequences_cycle_med_gan_weighted_30
#
#python test_cyclegan.py  --epoch "65" --dataroot /home/viktor/data/HighToLow_s7 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "65" --dataroot /home/viktor/data/HighToLow_s8 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cycle_med_gan_weighted /home/viktor/data/generated_sequences_cycle_med_gan_weighted_65
#
#python test_cyclegan.py  --epoch "80" --dataroot /home/viktor/data/HighToLow_s7 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "80" --dataroot /home/viktor/data/HighToLow_s8 --name "cycle_med_gan_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cycle_med_gan_weighted /home/viktor/data/generated_sequences_cycle_med_gan_weighted_80

#python test_cyclegan.py  --epoch "30" --dataroot /home/viktor/data/HighToLow_s7 --name "cycle_med_gan_spectral_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 7 --num_test 1200
#python test_cyclegan.py  --epoch "30" --dataroot /home/viktor/data/HighToLow_s8 --name "cycle_med_gan_spectral_weighted" --model cycle_medgan --load_size 512 --crop_size 512 --sequence 8 --num_test 1200
#mv /home/viktor/data/generated_sequences_cycle_med_gan_spectral_weighted /home/viktor/data/generated_sequences_cycle_med_gan_spectral_weighted_30

#python test_cyclegan.py  --epoch "10" --dataroot /home/viktor/data/d1_1200 --name "cyclegan_50ep" --model cycle_gan --load_size 512 --crop_size 512 --num_test 1200
#python test_cyclegan.py  --epoch "10" --dataroot /home/viktor/data/d1_1200 --name "cyclegan_50ep" --model cycle_gan --load_size 512 --crop_size 512 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_50ep /home/viktor/data/generated_sequences_cyclegan_50ep_10_synthetic_d1_1200
#
python test_cyclegan.py  --epoch "30" --dataroot /home/viktor/data/d1_1200 --name "cyclegan_weighted_gaussian_20" --model cycle_gan --load_size 512 --crop_size 512 --num_test 1200
mv /home/viktor/data/generated_sequences_cyclegan_weighted_gaussian_20 /home/viktor/data/generated_sequences_cyclegan_weighted_gausssian_20_synthetic_d1_1200
python test_cyclegan.py  --epoch "30" --dataroot /home/viktor/data/d2_1200 --name "cyclegan_weighted_gaussian_20" --model cycle_gan --load_size 512 --crop_size 512 --num_test 1200
mv /home/viktor/data/generated_sequences_cyclegan_weighted_gaussian_20 /home/viktor/data/generated_sequences_cyclegan_weighted_gausssian_20_synthetic_d2_1200

#python test_cyclegan.py --epoch "125" --dataroot /home/viktor/data/d1_1200 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --num_test 1200
##python test_cyclegan.py --epoch "125" --dataroot /home/viktor/data/d2_1200 --name "cyclegan_weighted" --model cycle_gan --load_size 512 --crop_size 512 --num_test 1200
#mv /home/viktor/data/generated_sequences_cyclegan_weighted /home/viktor/data/generated_sequences_cyclegan_weighted_synthetic_d1