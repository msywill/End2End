python -m pytorch_fid /home/viktor/catheter-CycleGAN/catheter-CycleGAN/HighToLow_s7/testB /home/viktor/catheter-CycleGAN/cyclegan_50ep/s7 --device cuda:0
# commandline python -m pytorch_fid path/to/dataset1 path/to/dataset2
# real low-dose /home/viktor/catheter-CycleGAN/catheter-CycleGAN/HighToLow_s7/testB
# fake low-dose /home/viktor/catheter-CycleGAN/cyclegan_50ep/s7