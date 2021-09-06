echo -e "The pretrained models will stored in the 'pretrained_models' folder\n"
gdown "https://drive.google.com/uc?id=1I_Fx48HzxizF7AA5etLrDr0yNViaI_8o"
tar xfzv pretrained_models.tgz

echo -e "Cleaning\n"
rm pretrained_models.tgz

echo -e "Downloading done!"
