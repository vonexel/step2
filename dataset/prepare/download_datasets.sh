cd dataset

echo -e "Downloading HumanML3D & KIT-ML datasets"
gdown --fuzzy https://drive.google.com/file/d/148MT29T50D-zudCzQjDqu35T8PI7tFOq/view?usp=drive_link
gdown --fuzzy https://drive.google.com/file/d/1_dCSSpzgUySxwWGeCMfKjYHyo-Zm_YzL/view?usp=drive_link

mkdir -p HumanML3D KIT-ML
echo -e "Extracting datasets into dedicated directories"
unzip HumanML3D.zip -d HumanML3D/
unzip KIT-ML.zip -d KIT-ML/

echo -e "Cleaning\n"
rm HumanML3D.zip KIT-ML.zip
echo -e "Completed!"