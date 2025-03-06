# Update HAPT dataset in the data folder
echo 'Copying HAPT_daghar_like'
cp -r ../data/transitions/processed/HAPT_daghar_like /workspaces/shared/data/transitions/standartized_balanced
echo 'Copying HAPT_concatenated_in_user_files'
cp -r ../data/transitions/processed/HAPT_concatenated_in_user_files /workspaces/shared/data/transitions/standartized_unbalanced

# Update Recodgait dataset in the data folder
cp -r ../data/authentication/processed/RG_daghar_like /workspaces/shared/data/authentication/standartized_balanced
mkdir /workspaces/shared/data/authentication/standartized_unbalanced
cp -r ../data/authentication/processed/RG_concatenated_in_user_files /workspaces/shared/data/authentication/standartized_unbalanced
cp -r ../data/authentication/processed/RG_concatenated_in_user_files_accel_duplicated /workspaces/shared/data/authentication/standartized_unbalanced

