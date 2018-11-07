#!/bin/bash

if [ -z "$1" ]; then
    echo "team name is required.
Usage: ./CollectSubmission team_*"
    exit 0
fi

files="./model_checkpoints/*.meta
Assignment1-1_Data_Curation.ipynb
Assignment1-2_NN_from_scratch.ipynb
Assignment1-3_NN_with_TF.ipynb"

for file in $files
do
    if [ ! -f $file ]; then
        echo "Required $file not found."
        exit 0
    fi

    if [ ${file: -6} == ".ipynb" ]

    then
        echo "Converting $file to python file"
        jupyter nbconvert --to python "$file"
    fi
done


rm -f $1.tar.gz
mkdir $1
cp -r ./model_checkpoints ./*.py $1/
tar cvzf $1.tar.gz $1
