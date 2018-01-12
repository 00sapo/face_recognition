#!/bin/bash
# This script creates a new trainingset-testingset and launch the face recognition program

if [ "$1" == "-h" ] || [ -z $1 ]
then
	echo "usage: $0 <dataset-path>"
	exit 1
fi

dataset_path=$1

if [ ! -d $dataset_path ]
then
	echo "dataset path not found"
	exit 1
fi

# cp -rn $dataset_path $dataset_path
bash ./source/testingset_building.sh ${dataset_path}
cd build-debug
./FaceRecognition --saveTrainingset=../preprocessed_trainingset --saveValidationset=../preprocessed_validationset --dataset=../dataset/hpdb
./FaceRecognition --loadTrainingset=../preprocessed_trainingset/ --loadValidationset=../preprocessed_validationset/ --query=../testingset/

cd ..
rm -r preprocessed_validationset preprocessed_validationset
mv ${dataset_path}/removed/* ${dataset_path}/
mv testingset/* ${dataset_path}/
rm -r testingset
rm -r ${dataset_path}/removed
