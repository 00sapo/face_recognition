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
echo "---------- TESTING WITH RGB AND DEPTH --------------"
./FaceRecognition --train --useRGB --useDepth --loadTrainingset=../preprocessed_trainingset/ --loadValidationset=../preprocessed_validationset/ --query=../testingset/ --idmap=../source/map.txt --unknown=../removed_id.txt
echo "------------ TESTING WITH RGB ONLY -----------------"
./FaceRecognition --train --useRGB --loadTrainingset=../preprocessed_trainingset/ --loadValidationset=../preprocessed_validationset/ --query=../testingset/ --idmap=../source/map.txt --unknown=../removed_id.txt
echo "----------- TESTING WITH DEPTH ONLY ----------------"
./FaceRecognition --train --useDepth --loadTrainingset=../preprocessed_trainingset/ --loadValidationset=../preprocessed_validationset/ --query=../testingset/ --idmap=../source/map.txt --unknown=../removed_id.txt

cd ..
rm -r preprocessed_trainingset preprocessed_validationset
cp -rn ${dataset_path}/removed/* ${dataset_path}/
cp -rn testingset/* ${dataset_path}/
rm -r testingset
rm -r ${dataset_path}/removed
echo "Removed ids: "
cat removed_id.txt
rm removed_id.txt
