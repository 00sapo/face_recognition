#!/bin/bash

NUMBER_OF_FILES_PER_ID=40
NUMBER_OF_ID_TO_REMOVE=5
initial_dir = $(pwd)

if [ "$1" == "-h" ] || [ -z $1 ]
then
	echo "usage: $0 <dataset-path> [<NUMBER_OF_FILES_PER_ID> <NUMBER_OF_IDS_TO_REMOVE>]"
	exit 1
fi

dataset_path=$1

if [ ! -d $dataset_path ]
then
	echo "dataset path not found"
	exit 1
fi

if [ ! -z $2]
then
	NUMBER_OF_FILES_PER_ID=$2
fi

if [ ! -z $3]
then
	NUMBER_OF_ID_TO_REMOVE=$3
fi

mkdir testingset
if [ $? -ne 0 ]
then
	echo "moving testingset to testingset.back"
	mv testingset testingset.back
	mkdir testingset
fi

testingpath=$(realpath testingset)

cd $dataset_path
ls ??/frame_*.png > /dev/null
if [ $? -ne 0 ]
then
	echo "dataset inconsistent..."
	exit 1
fi

echo "creating testing set..."
for dir in `ls -d ??/`
do
	mkdir $testingpath/$dir
	cp ${dir}*.cal $testingpath/$dir
	for f in `ls ${dir}frame_*.bin | shuf -n $NUMBER_OF_FILES_PER_ID`
	do
		mv $f $testingpath/$dir
		mv ${f/depth.bin/rgb.png} $testingpath/$dir
	done
done

mkdir removed
if [ $? -ne 0 ]
then
	echo "moving removed to removed.back"
	mv removed removed.back
	mkdir removed
fi
echo "removing identities from training set..."
# this is for extended syntax
shopt -s extglob
for dir in `ls  !(03|15|18|05|21|02|22|07|*.*) -d | shuf -n $NUMBER_OF_ID_TO_REMOVE`
do
	mv $dir removed
	echo $dir >> $initial_dir}/removed_id.txt
done
