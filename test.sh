export CUDA_VISIBLE_DEVICES=0

dir=./wap-14

for file in $dir/*.pkl
do
	echo $file
	for year in '14' '16' '19'
	do
	python -u ./translate.py -k 10 $file \
	./dictionary.txt \
	./CROHME/${year}_test_images.pkl \
	./CROHME/${year}_test_labels.txt \
	${dir}/${year}.txt \
	${dir}/${year}.wer
		
	done
done