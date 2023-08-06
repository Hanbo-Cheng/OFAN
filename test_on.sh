export CUDA_VISIBLE_DEVICES=0

dir=./wap_optical-14

for file in $dir/*.pkl
do
	echo $file
	for year in '14' '16' '19'
	do
	python -u ./translate_on.py -k 10 $file \
	./dictionary.txt \
	./CROHME/${year}_test_images.pkl \
	./CROHME/online_flow_test_feature_${year}.pkl \
	./CROHME/${year}_test_labels.txt \
	${dir}/test_decode_result_of_${year}_${file##*wap_}.txt \
	${dir}/test_of_${year}_${file##*wap_}.wer
		
	done
done
