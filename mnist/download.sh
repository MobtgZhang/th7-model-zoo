result_dir="./data"
file_flag=false
if [ -d $result_dir  ]; then
    echo "The directory ${result_dir} exists!"
else
    mkdir $result_dir
    file_flag=true
fi
if [ "$(ls $result_dir)" ]; then
    echo "The directory ${result_dir} files exists!"
else
    wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O $result_dir/train-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O $result_dir/train-labels-idx1-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O $result_dir/t10k-images-idx3-ubyte.gz
    wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O $result_dir/t10k-labels-idx1-ubyte.gz      
fi
echo "Uncompress files"
for filename in `ls ${result_dir}/*.gz`; do
    echo $filename
    gunzip -dv $filename 
done

