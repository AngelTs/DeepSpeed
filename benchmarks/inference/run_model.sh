set -ex

model=$1

version=0
log_path=results/${model}_v${version}
mkdir -p ${log_path}

echo "baseline $log_path"
deepspeed --num_gpus 1 gpt-bench.py -m "${model}" &> ${log_path}/base.log

cd ../../
git checkout staging-ds-azure-v1
cd -
echo "ds azure $log_path"
deepspeed --num_gpus 1 gpt-bench.py --deepspeed -m "${model}" &> ${log_path}/dsAzure.log

cd ../../
git checkout master
cd -
echo "ds public $log_path"
deepspeed --num_gpus 1 gpt-bench.py --deepspeed -m "${model}" &> ${log_path}/dsPublic.log
