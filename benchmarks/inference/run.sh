set -ex

model="EleutherAI/gpt-neo-2.7B"

version=0
log_path=results/${model}_v${version}
mkdir -p ${log_path}

echo "baseline"
deepspeed --num_gpus 1 gpt-bench.py -m "${model}" &> ${log_path}/base.log

cd ../../
git checkout staging-ds-azure-v1
cd -
echo "ds azure"
deepspeed --num_gpus 1 gpt-bench.py --deepspeed -m "${model}" &> ${log_path}/dsAzure.log

cd ../../
git checkout master
cd -
echo "ds public"
deepspeed --num_gpus 1 gpt-bench.py --deepspeed -m "${model}" &> ${log_path}/dsPublic.log

