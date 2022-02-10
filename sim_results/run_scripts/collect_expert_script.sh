source ~/.bashrc

cp /home/mrudolph/Documents/multi_agent_learning/team_generation/expert_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json
cd /home/mrudolph/Documents/robotarium_experiments/sim_results

runs=400
echo expert
python robotarium_results.py \
--task coverage \
--policy_path ~/Documents/multi_agent_learning/expert/simple_coverage/checkpoint00900 \
--save_path  ./results/expert \
--show False \
--runs $runs

python robotarium_results.py \
--task navigation \
--policy_path ~/Documents/multi_agent_learning/expert/simple_navigation/checkpoint01000 \
--save_path  ./results/expert \
--show False \
--runs $runs

python robotarium_results.py \
--task transport \
--policy_path ~/Documents/multi_agent_learning/expert/simple_transport/checkpoint00900 \
--save_path  ./results/expert \
--show False \
--runs $runs


cd -