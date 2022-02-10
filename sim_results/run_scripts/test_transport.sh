source ~/.bashrc

cd /home/mrudolph/Documents/robotarium_experiments/sim_results
cp /home/mrudolph/Documents/multi_agent_learning/team_generation/random_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json

runs=1

python robotarium_results.py \
--task transport \
--policy_path ~/Documents/multi_agent_learning/expert/simple_transport/checkpoint01000 \
--save_path none \
--show True \
--runs $runs