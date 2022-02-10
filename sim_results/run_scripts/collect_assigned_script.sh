source ~/.bashrc

cp /home/mrudolph/Documents/multi_agent_learning/team_generation/assigned_teams.json \
/home/mrudolph/Documents/multi_agent_learning/team_generation/teams.json

cd /home/mrudolph/Documents/robotarium_experiments/sim_results
runs=400
echo assigned
python robotarium_results.py \
--task coverage \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_coverage/m_01000 \
--save_path  ./results/assigned \
--show False \
--runs $runs

python robotarium_results.py \
--task navigation \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_navigation/m_01000 \
--save_path  ./results/assigned \
--show False \
--runs $runs

python robotarium_results.py \
--task transport \
--policy_path ~/Documents/multi_agent_learning/expert/learned_rewards/simple_transport/m_01000 \
--save_path  ./results/assigned \
--show False \
--runs $runs

cd -